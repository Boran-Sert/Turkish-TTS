"""Measures server-side TTS latency and throughput over the WebSocket endpoint."""

import argparse
import asyncio
import json
import pathlib
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone

import websockets

BYTES_PER_SAMPLE = 2
DEFAULT_URL = "ws://127.0.0.1:8000/v1/tts/stream"

TEXTS = [
    ("kisa", "Merhaba, nasılsınız?"),
    ("orta", "Siparişiniz kargoya verildi ve tahmini teslim tarihi yarın olarak görünüyor."),
    (
        "uzun",
        "Hesaplarınız, kartlarınız, para transferleriniz ve ödemelerinizle ilgili konularda "
        "size hızlı ve güvenli bir şekilde yardımcı olmak için buradayım. Yapmak istediğiniz "
        "işlemi doğal bir şekilde yazabilir veya sorabilirsiniz.",
    ),
]


class VramSampler:
    """Samples whole-GPU memory use in the background while the benchmark runs."""

    def __init__(self, interval: float = 0.25):
        """Prepares a sampler that polls nvidia-smi at the given interval."""
        self._interval = interval
        self._stop = threading.Event()
        self._thread = None
        self.peak_mb = 0
        self.gpu_name = _gpu_name()

    def _loop(self):
        """Polls GPU memory until stopped, keeping the maximum."""
        while not self._stop.wait(self._interval):
            used = _gpu_memory_mb()
            if used is not None:
                self.peak_mb = max(self.peak_mb, used)

    def __enter__(self):
        """Starts sampling."""
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        """Stops sampling."""
        self._stop.set()
        self._thread.join(timeout=2.0)


def _nvidia_smi(query: str) -> str:
    """Runs one nvidia-smi query and returns its first line, or an empty string."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def _gpu_name() -> str:
    """Returns the first GPU's name, or 'bilinmiyor'."""
    return _nvidia_smi("name") or "bilinmiyor"


def _gpu_memory_mb():
    """Returns the first GPU's used memory in MiB, or None."""
    value = _nvidia_smi("memory.used")
    return int(value) if value.isdigit() else None


async def one_request(websocket, text: str) -> dict:
    """Sends one synthesis request and times the first byte and the whole stream."""
    sent_at = time.perf_counter()
    await websocket.send(json.dumps({"text": text}))

    first_byte_at = None
    audio_bytes = 0
    sample_rate = None

    while True:
        message = await websocket.recv()
        if isinstance(message, bytes):
            if first_byte_at is None:
                first_byte_at = time.perf_counter()
            audio_bytes += len(message)
            continue

        event = json.loads(message)
        name = event.get("event")
        if name == "stream_start":
            sample_rate = event["sample_rate"]
        elif name == "stream_end":
            break
        elif name == "error":
            raise RuntimeError(f"{event.get('error_type')}: {event.get('message')}")

    finished_at = time.perf_counter()
    audio_seconds = audio_bytes / (sample_rate * BYTES_PER_SAMPLE)
    wall_seconds = finished_at - sent_at

    return {
        "ttfb_ms": (first_byte_at - sent_at) * 1000 if first_byte_at else None,
        "wall_seconds": wall_seconds,
        "audio_seconds": audio_seconds,
        "rtf": wall_seconds / audio_seconds if audio_seconds else None,
        "audio_bytes": audio_bytes,
        "sample_rate": sample_rate,
    }


async def client_session(url: str, repeats: int) -> list:
    """Runs every benchmark text `repeats` times over a single connection."""
    results = []
    async with websockets.connect(url, max_size=None) as websocket:
        for _ in range(repeats):
            for label, text in TEXTS:
                measurement = await one_request(websocket, text)
                measurement["text"] = label
                results.append(measurement)
        await websocket.send(json.dumps({"event": "close"}))
    return results


def percentile(values: list, fraction: float) -> float:
    """Returns the value at the given fraction of a sorted sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def summarize(values: list) -> dict:
    """Reduces a sample to median, p95 and mean."""
    clean = [v for v in values if v is not None]
    if not clean:
        return {"p50": None, "p95": None, "mean": None}
    return {
        "p50": round(statistics.median(clean), 3),
        "p95": round(percentile(clean, 0.95), 3),
        "mean": round(statistics.fmean(clean), 3),
    }


async def run_level(url: str, concurrency: int, repeats: int) -> dict:
    """Drives `concurrency` parallel clients and aggregates their measurements."""
    started = time.perf_counter()
    batches = await asyncio.gather(
        *(client_session(url, repeats) for _ in range(concurrency))
    )
    elapsed = time.perf_counter() - started

    flat = [m for batch in batches for m in batch]
    audio_total = sum(m["audio_seconds"] for m in flat)

    return {
        "concurrency": concurrency,
        "requests": len(flat),
        "wall_seconds": round(elapsed, 3),
        "audio_seconds_total": round(audio_total, 3),
        "throughput_audio_per_wall": round(audio_total / elapsed, 3) if elapsed else None,
        "ttfb_ms": summarize([m["ttfb_ms"] for m in flat]),
        "rtf": summarize([m["rtf"] for m in flat]),
    }


def local_config() -> dict:
    """Reads the model settings the local tts_config.json declares."""
    path = pathlib.Path(__file__).resolve().parent.parent / "tts_config.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    return {
        "inference_timesteps": data["model"]["inference_timesteps"],
        "cfg_value": data["model"]["cfg_value"],
        "max_length": data["model"].get("max_length"),
        "chunk_duration_ms": data["streaming"]["chunk_duration_ms"],
        "min_chars": data["text_processing"]["min_chars"],
        "enable_lookbehind": data["streaming"]["enable_lookbehind"],
        "use_torch_compile": data["system"]["use_torch_compile"],
        "voxcpm_gpu_ids": data["system"]["voxcpm_gpu_ids"],
    }


async def measure(url: str, levels: list, repeats: int) -> dict:
    """Runs the whole sweep and returns the full report."""
    with VramSampler() as vram:
        runs = []
        for concurrency in levels:
            print(f"[*] eszamanlilik={concurrency} ...", flush=True)
            runs.append(await run_level(url, concurrency, repeats))

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "gpu": vram.gpu_name,
        "peak_vram_mb": vram.peak_mb,
        "repeats": repeats,
        "config": local_config(),
        "runs": runs,
    }


def print_report(report: dict) -> None:
    """Prints one report as a table."""
    print(f"\nGPU: {report['gpu']}   tepe VRAM: {report['peak_vram_mb']} MiB")
    print(f"config: {json.dumps(report['config'], ensure_ascii=False)}")
    print(f"\n{'eszaman':>8} {'istek':>6} {'TTFB p50':>10} {'TTFB p95':>10} {'RTF p50':>9} {'RTF p95':>9} {'verim':>7}")
    for run in report["runs"]:
        print(
            f"{run['concurrency']:>8} {run['requests']:>6} "
            f"{run['ttfb_ms']['p50']:>10} {run['ttfb_ms']['p95']:>10} "
            f"{run['rtf']['p50']:>9} {run['rtf']['p95']:>9} "
            f"{run['throughput_audio_per_wall']:>7}"
        )
    print("\nRTF = duvar saati / uretilen ses suresi; kucuk olan iyi.")


def print_comparison(baseline: dict, current: dict) -> None:
    """Prints baseline against current, with the change for each level."""
    by_level = {run["concurrency"]: run for run in baseline["runs"]}
    print(f"\n{'eszaman':>8} {'TTFB p50':>22} {'RTF p50':>22}")
    print(f"{'':>8} {'onceki -> simdi':>22} {'onceki -> simdi':>22}")
    for run in current["runs"]:
        old = by_level.get(run["concurrency"])
        if not old:
            continue
        cells = []
        for key in ("ttfb_ms", "rtf"):
            before, after = old[key]["p50"], run[key]["p50"]
            change = f"{(after - before) / before * 100:+.1f}%" if before else "n/a"
            cells.append(f"{before} -> {after} ({change})")
        print(f"{run['concurrency']:>8} {cells[0]:>22} {cells[1]:>22}")
    print(f"\ntepe VRAM: {baseline['peak_vram_mb']} -> {current['peak_vram_mb']} MiB")


def main() -> None:
    """Parses arguments, then measures or compares."""
    parser = argparse.ArgumentParser(description="Turkish TTS sunucu benchmark'i")
    parser.add_argument("--url", default=DEFAULT_URL, help="WebSocket adresi")
    parser.add_argument("--concurrency", default="1,2,4", help="virgulle ayrilmis eszamanlilik seviyeleri")
    parser.add_argument("--repeats", type=int, default=3, help="her istemcinin metin setini kac kez kosacagi")
    parser.add_argument("--out", help="raporun yazilacagi JSON dosyasi")
    parser.add_argument("--compare", help="karsilastirilacak baseline JSON dosyasi")
    args = parser.parse_args()

    levels = [int(part) for part in args.concurrency.split(",") if part.strip()]
    report = asyncio.run(measure(args.url, levels, args.repeats))
    print_report(report)

    if args.compare:
        baseline = json.loads(pathlib.Path(args.compare).read_text(encoding="utf-8"))
        print_comparison(baseline, report)

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nrapor yazildi: {args.out}")


if __name__ == "__main__":
    main()
