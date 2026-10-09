# Turkish-TTS

Türkçe metin-konuşma (TTS). Trendyol'un VoxCPM2 üzerine ince ayarladığı ağırlıkları hem
kütüphane hem HTTP/WebSocket servisi olarak sunar, ve geliştiricinin kendi verisiyle LoRA
ince ayarı çekmesine izin verir.

Hedef ortam **sunucu GPU'su**. Tek GPU/laptop yalnızca geliştirme ve test içindir: model 2B
parametre ve ~8 GB VRAM istiyor (`src/turkish_tts/_vendor/voxcpm/README.upstream.md`).

## Dizin yapısı

| Yol | Ne |
|---|---|
| `src/turkish_tts/__init__.py` | Genel yüzey: `TurkishTTS` (`stream`, `synthesize`, `save`, `finetune`) |
| `src/turkish_tts/settings.py` | Ayarlar. `TTS_<BOLUM>_<ALAN>` env > JSON dosya > varsayılan |
| `src/turkish_tts/engine.py` | GPU havuzu (`VoxCPMPool`), `acquire()` ile kiralama |
| `src/turkish_tts/finetune.py` | LoRA ince ayarı sarmalayıcısı |
| `src/turkish_tts/weights.py` | Ağırlık indirme |
| `src/turkish_tts/api/` | FastAPI uygulaması ve rotalar |
| `src/turkish_tts/_vendor/voxcpm/` | Upstream VoxCPM'in vendor'lanmış, **değiştirilmiş** kopyası (Apache-2.0) |
| `Trendyol-TTS/` | Referans: model kartı, tokenizer, `merge_manifest.json`. Ağırlık içermez |
| `tests/`, `tests/vendor/` | Kendi testlerimiz ve korunan upstream testleri |
| `benchmarks/bench.py` | Sunucu ölçümü (TTFB, RTF, p95, tepe VRAM) |

İstek akışı:

```
rota → VoxCPMPool.acquire()        # havuzdan bir GPU modeli kirala (timeout'lu)
     → synthesize_stream(model, text)
     → sentence_source(text)       # TextBuffer ile cümlelere böl
     → VoxCPM.generate_stream_from_text_source()
     → AudioChunk akışı (PCM16-LE, 48 kHz)
```

## Komutlar

```bash
uv pip install -e ".[dev]"            # geliştirme kurulumu
uv pip install -e ".[finetune]"       # ince ayar da gerekiyorsa
turkish-tts-fetch                     # ağırlıkları HF önbelleğine indir
turkish-tts-server                    # sunucu, 0.0.0.0:8000
python -m pytest tests/               # testler, model gerektirmez
python benchmarks/bench.py --concurrency 1,2,4 --out benchmarks/baseline.json
docker compose -f docker/compose.yaml up
```

Kütüphane olarak:

```python
from turkish_tts import TurkishTTS

tts = TurkishTTS()
tts.save("Merhaba dünya.", "out.wav")

TurkishTTS.finetune("data/train.jsonl", "runs/benim-sesim", steps=1000)
```

## Kurallar

**Ağırlıklar git'te değil.** `.gitignore` `*.safetensors`/`*.pth` vb. dışlıyor. Ağırlıklar HF
önbelleğine iner (`turkish-tts-fetch`). `Trendyol-TTS/` yalnızca referans config+tokenizer'dır;
ağırlık dosyası eklemeye çalışma.

**`_vendor/voxcpm/` vendor'lanmış ve Apache-2.0.** Orada bir dosyayı değiştirir veya eklersen
`LICENSE-NOTICE.md`'deki değiştirilen/eklenen dosya listesini güncelle — Apache-2.0 §4(b) bunu
zorunlu kılıyor. Değiştirilen dosyalar `# NOTICE:` başlığı taşır; silme. Vendor'a dokunmadan
çözülebilecek bir şeyi vendor'da çözme — örneğin ek veri doğrulaması `finetune.py`'ye yazılır.

**Ölçmeden optimize etme.** Performans değişikliği için önce `benchmarks/baseline.json`'a karşı
ölçüm gerekir. Daha önce bir commit (`c13b2c9`) "takılma"yı ölçmeden config'i yavaşlatarak
çözmeye çalıştı ve asıl sebep (havuz sızıntısı) gözden kaçtı. Model ayarları için
`Trendyol-TTS/merge_manifest.json` içindeki `clean_default` ve `avoid_as_general_default`
alanları bağlayıcıdır.

**Commit mesajları Conventional Commits.** Sürümleme bunlardan türetiliyor:
`feat!` major, `feat` minor, `fix`/`perf`/`docs`/`refactor`/`style`/`build` yama,
`chore`/`ci`/`test` yayın üretmez. Sürüm numarasını elle düzenleme — `pyproject.toml`
ve `__version__` yayın akışı tarafından yazılır.

**Dil.** Kod, tanımlayıcılar ve docstring'ler İngilizce; log mesajları ve satır içi yorumlar
Türkçe. Her dosyanın ve her fonksiyonun tek cümlelik docstring'i olur. Log ve başlıklarda
emoji kullanma.

## Tuzaklar

- **`torch.compile` triton yoksa sessizce atlanıyor.** `_vendor/voxcpm/model/voxcpm2.py`
  içindeki `optimize()` triton import'unu deniyor; Windows'ta triton yok, kazanç sıfırdır.
  Gerçek compile yalnızca Linux sunucu imajında geçerli.
- **Çıkış örnekleme hızı 48000, 24000 değil.** VAE 16 kHz alır, 48 kHz verir
  (`Trendyol-TTS/config.json` → `audio_vae_config`).
- **Ayarlar import anında okunur.** `settings` tek seferlik; çalışırken yeniden yüklenmez.
  Değişiklik sunucu yeniden başlatması ister.
- **Varsayılan `pip install torch` Windows'ta CPU wheel'i çeker.** CUDA için index-url gerekir;
  aksi halde `_resolve_devices()` açılışta hata verir.
- **v1 model kopyaları hâlâ duruyor.** `model/voxcpm.py`, `audiovae/audio_vae.py`,
  `locdit/local_dit.py` import ediliyor ama VoxCPM2 checkpoint'i için hiç örneklenmiyor.
  Silinmeleri `torchaudio` bağımlılığını da düşürür; gerçek bir çıkarım testi yapılabilene
  kadar bekletiliyor.
