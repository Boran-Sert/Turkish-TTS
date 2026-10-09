# Entegrasyon Kılavuzu

Turkish-TTS servisinin HTTP ve WebSocket yüzeyi. Kurulum ve yapılandırma için
[README.md](README.md).

Bütün ses çıktısı **tek kanal, 16 bit işaretli, little-endian PCM, 48000 Hz**. İkili
çerçeveler ham PCM taşır, başlık içermez.

## Hangi yolu seçmeli

| Durum | Yol |
|---|---|
| Dosya üret, gecikme önemsiz | `POST /v1/tts` |
| Tarayıcıda veya `curl` ile çalarken akış | `POST /v1/tts/stream` |
| En düşük gecikme, cümle olayları, tek bağlantıda çok istek | `WS /v1/tts/stream` |

## HTTP

### POST /v1/tts

```http
POST /v1/tts
Content-Type: application/json

{"text": "Merhaba, bugün nasılsınız?", "voice": "kadın"}
```

Tam bir WAV dosyası döner (`Content-Type: audio/wav`). Metin 1-5000 karakter olmalı.
`voice` isteğe bağlıdır; verilmezse modelin varsayılan sesi kullanılır. Bilinmeyen bir
ses adı `422` döner ve hata mesajı mevcut sesleri listeler.

### GET /v1/voices

Klonlama için kullanılabilir sesleri döner:

```json
{"voices": [{"name": "kadın", "has_transcript": true}]}
```

Sesler sunucudaki `voices_dir` dizininden okunur; bir ses eklemek için o dizine bir wav
koymak yeterlidir.

### POST /v1/tts/stream

Aynı gövde. Önce 44 baytlık WAV başlığı, ardından üretildikçe PCM çerçeveleri gönderilir.
Başlıktaki uzunluk alanları bilinmediği için azami değerde bırakılır; dosya olarak
kaydedecekseniz `POST /v1/tts` kullanın.

### GET /health ve GET /ready

`/health` her zaman `200 {"status":"ok"}` döner; model yüklenirken de cevap verir, canlılık
yoklaması için bunu kullanın.

`/ready` havuz hazır olana kadar `503` döner:

```json
{"ready": true, "pool_size": 2, "idle": 1}
```

Kubernetes'te `/health` livenessProbe, `/ready` readinessProbe olmalı. Model yüklenmesi
dakikalar sürebileceği için readinessProbe'a geniş bir `initialDelaySeconds` verin.

### HTTP durum kodları

| Kod | Anlamı | Ne yapmalı |
|---|---|---|
| `200` | Ses döndü | - |
| `422` | Gövde geçersiz (boş metin, 5000 karakter aşımı) | İsteği düzeltin |
| `503` | Bütün modeller meşgul veya havuz henüz hazır değil | Geri çekilip tekrar deneyin |
| `500` | Sentez hata verdi | Sunucu loglarına bakın |

## WebSocket

`ws://host:8000/v1/tts/stream`

Tek bağlantıda istediğiniz kadar istek gönderebilirsiniz. Model yalnızca bir isteğin
üretimi boyunca ayrılır, istekler arasında serbest kalır — uzun ömürlü bir bağlantı GPU
tutmaz.

### Akış

```
istemci -> {"text": "Merhaba dünya. İkinci cümle.", "voice": "kadın"}
sunucu  <- {"event":"stream_start","format":"pcm16_le","sample_rate":48000}
sunucu  <- <ikili PCM çerçevesi>            (birçok kez)
sunucu  <- {"event":"sentence_complete","sentence_index":0}
sunucu  <- <ikili PCM çerçevesi>
sunucu  <- {"event":"sentence_complete","sentence_index":1}
sunucu  <- {"event":"stream_end"}

istemci -> {"text": "Başka bir istek."}      (aynı bağlantıda tekrar)
...
istemci -> {"event":"close"}
sunucu  <- kapanış kodu 1000
```

### Gönderilen çerçeveler

| Alan | Değer |
|---|---|
| `{"text": "..."}` | Sentezlenecek metin. Boş veya yalnızca boşluk olmamalı |
| `{"voice": "..."}` | İsteğe bağlı ses adı. Her istekte ayrı verilebilir |
| `{"event": "close"}` | Oturumu düzgün kapat |

Bağlantıyı doğrudan koparmak da güvenlidir; sunucu ayrılmış modeli her durumda bırakır.

### Alınan çerçeveler

**`stream_start`** — ses çerçeveleri başlamadan önce bir kez.

```json
{"event": "stream_start", "format": "pcm16_le", "sample_rate": 48000}
```

Örnekleme hızını sabit yazmayın, bu alandan okuyun.

**İkili çerçeve** — ham PCM16-LE. Boyut sabit değildir;
`streaming.chunk_duration_ms` ayarına göre değişir.

**`sentence_complete`** — bir cümlenin sesi tamamlandığında.

```json
{"event": "sentence_complete", "sentence_index": 0}
```

**`stream_end`** — bu isteğin sesi bitti. Bağlantı açık kalır, yeni istek gönderebilirsiniz.

**`error`** — her hata çerçevesi `error_type` **ve** `message` taşır; iki alanı da
koşulsuz okuyabilirsiniz.

```json
{"event": "error", "error_type": "PoolBusy", "message": "Tüm modeller meşgul (1 adet), 30.0 saniyede boşalmadı."}
```

| `error_type` | Kapanış kodu | Sebep |
|---|---|---|
| `ValidationError` | `1008` | `text` boş veya eksik |
| `VoiceNotFound` | `1008` | `voice` adı sunucuda yok |
| `PoolBusy` | `1013` | Zaman aşımı içinde GPU boşalmadı; geri çekilip tekrar deneyin |
| `IdleTimeout` | `1000` | `ws_idle_timeout_s` boyunca mesaj gelmedi |
| diğer | `1011` | Sunucu tarafı hata |

## İstemci örnekleri

### Python

```python
import asyncio
import json
import websockets

async def seslendir(text: str) -> bytes:
    """Bir metni seslendirip ham PCM16 döner."""
    pcm = bytearray()
    async with websockets.connect("ws://127.0.0.1:8000/v1/tts/stream") as ws:
        await ws.send(json.dumps({"text": text}))
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm += message
                continue
            event = json.loads(message)
            if event["event"] == "stream_end":
                break
            if event["event"] == "error":
                raise RuntimeError(f"{event['error_type']}: {event['message']}")
        await ws.send(json.dumps({"event": "close"}))
    return bytes(pcm)

asyncio.run(seslendir("Merhaba dünya."))
```

Çalışan, hoparlörden çalan tam örnek: [`examples/client.py`](examples/client.py).
`--play` için `pyaudio` gerekir, paketin bağımlılığı değildir.

### C# (.NET)

```csharp
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;

async Task<byte[]> Seslendir(string text, CancellationToken ct)
{
    using var ws = new ClientWebSocket();
    await ws.ConnectAsync(new Uri("ws://127.0.0.1:8000/v1/tts/stream"), ct);

    var request = JsonSerializer.SerializeToUtf8Bytes(new { text });
    await ws.SendAsync(request, WebSocketMessageType.Text, true, ct);

    using var pcm = new MemoryStream();
    var buffer = new byte[64 * 1024];

    while (true)
    {
        var result = await ws.ReceiveAsync(buffer, ct);

        if (result.MessageType == WebSocketMessageType.Binary)
        {
            pcm.Write(buffer, 0, result.Count);
            continue;
        }

        using var doc = JsonDocument.Parse(Encoding.UTF8.GetString(buffer, 0, result.Count));
        var name = doc.RootElement.GetProperty("event").GetString();

        if (name == "stream_end") break;
        if (name == "error")
            throw new InvalidOperationException(
                $"{doc.RootElement.GetProperty("error_type").GetString()}: " +
                doc.RootElement.GetProperty("message").GetString());
    }

    await ws.CloseAsync(WebSocketCloseStatus.NormalClosure, null, ct);
    return pcm.ToArray();
}
```

Uzun metinlerde tek bir `ReceiveAsync` çağrısı bütün mesajı almayabilir; üretimde
`result.EndOfMessage` bitene kadar okumaya devam edin.

### Java (OkHttp)

```java
OkHttpClient client = new OkHttpClient();
Request request = new Request.Builder()
        .url("ws://127.0.0.1:8000/v1/tts/stream")
        .build();

ByteArrayOutputStream pcm = new ByteArrayOutputStream();

client.newWebSocket(request, new WebSocketListener() {
    @Override public void onOpen(WebSocket ws, Response response) {
        ws.send("{\"text\":\"Merhaba dünya.\"}");
    }

    @Override public void onMessage(WebSocket ws, ByteString bytes) {
        try { pcm.write(bytes.toByteArray()); } catch (IOException ignored) { }
    }

    @Override public void onMessage(WebSocket ws, String text) {
        JSONObject event = new JSONObject(text);
        switch (event.getString("event")) {
            case "stream_end" -> ws.send("{\"event\":\"close\"}");
            case "error" -> throw new IllegalStateException(
                    event.getString("error_type") + ": " + event.getString("message"));
            default -> { }
        }
    }
});
```

### Tarayıcı (Web Audio)

```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/v1/tts/stream");
ws.binaryType = "arraybuffer";

const audio = new AudioContext();
let sampleRate = 48000;
let playAt = 0;

ws.onopen = () => ws.send(JSON.stringify({ text: "Merhaba dünya." }));

ws.onmessage = (message) => {
  if (typeof message.data !== "string") {
    const pcm = new Int16Array(message.data);
    const buffer = audio.createBuffer(1, pcm.length, sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) channel[i] = pcm[i] / 32768;

    const source = audio.createBufferSource();
    source.buffer = buffer;
    source.connect(audio.destination);
    playAt = Math.max(playAt, audio.currentTime);
    source.start(playAt);
    playAt += buffer.duration;
    return;
  }

  const event = JSON.parse(message.data);
  if (event.event === "stream_start") sampleRate = event.sample_rate;
  if (event.event === "error") console.error(event.error_type, event.message);
  if (event.event === "stream_end") ws.send(JSON.stringify({ event: "close" }));
};
```

Çalışan tam örnek: [`src/turkish_tts/api/static/index.html`](src/turkish_tts/api/static/index.html).

## LLM çıktısını seslendirme

Bir dil modelinin ürettiği metni seslendirirken iki yol var.

**Cümle cümle gönderin (önerilen).** LLM'den gelen token'ları kendi tarafınızda biriktirin,
cümle sonunu gördükçe aynı WebSocket bağlantısı üzerinden bir `{"text": ...}` gönderin.
Bağlantı tek, istekler sıralı; her `stream_end` sonrası bir sonraki cümleyi yollayın.

**Tamamını bekleyin.** LLM bitince tek istek gönderin. Daha basit, ilk sese kadar süre
LLM'in toplam süresi kadar uzar.

Sunucu şu an token akışını kendi içinde tamponlamıyor; cümleye bölmeyi istemci yapar.

## Üretim notları

- **Kimlik doğrulama yoktur.** Servisi doğrudan internete açmayın; önüne kimlik doğrulama
  ve hız sınırlaması yapan bir ağ geçidi koyun.
- **Eşzamanlılık GPU sayısı kadardır.** VoxCPM2 toplu işleme desteklemiyor; bir model aynı
  anda bir istek üretir. `system.voxcpm_gpu_ids` ile model örneği sayısını belirlersiniz.
- **503 ve 1013'ü geri çekilerek karşılayın.** Bunlar aşırı yüklenme sinyalidir, hata değil.
- **CORS varsayılanı `["*"]`.** Üretimde `api.cors_allowed_origins` ile daraltın.
- **Örnekleme hızını sabit yazmayın.** `stream_start` çerçevesinden okuyun.
