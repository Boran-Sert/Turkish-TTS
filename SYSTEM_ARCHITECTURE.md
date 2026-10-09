# Mimari ve Tasarım Kararları

Bu belge sistemin neden böyle kurulduğunu anlatır. Kullanım için [README.md](README.md),
protokol için [API_INTEGRATION_GUIDE.md](API_INTEGRATION_GUIDE.md).

## Bileşenler

```
turkish_tts/
  __init__.py     TurkishTTS: senkron kütüphane yüzeyi
  engine.py       VoxCPMPool: GPU başına bir model örneği
  settings.py     env > JSON > varsayılan
  finetune.py     LoRA ince ayarı sarmalayıcısı
  api/            FastAPI: HTTP ve WebSocket rotaları
  _vendor/voxcpm/ VoxCPM2'nin budanmış, değiştirilmiş kopyası (Apache-2.0)
```

Model `Trendyol/Trendyol-TTS`: VoxCPM2 tabanına 20+ saat özel Türkçe veriyle çekilmiş LoRA
**ağırlıklara merge edilmiş** durumda (`Trendyol-TTS/merge_manifest.json`). Çalışma anında
adaptör yüklenmez.

## Üretim yolu

```
istek
  -> VoxCPMPool.acquire()                  havuzdan model kirala (timeout'lu)
  -> sentence_source(text)                 TextBuffer ile cümlelere böl
  -> generate_stream_from_text_source()    cümle cümle otoregresif üretim
  -> AudioVAE streaming decode             16 kHz latent -> 48 kHz ses
  -> AudioChunk                            PCM16-LE parçaları
```

Üretim senkron bir generator'dır. `async_generate_stream` her adımı bir worker thread'e
devrederek bunu event loop'a bağlar, böylece tek bir sentez bütün sunucuyu durdurmaz.

## Karar: modeli çağıran kiralar

Model örnekleri `acquire()` bağlamıyla alınır ve bağlam çıkışında bırakılır:

```python
async with pool.acquire() as model:
    async with aclosing(synthesize_stream(model, text)) as chunks:
        async for chunk in chunks:
            ...
```

Daha önce model, ses üreticisinin kendi `finally` bloğunda bırakılıyordu. İstemci akışın
ortasında koparsa o generator askıda kalıyor ve bırakma, CPython'un terk edilmiş async
generator'ı sonlandırmasına kalıyordu. Bir referans hayatta kalırsa — eski kodda `except
Exception as e:` traceback'i tam bunu yapıyordu — model hiç geri dönmüyor ve sonraki istek
süresiz bekliyordu. Tek GPU'lu bir sunucu ilk kopan bağlantıdan sonra kullanılamaz hale
geliyordu.

Şimdiki yapıda bırakma, istemci koparsa da, sentez hata verirse de, akış yarıda bırakılırsa
da deterministik. `tests/test_pool.py` ve `tests/test_api.py` üç yolu da regresyon testiyle
koruyor.

## Karar: beklemek yerine reddetmek

`acquire()` bir timeout'la çalışır (`pool_acquire_timeout_s`, varsayılan 30 s). Süre
dolarsa HTTP **503**, WebSocket **1013** döner. Sınırsız kuyruk yok: aşırı yükte istemci
hızlıca bilgilenir ve geri çekilebilir, istekler sessizce birikmez.

## Karar: toplu işleme denenmedi

VoxCPM2 toplu çıkarımı desteklemiyor; KV önbelleği `batch_size=1` ile ayrılıyor. Eşzamanlılık
bu yüzden yalnızca model örneği sayısıyla ölçeklenir (`system.voxcpm_gpu_ids`). Model ~8 GB
VRAM istediği için 24 GB'lık bir GPU'ya yaklaşık iki örnek sığar.

Otoregresif döngüye elle batching eklemek yerine, gerçek çok kiracılı kapasite için
upstream'in işaret ettiği vLLM-Omni / Nano-vLLM runtime'ları değerlendirilmelidir; sürekli
toplu işleme (continuous batching) oradan gelir.

## Gecikme nereden geliyor

| Kaynak | Ayar | Etki |
|---|---|---|
| İlk ses paketi için biriktirme | `streaming.chunk_duration_ms` | Doğrudan TTFB. 48 kHz'de 1200 ms, ilk bayt gitmeden 57.600 örnek biriktirmek demek |
| Difüzyon adımı sayısı | `model.inference_timesteps` | Adım başına maliyet doğrusal |
| Yönlendirme | `model.cfg_value` | 1.0'ın üstünde her difüzyon adımı iki kat batch ile çalışır |
| Cümle eşiği | `text_processing.min_chars` | Küçük değer daha erken ilk ses, daha parçalı prozodi |
| Lookbehind | `streaming.enable_lookbehind` | Açıkken her cümle için referans ses yeniden kodlanır |

`model.cfg_value` için Trendyol'un model kartı 2.0 öneriyor ve 2.5 üstünü kırpılma riski
nedeniyle açıkça önermiyor (`merge_manifest.json` → `avoid_as_general_default`).

## Ölçüm

Performans iddiası ölçümle gelir. `benchmarks/bench.py` sunucuyu WebSocket üzerinden
sürerek TTFB, RTF, p50/p95, verim ve tepe VRAM raporlar; raporun içine o anki ayarlar da
gömülür, çünkü sayılar ayarlar bilinmeden karşılaştırılamaz.

RTX 4060 Laptop'ta ölçülen: TTFB p50 341 ms, RTF p50 1.045, tepe VRAM ~5.7 GB
(`benchmarks/baseline.json`). Üç ayar değişikliği TTFB'yi %88, RTF'yi %50 düşürdü:
`chunk_duration_ms` 1200→200, `max_length` 8192→4096, `cfg_value` 2.8→2.0.

`max_length` en büyük tek kazanç oldu, çünkü SDPA her AR adımında tüm KV penceresi
üzerinde çalışıyor. 2048 bir miktar daha hızlıdır ama üretim döngüsünün kendi üst sınırı
4096 olduğu için önbellek taşabilir; varsayılan bu yüzden 4096.

`cfg_value` hızı etkilemiyor: CFG zaten 1.0'ın üstündeki her değer için adımı 2× batch ile
çalıştırıyor. 2.0 tercih edilme sebebi model kartının önerisi ve kırpılma riski.

Upstream RTX 4090 için RTF ~0.30, hızlandırılmış runtime'larla ~0.13 bildiriyor; bunlar
upstream'in sayılarıdır.

## Vendor'lanmış VoxCPM

`_vendor/voxcpm/` upstream'in değiştirilmiş kopyasıdır. Çıkarım ve ince ayar dışındaki her
şey çıkarıldı: Gradio demoları, `voxcpm` CLI'si, zaman damgası araçları, upstream yayın
iş akışı. Akış katmanı (`streaming.py`, `streaming_async.py`, `text_buffer.py`) ve
yapılandırma köprüsü (`config_loader.py`) bu projenin eklemeleridir.

Vendor'a dokunmadan çözülebilecek bir şey vendor'da çözülmez; örneğin ek veri doğrulaması
`finetune.py` içindedir. Her değişiklik `LICENSE-NOTICE.md`'de listelenir (Apache-2.0 §4(b)).

Hâlâ duran ölü ağırlık: VoxCPM v1 model, VAE ve DiT kopyaları import ediliyor ama VoxCPM2
checkpoint'i için hiç örneklenmiyor. Silinmeleri `torchaudio` bağımlılığını da düşürür.

## Kasıtlı olarak yapılmayanlar

- **Kimlik doğrulama ve hız sınırlaması.** Servisin önüne bir ağ geçidi konulmalı.
- **Ses klonlama yüzeyi.** Model destekliyor, API bir `voice` parametresi sunmuyor.
- **Kuantizasyon.** GGUF/ONNX varyantları yalnızca harici projelerde var.
- **Metin normalizasyonu.** Sayı ve kısaltma açma varsayılan olarak kapalı.
- **Çalışma anında yapılandırma yenileme.** Ayarlar import anında okunur.
