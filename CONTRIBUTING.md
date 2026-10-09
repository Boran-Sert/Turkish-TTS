# Katkı Rehberi

Katkılara açıktır. Hata bildirimi, döküman düzeltmesi ve kod katkısı aynı süreçten geçer.
Büyük bir değişikliğe başlamadan önce bir issue açıp yaklaşımı tartışın; böylece kimse
reddedilecek bir işe hafta harcamaz.

## Geliştirme ortamı

Python 3.10 veya üstü gerekir.

```bash
git clone https://github.com/Boran-Sert/Turkish-TTS.git
cd Turkish-TTS

# CUDA'lı torch. Varsayılan pip Windows'ta CPU tekerleğini çeker.
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

uv pip install -e ".[dev]"            # geliştirme
uv pip install -e ".[dev,finetune]"   # ince ayar koduna da dokunacaksanız
```

Testler GPU ve model ağırlığı gerektirmez:

```bash
python -m pytest tests/
```

Ağırlıkları yalnızca gerçek sentez denemek için indirin (~5 GB, bir kez):

```bash
turkish-tts-fetch
turkish-tts-server
```

## Pull request süreci

1. `main`'den bir dal açın.
2. Değişikliği yapın ve **testini yazın**.
3. `python -m pytest tests/` yerelde geçsin.
4. Conventional Commits biçiminde commit atın (aşağıya bakın).
5. PR açın, ne yaptığınızı ve neden yaptığınızı yazın.

CI üç şeyi kontrol eder: testler (Python 3.10/3.11/3.12), yayın kurallarının geçerliliği ve
paketin gerçek bir wheel olarak kurulup import edilebilmesi. Üçü de geçmeden merge edilmez.

## Commit biçimi ve sürümleme

Sürüm numarası commit mesajlarından otomatik üretilir
([Conventional Commits](https://www.conventionalcommits.org/)). `main`'e merge edilen her
değişiklik, testler geçerse, kendi sürümünü ve PyPI yayınını tetikler.

| Commit | Örnek | Sonuç |
|---|---|---|
| `feat!:` veya `BREAKING CHANGE:` | `feat!: voice parametresi zorunlu` | major, `0.x` iken doğrudan `1.0.0` |
| `feat:` | `feat: ses klonlama ekle` | minor |
| `fix:` `perf:` `docs:` `refactor:` `style:` `build:` | `fix: havuz sızıntısını kapat` | yama |
| `chore:` `ci:` `test:` | `ci: iş akışını düzelt` | yayın yok |

Kurallar `pyproject.toml` içindeki `[tool.semantic_release]` altında tanımlı ve
`tests/test_release_config.py` ile korunur; bir kuralı değiştirirseniz test kırılır.

**Sürüm numarasını elle düzenlemeyin.** `pyproject.toml` ve `turkish_tts.__version__`
yayın akışı tarafından yazılır.

## Kod kuralları

**Docstring.** Her dosyanın başında ve her fonksiyonda bir docstring olur. **Tek, basit bir
cümle.** Karmaşıklık notu (`O(1)`, `O(N)` gibi) yazmayın.

```python
"""Pool of VoxCPM model instances, one per GPU, rented out with a timeout."""


def sample_rate_of(model: VoxCPM) -> int:
    """Returns the model's output sample rate, 48 kHz for VoxCPM2."""
    return model.tts_model.sample_rate
```

**Uzun satır içi yorum yazmayın.** Kodun ne yaptığını tekrar eden yorum gereksizdir. Bir
yorum gerçekten kaçınılmazsa kısa tutun ve *neden*i açıklasın, *ne*yi değil.

**En az kodu yazın.** İstenmeyen özellik, tek kullanımlık soyutlama ve gereksiz
yapılandırma seçeneği eklemeyin. Gerçekleşemeyecek durumlar için hata yakalama koymayın.

**Cerrahi değişiklik.** Yalnızca görevin gerektirdiği satırlara dokunun. Yoldan geçerken
gördüğünüz kodu, yorumu veya biçimlendirmeyi "iyileştirmeyin" — ayrı bir PR'a ayırın.
Kendi değişikliğinizin kullanılmaz hale getirdiği import ve değişkenleri silin.

**Dil.** Kod, tanımlayıcılar ve docstring'ler **İngilizce**; log mesajları ve satır içi
yorumlar **Türkçe**. Depodaki baskın kullanım budur, bozmayın. Log ve başlıklarda emoji
kullanmayın.

**Tip ipuçları.** Public fonksiyonların imzalarında kullanın. İçeride zorunlu değil.

## Test

Hata düzeltiyorsanız **önce hatayı gösteren testi yazın**, sonra geçirin. Böylece
düzeltmenin gerçekten bir şey düzelttiği kanıtlanır ve hata geri gelmez.

Yeni davranış ekliyorsanız testi de gelir. Testler model ağırlığı veya GPU istemeyecek
şekilde yazılır; mevcut testler bunu sahte bir model havuzuyla yapıyor, örnek olarak
`tests/test_api.py` dosyasına bakın.

## Performans değişiklikleri

**Ölçmeden optimize etmeyin.** Performans iddiası taşıyan bir PR'da öncesi/sonrası ölçüm
bulunmalı:

```bash
turkish-tts-server &
python benchmarks/bench.py --concurrency 1,2 --compare benchmarks/baseline.json
```

Bu kural deneyimden geliyor: geçmişte bir commit, bir takılma sorununu ölçmeden config'i
yavaşlatarak çözmeye çalıştı ve asıl sebep (havuzdaki model sızıntısı) gözden kaçtı.

Model ayarları için `Trendyol-TTS/merge_manifest.json` içindeki `clean_default` ve
`avoid_as_general_default` alanları bağlayıcıdır.

## Vendor'lanmış kod

`src/turkish_tts/_vendor/voxcpm/` upstream [VoxCPM](https://github.com/OpenBMB/VoxCPM)'in
değiştirilmiş kopyasıdır ve Apache-2.0 ile lisanslıdır.

- Orada bir dosyayı değiştirir veya yeni dosya eklerseniz `LICENSE-NOTICE.md`'deki listeyi
  güncelleyin. Apache-2.0 §4(b) bunu zorunlu kılıyor.
- Değiştirilmiş dosyalar `# NOTICE:` başlığı taşır; bu başlıkları silmeyin.
- Vendor'a dokunmadan çözülebilecek bir şeyi vendor'da çözmeyin. Örneğin ek veri
  doğrulaması `src/turkish_tts/finetune.py` içine yazılır.

## Yapay zeka ile üretilen kod

Yapay zeka araçlarını kullanmanız serbest. Ancak gönderdiğiniz kodun sorumluluğu sizde:

- **Gönderdiğiniz her satırı okuyun ve anlayın.** Açıklayamadığınız kodu PR'a koymayın.
- **Çalıştırın.** "Derlendi" yeterli değil; testleri koşun, davranışı doğrulayın.
- **Yukarıdaki kod kurallarına uydurun.** Üretilen kod tipik olarak gereğinden fazla
  yorum, gereksiz try/except ve tek kullanımlık yardımcı fonksiyon içerir. Temizleyin.
- **Uydurulmuş API'lere dikkat edin.** Var olmayan fonksiyon, parametre ve ayar isimleri
  sık görülür. Koddan doğrulayın.
- **Ölçüm ve iddiaları doğrulayın.** Yapay zekanın "bu daha hızlı" demesi ölçüm değildir.

Gözden geçirilmemiş, kuralları tutmayan üretim kodu içeren PR'lar kapatılır.

## Göndermeyin

- Model ağırlıkları (`*.safetensors`, `*.pth`, `*.onnx`). `.gitignore` bunları dışlıyor,
  zorla eklemeyin.
- `.env`, token, API anahtarı veya herhangi bir kimlik bilgisi.
- Üretilmiş çıktılar: `dist/`, `build/`, `logs/`, `.pytest_cache/`.
- İzni veya kaynağı belirsiz ses kayıtları. Ses klonlama için eklenen her referans kaydın
  izinli olması gerekir.

## Lisans

Katkınız projenin lisansı olan Apache-2.0 altında yayınlanır.
