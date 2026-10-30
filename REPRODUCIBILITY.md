# Yeniden üretilebilirlik

Sürüm: 0.11.0 · Tarih: 2026-10-25 · Kapsam: latent, EN/TR tam matris, inference, CBE, lexical, reasoning, isteğe bağlı eğitim ve bilimsel raporlama.

Deterministik veri üreticisi, iki exact oracle ve bütün dataset doğrulaması
uygulandı. Türkçe renderer/morfoloji ve kapalı yüzeyden-mantığa parser hazırdır.
Model çalıştırıcısı, kesin model-cevap parserı ve dry-run altyapısı hazırdır.
İstatistiksel analiz/puanlama modülü uygulanmıştır; gerçek model verisi üzerinde
analiz ve eğitim yapılmamıştır.
Hiçbir model çağrısı veya model deney sonucu yoktur.
`manifest` çıktısı `planning_only` niteliğinde bir plan ve köken kaydıdır;
deneyin yapıldığına, verinin doğrulandığına veya modellerin seçildiğine kanıt
değildir. Bilimsel kararlar ve çalıştırma kapıları [deney protokolündedir](EXPERIMENT_PROTOCOL.md).

## 1. Ortam ve kurulum

Desteklenen Python aralığı `>=3.11,<3.14`; Prompt 7'de 706 test gerçek BPE
bağımlılıklarıyla Python 3.12.14 ve Python 3.13.1 üzerinde ayrı ayrı geçti
(atlanan test yok). Python 3.11 henüz test edilmedi. Temel katmanlar standart
kütüphaneyi, TOML için `tomllib`, testler için `unittest` kullanır. v0.4.1
BPE katmanı üçüncü taraf bağımlılıklarını ayrı hash kilidiyle sabitler;
çevrimdışı kurulum §8 içindedir. Eski aşamaların test/kurulum kayıtları
[PROGRESS.md](PROGRESS.md) içinde tarihleriyle korunur.

Proje kökünde önerilen kurulum:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-build.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
```

`python3` için yukarıdaki aralığa uyan yorumlayıcı seçilmelidir. Build backend
`setuptools==84.1.0` olarak sabitlenir. `requirements-build.lock` içindeki
wheel SHA-256 değeri
`5231689c6ac8156cd155cfb98c3fd6fb252c3e7e9bd3d78b64d3c55f132624c9`'dır;
[PyPI sürüm kaydı](https://pypi.org/project/setuptools/84.1.0/) üzerinden
doğrulanmıştır. Hash denetimini veya build izolasyonunu açıp kapatmayı sessizce
değiştirmeyin; farklı kurulum yeni ortam kaydı gerektirir.

Bu kilit tüm işletim sistemini, yorumlayıcı derlemesini, `pip` sürümünü veya
gelecekteki veri/model bağımlılıklarını sabitlemez. Python 3.12.14,
setuptools 84.1.0, pip 26.2.2 ve wheel 0.48.0 ile ağsız wheel üretimi,
yerel hedef klasöre wheel kurulumu ve editable kurulum kontrolü yapıldı.
Build izolasyonu kapalıydı; mevcut sabit sürümlü backend kullanıldı. Editable
kontrolü sistem paketlerini gören geçici bir sanal ortamda yapıldı. PyPI'den
temiz ortamda hash kilidiyle indirme denenmedi; hash resmi sürüm kaydından
doğrulandı. Kontrollerin ayrıntısı [PROGRESS.md](PROGRESS.md) içindedir.
Prompt 3'te v0.3.0 wheel yeniden oluşturulup kuruldu; kurulu
`turkish-cbe-render-review` 252 satırla ve aynı CSV hash'iyle doğrulandı.
v0.3.0 editable kurulum ayrıca tekrarlanmadı; önceki editable kontrolü tarihsel
v0.2.0 kaydıdır.

Paket kurmadan, kaynak ağacından mevcut komutlar:

```sh
PYTHONPATH=src python3 -m turkish_cbe validate configs/pilot.toml
PYTHONPATH=src python3 -m turkish_cbe manifest configs/pilot.toml --output work/pilot-plan.json
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Bu komutlar veri üretmez ve ağa/model API'sine bağlanmaz. `validate` seçilen
konfigürasyonu denetler ve planın bütçesini hesaplar. `manifest` mevcut çıktı
dosyasının üzerine yazmayı reddeder; yeni bir kayıt için yeni dosya adı seçin.
Kaynak ağacından test başarısı, kurulmuş paketin doğrulandığı anlamına gelmez.

### Latent veriyi doğrulama ve yeniden üretme

```sh
PYTHONPATH=src python3 -m data_generation verify dataset_manifest.json --replay
PYTHONPATH=src python3 -m data_generation generate --seed 11 --output work/reproduced-latent --manifest work/reproduced-manifest.json
```

Varsayılan n, örnek sayısı, isim/sorgu türleri ve d ızgarası CLI ile değişebilir.
Tek bir n/kol/d bloğunun RNG akışı diğer blokların seçimine bağlı değildir.
İsim grubu listesi değişirse aynı soyut ordering_id farklı etiket alabilir;
bu eşlenik türevlerin split sızıntısını engellemek için bilinçli tasarımdır.
İsim havuzu, algoritma veya serileştirme değişimi generator_version artışı ister.

`data/latent/` 71 JSONL dosyası, `dataset_manifest.json` gerçek veri manifestidir.
Plan manifestinden farklıdır. Veri manifesti timestamp içermez; sabit plan,
kod, ortam ve göreli yerleşimde kendisi de deterministiktir. Farklı Python/OS
ortamı manifest metaverisini değiştirebilir; veri byte hashleri ayrı karşılaştırılır.

Her iki oracle bütün kayıtlarda çalışır. `verify --replay` disk dosyalarını
okuyup semantiği, dengeyi, ortak sıra kimliklerini, isim/mesafe metadatasını,
boyut ve hashleri denetler; sonra bellekte aynı veriyi üretip 71 hash ile
karşılaştırır. Aynı veri Python 3.13.1 ve 3.12.14 ile replay kontrolünden geçti.
İki tam üretimin bütün JSONL hashleri de eşleşti. Yeni çıktı hedefi gereklidir;
mevcut veri sessizce silinmez. Başarısız çağrı kendi kısmi çıktısını temizler.

Üretilmiş `data/latent/` Git dışında tutulur; teslim ZIP'i veri ve manifesti
içerir. Sonraki arşiv yayımlamada ZIP ve iç dosya hashleri ayrıca korunmalıdır.

### Türkçe sunum ve insan kontrol raporu

```sh
PYTHONPATH=src python3 -m renderers.review --output work/turkish_samples.csv --manifest work/turkish_samples_manifest.json --seed 31 --include-negation
```

252 satır, 36 farklı kaynak sıra ve P0–P6 üretir. P6 izin bayrağı yoksa
216 P0–P5 satırı vardır. Eski insan raporu/manifest üzerine yazılmaz. İnsan
kararı alanları boş kalır; bir kopyada yapılan insan incelemesi yeni artefakt
olarak hashlenmelidir. Otomatik sonuçlar gerçek insan onayı gibi sunulmaz.

Renderer sürümü 0.3.0, latent generator sürümü 0.2.0'dır. Yeni renderer eski
veriyi değiştirmez. Raporda kaynak dataset manifesti, altı JSONL ve 36 kaynak
kimliği, renderer/normalizer kodu, CSV hashleri ve seed kayıtlıdır. Aynı kaynak,
kod ve seed ile 252 satırlık CSV Python 3.13.1 ve 3.12.14'te aynı byte hashini
verdi. Ortam bilgileri yüzünden rapor manifestinin tamamı sürümler arasında
bayt eşitliği taşımaz. Canonical sorgu ve normal-form hashleri varyantlar
arasında korunur; görünür promptun kendi hash'i ayrıca tutulur.

Türkçe yazım/ek ve cümle tanıma kapalı bir sözlüktür; genel morfolojik
çözümleyici veya serbest paraphrase doğrulayıcısı değildir. Beklenmeyen metin,
yanlış ek, ek/eksik öncül, değişmiş sorgu/bağlam, yanlış faktör veya izinsiz
liste permütasyonu reddedilir. Normal form gerçek metinden hesaplanır;
metadata içindeki iddia doğru kabul edilmez. Ayrıntılı insan inceleme sınırı
[data/validation/README.md](data/validation/README.md) içindedir.
Metin grameri renderer'dan ayrı yazılmıştır; ilgi/ayrılma eklerinin kural
fonksiyonları ortaktır. Bağımsız beklenen-biçim testleri bu ortak hata riskini
azaltır; insan ek/doğallık incelemesinin yerine geçmez.

## 2. Konfigürasyon sözleşmesi

`configs/pilot.toml`, `configs/baseline.toml` ve `configs/advanced.toml` ayrı,
tam konfigürasyonlardır. Kalıtım ve ortam değişkeninden değer ezme yoktur;
çalıştırılacak dosya açıkça seçilir. Bilinmeyen alanlar ve geçersiz değerler
reddedilmelidir. Bir varyantı veya örneklem hücresini sessizce çıkarmak,
varsayılanla düzeltmek ve farklı ayarlarla aynı deney adı altında devam etmek
yasaktır.

Model/provider/revision yer tutucuları `UNSELECTED` olarak kalır. Sonuç kaydı
ve isteğe bağlı eğitim kapalıdır. İleri konfigürasyon yalnız RQ5 prompting
taslağıdır; RQ3 uzaklık, RQ4 sıklık ve fine-tuning için yürütülebilir deney
planı değildir. Bu yer tutucuları geçerli sayan bir plan doğrulaması,
bilimsel çalıştırma izni vermez.

Her deneyden önce konfigürasyon, protokol sürümü, veri kümesi, şablonlar,
promptlar, model snapshot/revision ve analiz planı birlikte dondurulmalıdır.
Model çıktısı görüldükten sonraki değişiklik yeni sürüm ve gerekçe alır.
Pilot verisi temel değerlendirmeye taşınmaz.

## 3. Bugünkü plan manifesti

Manifest aşağıdaki bilgileri açık biçimde kaydeder:

- Dosyanın ham baytlarına ait SHA-256 ve doğrulanıp normalize edilmiş
  konfigürasyonun SHA-256 değeri. İlki yorum/boşluk değişimini de yakalar;
  ikincisi aynı ayarların karşılaştırılmasını sağlar.
- Proje kaynaklarının göreli dosya yolları, dosya başına SHA-256 değerleri ve
  birleştirilmiş ağaç özeti. Üretilmiş çalışma dosyaları kaynak olarak sayılmaz.
- Python sürümü, platform ve kurulu paket sürümleri; bulunabiliyorsa Git commit
  kimliği. Commit tek başına çalışma kopyasının kanıtı değildir; kaynak hashleri
  esas alınır. Git yoksa bu durum eksik değerle açıkça gösterilir.
- Planlanan seed/ayarlar ve hesaplanan çağrı bütçesi; veri kümesi ve
  model revision seçilmemişse `null`/`UNSELECTED` ve `planning_only` işareti.

Manifest hiçbir API anahtarını, kimlik bilgisini, ortam değişkeni dökümünü veya
secret değerini içermez. İleride kimlik bilgileri konfigürasyona, prompta ve
artefaktlara yazılmaz. Paket/platform bilgileri paylaşılmadan önce kişisel
yol veya yerel kurulum bilgisi içerip içermediği gözden geçirilir.

Aynı kaynak ve konfigürasyondan alınan kayıtlarda kaynak/ayar özetleri
eşleşmelidir. Zaman ve ortam alanları nedeniyle manifest dosyasının tamamının
bayt bayt aynı olması beklenmez. Manifestte veri hashinin bulunmaması,
üretilmiş veri varmış gibi yorumlanamaz.

## 4. Sonraki aşamanın veri ve çalışma kaydı

Latent veri üreticisi uygulandı; veri dosyaları sabit artefakt olarak
saklanmalı; sıralı kayıtlar, UTF-8/NFC politikası, satır sonu/serileştirme
biçimi, kayıt sayısı ve dosya SHA-256 değerleri tanımlanmalıdır. Her latent
`ordering_id` grubunun bütün dilleri, sorguları ve yüzey türevleri aynı splitte
kalmalıdır. Veri, şablon, isim havuzu, tokenizer ve sıklık derlemi sürümü ile
lisans/kaynak bilgileri ayrı kaydedilmelidir. Bir arşiv için hem arşiv dosyası
hem içerdiği veri dosyaları hashlenmelidir.

Seed tek başına yeterli değildir. Üretim seedleri, seed türetme algoritması,
örnekleme/permutasyon sırası ve üretici sürümü dondurulmalıdır. Gelecekte seed
türetme için süreçten sürece değişebilen Python `hash()` kullanılmamalıdır.
Veri üretimi aynı sabit ortamda tekrarlandığında artefakt hashleri
karşılaştırılır; fark varsa doğrulama durur. Bugünkü üretici SHA-256 namespace
ile ayrı RNG akışları kullanır ve determinismi test edilmiştir.

Gerçek model çalıştırıcısı eklenmeden önce kayıt şeması en az şunları içermelidir:

- Run/istek/prompt/latent/sunum kimlikleri, veri ve prompt hashleri; config,
  protokol, runner ve parser sürümleri.
- Sağlayıcı, istenen model kimliği ve revision ile yanıtın bildirdiği model
  snapshot/fingerprint; erişilemiyorsa açık `null` ve nedeni. İstenen seed,
  decoding, mesaj rolleri, zaman damgaları ve varsa request id.
- Ham istek/yanıtın izin verilen kısmı, son cevap, input/output/reasoning token
  sayıları, `finish_reason` ve kesilme durumu. Sağlayıcının vermediği alanlar
  tahmin edilmez; gizli muhakemeye erişim varsayılmaz.
- Her denemenin başarı/hata durumu, retry sırası ve gerekçesi, bekleme politikası,
  son başarısızlık ve tekrar kontrolü işareti. Ulaşım retry'ı ile planlı ikinci
  gözlem ayrı kayıtlardır; geçersiz cevaplar silinmez veya seçilerek yenilenmez.

Ham kayıtlar ve puanlar ayrı artefaktlar olmalıdır. Düzeltme ham yanıtı
değiştirmez; yeni parser/puanlama sürümü ve yeni çıktı üretir. Sentetik test
fixture'ları gerçek model kayıtlarıyla karıştırılmaz ve başarı sonucu olarak
raporlanmaz. Model yanıtı veya puan sonucu saklama akışı henüz uygulanmadı.

## 5. Tekrar üretimin düzeyleri ve sınırları

| Düzey | Hedef ve gerekli kanıt |
|---|---|
| Planı yeniden kurma | Aynı kaynak/config hashleri, doğrulama ve aynı çağrı bütçesi. Bu aşamanın kapsamı. |
| Latent veriyi yeniden üretme | Uygulandı: sabit üretici/seed ile aynı JSONL hashleri; iki Python sürümünde tam replay. |
| Tam eşleştirilmiş matrisi yeniden üretme | Uygulandı: v0.4.1 bütün 5.637.600 prompt ölçümü, iki gerçek BPE ve yüzeyler derin replay ile doğrulandı. |
| Türkçe kontrol raporunu üretme | Uygulandı: aynı 252 CSV satırı/byte hash'i iki Python sürümünde; insan kararları pending. |
| Arşivden yeniden puanlama | Aynı ham yanıtlar ve sabit parser/analiz ile aynı puan artefaktları. Sonraki aşama. |
| Model deneyini yeniden çalıştırma | Yeni API istekleriyle aynı tasarımın tekrarı; yeni run ve sonuç artefaktları. Bayt eşitliği garanti değildir. |

Tam kilitli bir yerel ortam bile uzak API'nin model yönlendirmesini, sunucu
güncellemesini, donanımını veya eşzamanlı hesaplama davranışını sabitlemez.
`temperature=0` ve seed, aynı yanıtın garantisi değildir. Snapshot
bulunamaması veya sürüm değişimi açık sınırlılık olarak raporlanır; farklı
revision ile tekrar deney aynı run'a eklenmez. İleride yerel eğitimde de
donanım, sürücü, sayısal kütüphaneler ve deterministik işlem ayarları ayrıca
sabitlenmelidir.

Yayımlama öncesi yeniden puanlama yolu şimdiden şu sıraya bağlanır:

1. Arşiv ve dosya hashlerini doğrula; config/protokol/kod sürümlerini eşleştir.
2. Ham yanıtlara dondurulmuş parserı uygula; invalid, eksik ve kesilmiş kayıtları
   koru. Ana paydalardan çıkarmama kuralını protokole göre uygula.
3. `ordering_id` eşlemeleri ve split/sunum kapsamasını doğrula; eksik çiftleri
   ve `V=0` durumlarını açıkça işle.
4. Ölçütleri ve küme bootstrap analizini sabit analiz seediyle yeniden hesapla;
   türetilmiş çıktı hashlerini ve farkları raporla.

Bu adımların kodu henüz yoktur; bugün bunları yapan bir CLI komutu sunulmaz.

## 6. Mevcut doğrulama sınırı ve çalıştırma kapısı

Testler konfigürasyon/provenance, üretici/iki oracle/isim havuzları/dataset
yanında Türkçe morfoloji, dokuz ilişki×P0–P6, semantik bozulmalar ve insan raporu
CLI'sini kapsar. n=2..5 tüm 152 permütasyonda bütün
sorgular iki oracle ile karşılaştırıldı; n=1024 de test edildi. Bu testler
genel Türkçe eşdeğerliğini veya insan düzeyinde dilsel geçerliliği,
istatistiksel gücü ya da API davranışını doğrulamaz.

Latent üretici/oracle, veri manifesti, Türkçe renderer, otomatik dilsel
kontroller ve model-cevap parserı/runner testleri hazırdır. Gerçek pilot için
kullanıcının model/erişim seçimi, çift değerlendirici şablon onayı ve seçilen
model/servis ortamının doğrulanması gerekir. Temel
değerlendirme ayrıca güç/duyarlılık hesabı, dondurulmuş analiz ve lisans/bütçe
kapılarını geçmelidir. Bu koşullar tamamlanmadan araştırma sonuçları veya
tam yeniden üretilebilir deney altyapısının hazır olduğu iddia edilmez.


## 7. Tarihsel v0.4.0: eşleştirilmiş veri ve frozen test

Yapısal veri konfigürasyonu `configs/matched_dataset.toml` ile sabittir.
Legacy üretici ve Türkçe kaynakları değiştirilmedi. Yeni English, aliases,
pairing, surface reader ve dataset builder kaynakları kök matched manifestte
SHA-256 ile kayıtlıdır. Her splitin JSONL şeması DATASET_CARD.md'dedir.

Teslim üretimi proje kökünde Python 3.12.14 (UCD 15.0.0) ile:

```sh
PYTHONPATH=src python3 -m experiments.dataset_builder build --config configs/matched_dataset.toml --output-root data --manifest experiment_dataset_manifest.json --legacy-manifest dataset_manifest.json
PYTHONPATH=src python3 -m experiments.dataset_builder verify --manifest experiment_dataset_manifest.json
```

İlk komut yalnız hedef splitler/manifest yokken çalışır. Yeniden üretim için
boş hedef seçin veya projeyi yeni bir klasöre çıkarın ve yalnız bu aşamanın
yeni splitlerini oluşturun. Eski manifest/verinin göreli yerleşimi korunursa
legacy guard kaydı da aynıdır. Manifestler zaman damgası taşımaz; aynı
kaynak/plan/Python/tokenizer ve göreli dosya yerleşimi deterministik byte
hashleri üretir. Dış ortamın farklı olması manifest ortam kaydını değiştirebilir.

Her source sorgu iki oracle ile denetlenir. Her tam prompt actual metinden
normal forma geri doğrulanır; partial converse maskesi iki dilde aynıdır.
F için tüm sorgularda aynı öncül/isim/converse planı kullanılır. Aynı source
order, aliasları ve bütün sorguları tek partition içindedir. Seed11 eski
geliştirme kaynağının endpoint dosyaları hash kontrolünden geçirilip legacy
çakışma guard'ına alınır. Grup/alias/görünür metin tekrarları reddedilir.

`groups/latent/prompts/cells.jsonl`, split manifestleri ve dış manifest ayrı
ama bağlı kayıtlardır. Hücre sayıları, gold marjinleri, kaynak kimlikleri,
beklenen bütün karşılaştırmalar, alias politikası, seed-plan replay ve dosya
hashleri birlikte doğrulanır. Başka dosyaya başvuru, eksik hücre veya
aynı kimliğin tekrarı kabul edilmez. Hash güncellemek semantik denetimin
yerine geçmez. Harici güvenilir hash/arşiv kaydı frozen sürümün kimliğini korur.

Referans tokenizer Python Unicode veri tabanı sürümünü kimliğe dahil eder.
Teslimin `unicode-lexical-v1-ucd-15.0.0` kimliğiyle derin replay için aynı
UCD gerekir. Python 3.13.1/UCD15.1.0 ile birim testleri çalışsa da tam matched
byte replay iddiası kurulmaz. Bu ortamda `--hashes-only` güvenilir hash
kontrolünü sağlar. Model tokenizer, chat şablonu/sistem mesajı ve model
revision yoktur; `model_token_count=null` kalır. Sayıları BPE gibi kullanmayın.

Training/development/pilot/test veri dosyaları üretilmesi eğitim ya da model
değerlendirmesi çalıştırılması değildir. Yapısal frozen durumu insan dilsel
onayı değildir. Model sonuçları, güvenirlik metrikleri, istatistiksel analiz
ve fine-tuning bu aşamaya dahil değildir.

Prompt 4'te 223 test iki Python sürümünde geçti. v0.4.0 wheel ağsız üretildi
ve yeni bir yerel hedefe kuruldu; dört kurulu CLI doğrulandı. Matched mini
üretim 16 grup / 48 kaynak sorgu / 400 prompt satırı / 400 karşılaştırma,
derin verify 20 dosya ve 304 farklı görünür metin verdi. Eski dilsel rapor
252 satırla aynı hash'i korudu. Güncel bütün Python kaynakları wheel'dekilerle
byte eşitliği kontrolünden geçti; bu aşamada editable kurulum tekrarlanmadı.


2026-10-24 son kontrolü: tam matched veri derin verify sonucunda 20 dosya,
3.000 order, 153.400 prompt satırı ve 145.602 farklı metin ile valid=true
verdi. Python 3.13.1 hashes-only kontrolü de geçti. Kaynak, veri ve plan
hashleri PROGRESS.md ve experiment_dataset_manifest.json içindedir.


## 8. Prompt 4 eksiklerinin tamamlanması — v0.4.1

Tam matris yeni complete_grid paketindedir; eski 24 Python kaynağı ve önceki
veri dosyaları değişmez. Kaynak parmak izi taraması yeni paketi eski sürümün
kapsamına katmaz; eski frozen bankanın derin replay sözleşmesi korunur.

Gerçek BPE için tiktoken 0.12.0, regex 2026.9.29 ve bütün transitif bağımlılıklar
requirements-tokenizers.lock ile sabittir. Kilit, resmî sürüm dosyalarının
platform hashlerini içerir. Vendor wheel'leri CPython 3.12/macOS arm64'tür;
bu ortamda hash kilidiyle tamamen ağsız kurulum doğrulanmıştır:

```sh
python3 -m pip install --no-index --find-links vendor/tokenizers --require-hashes -r requirements-tokenizers.lock
```

Diğer platformlar aynı kilitteki kendi wheel'ini bir kez indirir. Vocabularies
ve encoding kuralları proje tokenizers/ klasöründedir; üretim/ölçüm için
internet veya API anahtarı gerekmez. Her iki vocabulary hash'i sabittir;
regex/special-token tanımı sabit paketin resmî constructor'ıyla karşılaştırılır.

Tam üretim ve doğrulama (proje kökünde, CPython 3.12.14):

```sh
PYTHONPATH=src python3 -m complete_grid.dataset_builder build --config configs/matched_dataset.toml --output data/complete_grid --workers 4
PYTHONPATH=src python3 -m complete_grid.dataset_builder verify --manifest data/complete_grid/manifest.json --workers 4
PYTHONPATH=src python3 -m complete_grid.dataset_builder verify --manifest data/complete_grid/manifest.json --hashes-only
PYTHONPATH=src python3 -m complete_grid.dataset_builder export --manifest data/complete_grid/manifest.json --split test --n 1024 --relation comparison_speed --limit 100 --output work/speed_1024.jsonl
```

Hedef klasör/dosya exclusive oluşturulur. Worker 1 ve worker 2 aynı kaynak,
ölçüm sırası ve mantıksal satır digest'ini üretir; tek parent süreç SQLite'a
yazar. İşçi sayısı veri faktörü değildir. En fazla2×workers görev bellekte
bekler. Dört worker ile büyük üretim yapılır; bütün gerçek BPE sayıları
hesaplanır. Ortak ön ekler yalnız ispatlı regex sınırında yeniden kullanılır;
her tam metin hash'i gerçek prefix/suffix byte dizisinden üretilir.

SQLite dosya hash'i teslimdeki fiziksel artifact'i dondurur. SQLite/Python
sürümleri manifestte kayıtlıdır. Platformlar arasında SQL dosya byte'ları
yerine canonical ölçüm satırlarının construction_rows_sha256 değeri de
karşılaştırılabilir. Derin replay sabit kaynak, tokenizer ve Unicode ortamını
gerektirir; hash-only gelecekteki ortamlarda ayrıca kullanılabilir.

Yeni JSONL export her iki gerçek BPE sayısını ve tüm metni içerir. Eski v0.4.0
JSONL null alanları değiştirilmez; güncel ölçüm bankası onları sürümlü biçimde
tamamlar. Düz metin token sayısı sağlayıcının toplam chat fatura sayısı değildir.
Modellerin sonraki inference seçimleri bu tokenizasyon kontrolünden ayrıdır.


Son v0.4.1 kontrolü (2026-10-24): gerçek BPE bağımlılıklarıyla 284 test
Python 3.12.14 ve 3.13.1 üzerinde geçti, atlanan test yoktur. Tam üretim ve
dört worker ile **bütün veri için derin replay** başarılıdır: 3.000 sıra,
23.200 sorgu, 5.637.600 ölçülmüş prompt, 7.819.200 karşılaştırma; canonical
satır digest'i `1bb2f131d3f62422a14244baf91bcefa15286866bd2a66995bbf7b2058c8ece7`.
Python 3.13.1 portable hash/schema/coverage kontrolü ayrıca geçti. Kurulu
v0.4.1 wheel ile mini üretim, derin replay ve 243 tam metnin gerçek BPE
dışa aktarımı doğrulandı; 31 kurulu kaynak dosyası teslimle byte eşittir.
Tam dosya listesi, hashler, ortam ve sonuçlar PROGRESS.md içindedir.


## 9. Prompt 5 — dry-run, model kimliği ve değişmez inference kayıtları

Kullanıcı gerçek pilotu model seçimine bıraktı. Teslimdeki iki logs JSON
planı gerçek model çalışması değildir; actual_generation_calls=0 ve
contains_model_results=false alanları açıktır. Fixture sonuçları da gerçek
pilot sayılmaz; test dizinleri dışında yayımlanmaz.

PyYAML 6.0.3 bağımlılığı requirements-inference.lock içindeki 73 resmî PyPI
dosya hash'iyle sabittir. CPython 3.12/3.13 macOS arm64 wheel'leri teslimdedir
ve --no-index --require-hashes ile boş hedeflere kurulumları doğrulandı.
OpenAI/vLLM istemcisi stdlib HTTP kullanır. HF'nin opsiyonel torch/transformers
uyumluluk aralığı platforma özgü tam kilit değildir; gerçek CPU/GPU deneyi
öncesi bu ortam ayrıca dondurulmalıdır. Kurulu library sürümleri, Python ve
platform run planına kaydedilir; değişirse aynı planla devam edilmez.

Model yapılandırması tamamıyla plan içinde kopyalanır; spec/mesaj/kaynak
prompt, veri manifesti, cohort ve inference kod hashleri ayrı tutulur.
OpenAI seed desteklenmediğinde null zorunludur. HF/vLLM immutable model ve
tokenizer commit'i, template hash'i ve tarih kullanan template için açık
template tarihi ister. vLLM API
yalnız serving sürümünü ve sunulan model ID'sini doğrulayabilir; ağırlık
commit'i sunucu tarafından kanıtlanmıyorsa
`revision_verification=served_id_only_weights_revision_declared`
olarak kalır. Sunucunun launch komutu ve ağırlık dosyaları ayrıca arşivlenir.

Pilot default n=4/8 × 4 dengeli latent sıra × 4 EN/TR canonical/equivalent
sunum=32 çağrı/model; full 9 n × 200 sıra × 4 sunum=7.200 çağrı/model. Model
seçimi bu cohort'u değiştirmez. Endpoint/internal/distance tabakaları ayrı
kalır. Bütçe aşımı veya context sorunu örnek azaltılarak gizlenmez.

Prompt 4'ün eski TR YES/NO talimatı kullanıcının istediği EVET/HAYIR ile
değiştirilir; bütün öncüller, bağlam ve soru aynı kalır. Yeni sistem/kullanıcı
mesaj SHA'sı, frozen prompt SHA'sından ayrıdır. 64 token heuristiği yeni
chat sayımı değildir. Gerçek OpenAI count endpoint ve HF/vLLM chat tokenizer
ölçümleri ayrı gerçek usage alanlarıyla kaydedilir.

Kalıcı sıralama: task registration → inflight checkpoint ve fsynced raw
request → provider → exclusive raw response/error → normalized outbox ve
terminal checkpoint. Crash sonrası tamamlanmış response yeniden normalize
edilir; yeni istek gönderilmez. Eksik/bozuk terminal ham kayıtlar reddedilir.
Yanıtı belirsiz bir call için açık resolve/reason/duplicate-risk acknowledgement
gerekir. Başarı, invalid ve provider failure ana kayıtları korunur.

Normalized JSONL yalnız eklenir; kesintide yazılmış doğru satır prefix'inin
eksik suffix'i outbox'tan eklenebilir. Eski byte'lar değiştirilmez. Cache
envelope hash'i source/response atfını bağlar; cache yeni bağımsız gözlem
sayılmaz. Ham response hiçbir zaman yeniden yazılmaz.

Full onayı gerçek, tam, cache kullanılmamış pilot ve frozen veri SHA'sını
onaylayan kullanıcı kaynaklı dilsel review artefaktına bağlıdır. Onaylanan
pilot/normalized/review hashleri, model/prompt/parser/deney eksenleri kontrol
edilir; onay full run'ın değişmez kayıtlarına da kopyalanır. Dry-run veya
fixture'lar bu kapıyı geçemez.

Resmî API kaynakları: [OpenAI Docs token counting](https://developers.openai.com/api/docs/guides/token-counting),
[reasoning](https://developers.openai.com/api/docs/guides/reasoning),
[rate limits](https://developers.openai.com/api/docs/guides/rate-limits),
[HF chat templates](https://huggingface.co/docs/transformers/chat_templating),
[vLLM OpenAI-compatible serving](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/).
Denetlenmiş gerçek test ve kurulum sonuçları PROGRESS.md'de kayıtlıdır.

## 10. Prompt 6 — deterministik CBE analizi

Dağıtım ve analiz 0.6.0; inference 0.5.0 ve önceki 43 Python kaynağı değişmedi.
Analiz sözleşmesi [ANALYSIS.md](ANALYSIS.md), uygulamadan önceki kararlar protokol
§16'dadır. Önceki §8 B=5000 taslağı bu kullanıcı talimatıyla B=2000 oldu.

`requirements-analysis.lock` 12 paket/485 resmî PyPI artifact hash'iyle
Matplotlib ve bütün transitive bağımlılıklarını sabitler. Kullanım için önce
requirements-inference.lock ve requirements-tokenizers.lock gerekir. Hash
kilitleri ayrı tutulur; temel metrik/istatistikler yalnız stdlib kullanır.
CPython 3.12.14 ve 3.13.1 macOS arm64 ortamları ayrı hedeflere çevrimdışı,
--require-hashes ile kuruldu; 24 wheel resmî hashleri ve tüm aktif bağımlılık
kenarları doğrulandı. Diğer platformlar henüz denenmedi. Model indirilmedi.

Planlı tasklar frozen banktan tekrar kurulur; active inference sürümü ve
ham/outbox/task/parser tutarlılığı denetlenir. Analiz mevcut ortak run kilidini
alır ve SQLite mode=ro kullanır; hiçbir inference artefaktı onarılmaz/ezilmez.
Eksik tasklar ve invalid cevaplar paydalardan çıkarılmaz. Cache, fixture ve
çakışan run/cohort girişleri CLI'de reddedilir. Yalnız replicate=0 ana analizdir.

Aynı girdi sırası değişse de kararlı küme sıralaması, n tabakaları ve SHA ile
türetilmiş alt seed aynı bootstrap çıktısını verir. Herhangi bir tabakada tek
küme veya tanımsız tekrar varsa aralık null olur; kapsam kaydı korunur. Çıktı
manifesti analiz/config/input/kod hashlerini, ortamı, paydaları ve çıktı hashlerini
bağlar. Grafik metadata/font ayarları sabittir; formatların byte eşitliği aynı
kilitli ortamda test edildi, farklı platformlarda garanti edilmez.

Gerçek sonuç yoksa sadece no_model_results manifesti yazılır; teslimdeki kayıt
performans tablosu/p-değeri/grafik içermez. Sentetik unit/integration çıktıların
tamamı geçici dizinlerde oluşturulup silindi. Güncel test/kurulum sayıları ve
dosya envanteri PROGRESS.md Prompt 6 bölümünde kayıtlıdır.

## 11. Prompt 7 — bağımsız lexical özelliklerin yeniden üretimi

Lexical modül/dağıtım 0.7.0; önceki 50 Python kaynağı byte düzeyinde aynı
kalır. Model sonuçları olmadan üretilen artefaktlar yalnız korpus/tokenizer
preprocessing'dir. Kullanım, paydalar ve sınırlamalar LEXICAL_ANALYSIS.md'dedir.

- UD Turkish-BOUN r2.18 immutable revision 0def94a229dcbc28d32192b0138b560c075c6d87,
  üç kaynak CoNLL-U ve orijinal CC BY-SA 4.0 README/LICENSE ile teslimdedir.
  corpus_manifest.json bütün dosyaları SHA-256/byte/URL ile bağlar.
- Qwen iki immutable tokenizer snapshot'ının beşer dosyası ve Apache-2.0 lisansı
  teslimdedir. Local SHA-256, Git blob SHA-1 veya LFS SHA-256 ve template pinleri
  sayımdan önce doğrulanır. Model ağırlığı indirilmez. Llama HTTP401 durumu
  açık metadata'dır; yerine başka tokenizer kullanılmaz.
- requirements-lexical.lock 20 tam pin/674 resmî PyPI release artifact hash'i
  içerir. 40 CP312/313 wheel ve iki bağımsız import ortamı doğrulandı. Ağsız
  ölçüm socket/urllib guard altında 0 network attempt verdi. Kütüphane kurulumu
  ve daha önceki kaynak edinimi bu guard'ın kapsamı dışındadır; kendileri
  kaynak URL/hash kayıtlı ayrı adımlardır.
- 18 frekans satırı, 216 bilingual vocabulary ve 1.458 prompt satırı ölçüldü.
  243 kaynak prompt yalnız n=4 önizlemesidir. Dört tokenizer yapılandırmasında
  972 plaintext ve üç Qwen yapılandırmasında 729 local-chat sayımı vardır.
  İki Python'da ham token ölçüm payload'u byte eşittir. Aynı ortam/formatsürümü
  dışındaki byte eşitliği garanti edilmez.
- Literal renderer predicate surface sayımları sıfır, 14 canonical kökün lemma
  sayımları pozitiftir. Yüzey MWT ve lemma bileşenleri ayrı paydadır; smoothing
  yoktur. Bu derlem modelin eğitim dağılımına veya sözcüklerin anlamına yönelik
  bir tahmin değildir.
- Model analizi yalnız doğrulanmış gerçek run'lara açılır; planlı invalidler,
  eksikler, ortak-valid paydalar ve latent kimlikler korunur. 18 tasarım hücresi
  regresyon tasarımında, 14 benzersiz lemma ise bağımsız lexical kapsam
  sayımında ayrıdır. Naif IID p-değeri/SE üretilmez.

Preprocessing manifesti corpus/asset/code/config/input/çıktı SHA'larını bağlar;
asset provenance bir kez saklanır ve her satır digest ile referans verir.
Katalog çıktısı analysis/lexical_bias/ altında aynen kopyalanmıştır. Model
performans grafiği figures/lexical içine yerleştirilmedi; durum açıklaması vardır.
Gerçek inference sonrası yeni bir çıktı dizini ve seçilen model ortamına uygun
review edilmiş yeni run planı kullanılmalıdır. Eğitim, fine-tuning veya bir
sonraki prompt bu aşamada çalıştırılmadı.

## 12. Prompt 8 — reasoning planı ve analizinin yeniden üretimi

0.8.0 uygulama paketi `src/reasoning/` altındadır. İstenen
`src/experiments/reasoning/README.md`, donmuş veri üreticisinin özyinelemeli
kaynak bağını bozmadan bu ayrımı açıklar. Önceki 57 Python kaynağı, dataset
manifestleri, SQLite bankası, model registry, lexical çıktılar ve eski P5
dry-run planları değiştirilmedi. Eski bir testin 50-kaynak sayımı, asıl 50
dosya kontrolünü koruyup yeni 8 reasoning dosyasını da koruma snapshot'ına
alarak güncellendi; üretim kodundaki hiçbir hash koşulu gevşetilmedi.

Tam test takımı **799 test** ile Python 3.12.14 ve 3.13.1 üzerinde ayrı ayrı
geçti; sırasıyla 105.506 ve 110.554 saniye, sıfır hata/skip. Önceki 706 teste
93 yeni reasoning testi eklendi. Yeni testler frozen frame'in 243 koordinatını,
A/D byte eşitliğini, strict final marker'ı, dört effort taşıma ayarını,
eşleştirme/payda/duyarlılık analizini, token/truncation tanılarını, immutable
resume/cache/ambiguity/onay yollarını ve geçici sentetik şekil dışa aktarımını
kapsar. Sentetik deneme çıktıları araştırma sonucu olarak teslim edilmedi.

Kurulu 0.8.0 wheel 65 kaynak dosyasıyla doğrulandı; 57 eski dosya aynıdır,
9 entry point vardır. Ağ erişimi kapalı smoke testi, yeni 144-çağrılık örnek
pilot planını ve aynı planın yeniden yüklenmesini, boş reasoning/CBE analizini
ve eski 192/43.200-çağrılık P5 planlarını doğruladı. Gerçek çağrı, ağ denemesi
ve model indirmesi sıfırdır. Yeni dependency yoktur; aynı inference/tokenizer/
analysis hash kilitleri kullanılır. Python 3.11 hâlâ ayrıca doğrulanmadı;
gerçek GPU/model/provider erişimi bu aşamada sınanmadı.

`results/reasoning/plans/pilot.json` ve `full.json` altı **kapalı örnek model**
üzerinden plan üretir, model seçimi değildir. Ortak output cap 2048, sampling
base registry'den, native capability listesi boştur. Pilot n=[4,8], sekiz
latent sıra, 32 sorgu, arm başına 128 prompt, 18 etkin arm ve 2.304 planlanan
üretim çağrısı içerir. Full dokuz n seviyesinde 1.800 latent sıra, 14.200
sorgu, arm başına 56.800 prompt ve 1.022.400 planlanan çağrıdır. Her iki planda
24 native effort hücresi unavailable/sıfır çağrıdır. Output üst sınırları
sırasıyla 4.718.592 ve 2.093.875.200 token; yeni chat sayımları değil input
proxy'leri 1.117.350 ve 6.854.692.842'dir. Genel maliyet bilinmeyen fiyatlar
nedeniyle null kalır; fiyatı bilinen OpenAI altkümesi ayrıca işaretlenir.
Her planın çağrı limiti/kapalı model ve full için pilot-onay engelleri vardır.
**Hiçbiri yürütülmedi.** Dosya/plan/cohort hashleri, hücre dengeleri ve bütçe
özeti `results/reasoning/dry_run_summary.json` içindedir.

```sh
PYTHONPATH=src python3 -m reasoning pilot --plan results/reasoning/plans/pilot.json --dry-run
PYTHONPATH=src python3 -m reasoning full --plan results/reasoning/plans/full.json --dry-run
PYTHONPATH=src python3 -m reasoning analyse --output results/reasoning/NEW_ANALYSIS --figures
```

P8 planları üretildikleri Python/paket/platform ve 29 ilgili kaynak hash'ine
bağlıdır. Yeni ortamda veya kaynak değişince yeni, farklı adlı plan üretilir;
eski dosya elle güncellenmez. Native effort için exact model/sürüm kaynağı,
sampling desteği ve API erişimi ayrıca doğrulanmalıdır. B/C açıklaması ile
native effort ayrı müdahalelerdir. Gerçek pilot sonrası insan onayı olmadan
full çalışmaz. No-model manifesti B=2000/seed=20261025 kaydeder; performans
tablosu veya grafik içermez. Sonraki prompt kapsamına geçilmedi.


## 13. Prompt 9 — çalıştırılmamış SFT/GRPO altyapısı

Eğitim modülü `src/training/` altında, dağıtım 0.9.0'dır. Önceki **65 Python
kaynağı byte düzeyinde aynı** kaldı; frozen SQLite, manifestler, isim/sorgu
kimlikleri, registry ve eski planlar değiştirilmedi. On yeni kaynakla 75
kaynak vardır. İlgili eski testin koruma snapshot'ı yeni modülleri kapsar;
eski 50-kaynak bütünlük iddiasını korur. Protokol §19 koddan önce yazıldı.

Tam takım Python 3.12.14 üzerinde **916 test / 127.472 saniye**, Python 3.13.1
üzerinde **916 test / 133.272 saniye** ile geçti; kayıtlı hata/skip sıfırdır.
Önceki 799 teste 117 test eklendi. **Yedi opsiyonel Torch sayısal/gradient testi
Torch olmadığı için kaydedilip çalıştırılmadı**; sıfır skip onları doğrulanmış
kılmaz. Gerçek autograd, CUDA, adapter eğitimi veya held-out inference smoke'u
yapılmadı. Ağ izni verilmesine rağmen CPU Torch kurulumu shell/Node PyPI
DNS/bağlantı hatasıyla sonuçlandı; bu sürümlerin mevcut olmadığını göstermez.
`results/training/backend_validation.json` kapsamı makinece okunur biçimde saklar.

Apple M1, 7 Metal çekirdeği ve 16 GiB birleşik bellek envanteri
`results/training/hardware.json` içindedir. Bu bellek CUDA VRAM sayılmaz.
CUDA/PyTorch/PEFT/Accelerate ve model ağırlığı yoktur. Bütün profiller ve
registry adayı kapalıdır. Eğitim/generation çağrısı ve optimizer adımı sıfırdır.
GPU saatleri, gerçek tokenlar ve maliyet bilinmediğinden null'dır.

Üç profil ayrı, immutable `results/training/plans/` JSON'larıdır. Pilot 8
latent/32 sorgu/96 query×relation unit ve 8 GRPO unit; ileri iki profil 300
latent/1900 sorgu/5700 unit ve 600 GRPO unit tutar. Eğitim n≤128'de training
partition'ından seçilir. Pilot JSONL 96 grup/384 sunumdur; manifest satır ve
dosya hashlerini bağlar. Testle latent/prompt hash kesişimi sıfırdır. Dört
held-out cohort n, vocabulary ve birleşik aktarımı ayrı tutar. Yüklem/soru
sözlüğü ayrışır, fakat İngilizce bağlamda higher/lower örtüşmesi vardır:
`clean_full_context_holdout=false`. İnsan dilsel değerlendirmesi beklemektedir.

Referans esinli profil SFT 357, paired/JS 713, GRPO 600 adım planlar; adım
kontrolü beş eğitim kolunda 600'dür. Pilot iki adımdır. Eşit adım, eşit token,
GPU zamanı veya benzersiz örnek anlamına gelmez. Bu bütçeler gerçek tüketim
veya öğrenme sonucu değildir. Her plan config/model/dataset ve 75 kaynak
hashini, bütün kurulu dağıtım sürümlerini, engelleri ve bütçeleri kaydeder.
`results/training/dry_run_summary.json` dosya/plan hashlerini birleştirir.

`torch==2.9.0`, `transformers==4.57.1`, `peft==0.17.1`,
`accelerate==1.11.0` doğrudan aday pinleridir. Resmî sürüm metadata'sı ve PEFT
API kaynağı incelendi; bu CUDA kombinasyonu çalıştırılarak doğrulanmadı.
Tam çözümlenmiş transitive/CUDA artifact hash kilidi henüz yoktur. Mevcut
inference/tokenizer/analysis kilitleri değiştirilmedi. Yeni GPU ortamı ve
sürümleri için yeni plan ve açık `--probe-cuda` gerekir; eski planlar elle
ortama uydurulmaz. Kopyalanan yerel ağırlık manifestinin bütün payload hashleri,
model/tokenizer revision ve chat template denetlenir. İndirme/CPU/MPS fallback
yoktur. Native BF16 ve en az 16 GiB boş CUDA eşiği toplam belleğe sığma kanıtı
değildir. Exact input/token ölçümü maksimum 1536'yı aşarsa kesme/düşürme
olmadan çalıştırma durur; yeni sınır yeni karşılaştırma planı gerektirir.

`run` ve `evaluate` varsayılan dry-run'dır. Gerçek işlem yalnız `--execute`
ile bütün kapılar geçerse başlar. Eğitim ham logları, rolloutlar, adapter ve
execution envanteri yeni run dizininde kalır; üzerine yazılmaz. Yarım eğitim
resume edilmez, aborted olarak korunur. Değerlendirme tamamlanmış adapter
payload'unu doğrular, bütün planlı paydaları korur ve kısmi journal'ı gerçek
tam cohort sonucu olarak kabul etmez. R/D/U, kappa ve coverage aynı CBE
modülünden alınır; model/arm/n/query-distance/transfer ayrımı korunur. Holm
ailesi kollar birlikte analiz edilerek düzeltilir. Boş rapor yalnız
`no_training_or_model_results` durumudur; CSV/LaTeX/grafik/p-değeri üretilmez.

Kurulu wheel 75 kaynak SHA'sı, 65 eski kaynak, UTF-8 README, CRC ve **10 entry
point** ile ağsız doğrulandı. Altı training-arm dry-run, base evaluation
dry-run, pilot export/plan ve no-results raporu; eski inference/reasoning/CBE
komutları aynı kurulumda geçti. Ağ denemesi/model çağrısı/ağırlık indirmesi
sıfırdır. GPU stack entegrasyonu bu wheel testinin kapsamı dışındadır.

GRPO örnekleme ve old/current token logp aynı sıcaklıkta hesaplanır.
Tam destekli top_p=1/top_k=0, fresh GenerationConfig ve
use_model_defaults=false, miras decoder warper'larını engeller. JS answer
scoring sıcaklık1'de kalır. Son adım ve rollout arası süre aşımı da denetlenir.
Bu düzeltmeleri kapsayan altı ek test ve bir opsiyonel Torch policy-gradient
testi vardır; sonuncusu çalıştırılmayan yedi backend testi arasındadır.

Formüller, kaynak yöntemden farklılıklar, GRPO normalization altında bonusun
etkisizliği ve komutlar [TRAINING_COMPARISON.md](TRAINING_COMPARISON.md)
içindedir. Altyapı GPU ortamında doğrulama ve pilot hazırlığına uygundur;
gerçek eğitim veya aynen replikasyon tamamlanmış sayılmaz. Prompt 10'a geçilmedi.


## 14. Prompt 10 — doğrulanmış kanıta bağlı bilimsel rapor

Yeni dağıtım0.10.0, raporlama paketi `reports/publication/` altındadır.
`src/` içindeki önceki75kaynak **byte düzeyinde aynı** kalır. Setuptools iki
kaynak kökünü paketler; namespace discovery kapalıdır. Altı yeni publication
modülü, toplam81paketlenmiş kaynak ve11entrypoint vardır. Bu yerleşim,
Prompt9'un bütün-src kaynak fingerprint'ini ve eski planların geçerli dry-run
olmasını korur. Eski üretici, renderer, istatistik veya eğitim kodu değiştirilmez.

```sh
PYTHONPATH=src:reports python3 -m publication --paper-dir paper_next --report-dir reports/next
# Kurulu eşdeğer:
turkish-cbe-paper --paper-dir paper_next --report-dir reports/next
```

Mevcut `paper/` ve `reports/current/` ezilmez. Reporter varsayılan native run
keşfi yapar; belirli run seçimi ve farklı cohort'lar için ayrı rapor şartları
[reports/README.md](reports/README.md) içindedir. Her rapor seed20261025,
bootstrap2000, %95 confidence, latent-instance cluster ve n strata kaydeder.
Girdi ham/normalize/checkpoint kayıtları native okuyucularla doğrulanır;
CSV/JSONL tahmin enjeksiyonu ve test fixture'ları gerçek sonuç kabul edilmez.
Kollar/model/n/query-distance/dil/relation/namecontrol metadata'sı korunur.
Raporlama istatistikleri yeniden tanımlamaz, mevcut analizleri kullanır.

Mevcut rapor **results_pending** durumundadır. 3.000latent/23.200sorgu/
5.637.600prompt/7.819.200karşılaştırma gerçek inşa istatistikleridir. Table1
33split×n satırıdır. Table2/3/4/6/7 birer açık not_run satırı; Table5'te
bir inventory+18korpus+216tokenizer ön işleme satırı ve bir not_run ilişki
analizi satırı, toplam236satır bulunur. Model N/accuracy/R/D/U/CI/p-değerleri
NA/null kalır. Ölçülmemiş sonuç sıfır başarıya çevrilmez. BeşRQ ilgili tabloya
ve bekleyen kanıta bağlanır; hiçbir hipotez desteklendi diye işaretlenmez.

İngilizce sekiz bölüm+birleştirilmiş manuscript+BibTeX, yedi CSV/LaTeX/Markdown
tablo ve iki yöntem şeklinin üçer formatı ile dört eksik-sonuç şekil metni,
**41 paper dosyası** oluşturur. LaTeX parçaları Unicode-capable engine ve
longtable/single-column entegrasyonu ister; tam journal şablonu derlenmedi.
CSV kayıpsızdır; kompakt metrik görünümlerinin stratumID'si CSV koşullarına
başvurur. Wilson aralığı ve bağımlılık yorumu açıkça etiketlenir; cluster
aralıkları ayrıca korunur. Şekiller bilimsel plotting araçlarıyla üretildi;
iki PNG görsel olarak incelendi. Gerçek performans eğrisi üretilmedi.

`reports/current/report_manifest.json`75girdi,82kaynak/pyproject SHA'sını,
ortamı/config'i, RQdurumlarını ve çıktı hashlerini bağlar; kendi hash'ini kendi
içine yazmaz. Girdiler rendering sırasında yeniden kontrol edilir. İki yeni
hedef birlikte yayımlanır; hata halinde tamamlanmış rapor görünümü bırakılmaz.
Ham dosyalara veya training/evaluation kayıtlarına yazılmaz.

Tam testler: Python3.12.14 **997/997**,131.631s; Python3.13.1 **997/997**,
151.570s; kayıtlı hata/skip0. Önceki916teste81raporlama testi eklendi.
Yedi opsiyonel Torch sayısal/gradient testi hâlâ çalıştırılmamıştır; bu sayıya
dahil değildir. Gerçek CUDA/Torch/LoRA/GPU eğitim doğrulaması yapılmadı.
Ağsız0.10.0 wheel kuruldu;81kaynak ve75eski byte-eşit kaynak/11CLI doğrulandı.
Kurulu reporter75girdiyi doğrulayıp41paperdosyasını kaynak çalışmasıyla **aynı
byte'larda** üretti. Önceki inference/reasoning/CBE ve altı training-arm dry-run
aynı wheel'de geçti. Ağ denemesi/model çağrısı/weightsdownload/optimizeradımı0.
Yeni üçüncü taraf bağımlılık veya değiştirilmiş hash kilidi yoktur. Python3.11
hâlâ ayrıca doğrulanmadı; farklı ortamda görsel byte eşitliği garanti edilmez.

İnsan dilsel incelemesi, gerçek model/pilot ve CUDA erişimi beklemektedir.
P4/P5 çoklu dönüşüm, endpoint d=n−1, lexical korpus kapsamı ve ENhigher/lower
örtüşmesi yorum sınırıdır. Kaynak CBE ile uyarlama ayrıdır; ilk Türkçe
benchmark/yayımlanmış deney/aynen eğitim replikasyonu iddiası yoktur.
Raporlama hazırdır; bu taslak deneysel yayın için henüz submission-ready
sayılmaz. Prompt11'e veya yeni deney aşamasına geçilmedi.


## 15. Prompt11 son denetim ve sürüm geçişi

`PYTHONPATH=src:reports python3 -m research_audit --project-root . --output reports/audit/replay_001` yeni, değişmez bir kanıt dizini üretir. `audit_manifest.json`87 kaynak dosyasını, pyproject ve frozen/paper girdilerini SHA-256 ile bağlar. Tam veri/query seed replay yanında stratified gerçek yüzey tekrar üretimi uygulanır; tüm5.637.600 promptun bu aşamada yeniden render edildiği iddia edilmez. Kapsam RESEARCH_AUDIT.md içindedir.

Dağıtım0.11.0 gerekli yürütme ve raporlama düzeltmelerini içerir. Katman API/şema sürümleri aynı kalır; tam kaynak hashleri ve dağıtım sürümü birlikte kaydedilmelidir. Frozen bankaya bağlı generator/renderer/complete_grid kaynakları değiştirilmedi. Prompt5/8/9 planları ve Prompt10 paper/reports/current tarihsel byte'ları korundu; güncel kaynakla execution fingerprint eşleşmezse yeni hedefte yeni plan üretilir, doğrulama gevşetilmez.

`reports/audit/source_migration.json` önce/sonra kaynak hashlerini, `migration_from_0_10.patch` ters uygulanabilir kaynak/test/pyproject farklarını tutar. Tarihsel0.10 kaynağa dönmek için yalnız ayrı kopyada `patch -p1 -R < reports/audit/migration_from_0_10.patch` kullanılabilir. Bu ters geçiş test edilir; çalışan proje geri alınmaz. Makale girdileri ve model olmayan placeholder sonuçlar değiştirilmez.

Yeni kaynakla küçük pilot planları ve üç eğitim bütçe planı `reports/audit/plans/` içinde tutulur. Planlama gerçek eğitim/değerlendirme değildir. Modeller kapalıdır; gerçek kullanım için kullanıcının model/erişim seçimi ve ilgili kaynak/review kapıları gereklidir. Büyük değerlendirmelerde CLI `--linguistic-review` gerçek onaylı review JSON yolunu alır. Şema/kapsam RESEARCH_AUDIT.md içindedir; bu teslimde gerçek approved receipt yoktur.

GPU bağımlılıkları ve7 koşullu Torch testi doğrulanmış sayılmaz. Python3.11 de henüz çalıştırılmadı. Yerel CPU/wheel kontrolü ağı ve model çağrılarını kapalı tutar; gerçek serving sürümünü ve sayısal GPU davranışını doğrulamaz.
