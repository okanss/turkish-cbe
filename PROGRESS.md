# Turkish-CBE ilerleme kaydı

Güncel sürüm: 0.10.0. Prompt 10 sonuçları bu dosyanın sonundadır;
önceki aşamaların kayıtları tarihsel olarak korunmuştur.

## 2026-10-23 — Prompt 1 tamamlandı

Kapsam: proje mimarisi, araştırma protokolü, konfigürasyon ve yeniden
üretilebilirlik iskeleti. Başlangıçta çalışma klasöründe kullanıcıya ait
kaynak kod/dosya bulunmuyordu; mevcut çalışan kod veya araştırma mimarisi
değiştirilmedi. Teslim proje kökü `outputs/turkish-cbe/` olarak seçildi.
Deney, veri üretimi, model API çağrısı, fine-tuning veya sonraki promptun
uygulamasına geçilmedi. Test fixture'ları model sonucu değildir.

## Yapılan işlemler ve alınan kararlar

1. Kullanıcının referansı arXiv v1 özet ve HTML üzerinden doğrulandı:
   [kaynak](https://arxiv.org/html/2610.22894v1), özellikle §3–4 ve Ek B/H.
   Kaynak arşiv kodu ve veri dosyaları indirilip incelenmedi. Özgün CBE
   yöntemi ile bu projenin yeni Türkçe önerileri protokolde ayrıldı.
2. **Uygulama kodundan önce** EXPERIMENT_PROTOCOL.md oluşturuldu. Başlangıç
   belirsizlikleri M01–M12 yazıldı; son incelemede sistem dili için M13
   eklendi. Dosya taslaktır; dış önkayıt veya yayınlanmış sonuç değildir.
3. Hedef görev katı tam sıra olarak sınırlandı. `n` giriş yükü; iç düğümler
   üzerinden `d` uzaklığı ayrı keşifsel kol. İlk aşamada görev üreticisi,
   oracle, renderer, model adaptörü, parser ve metrics kodu yazılmadı.
4. Latent/oracle/yüzey/model/puanlama sorumlulukları ve gelecek dört modülün
   sınırları README'de belirlendi. Kullanılmayan boş modül/sağlayıcı/eğitim
   soyutlaması oluşturulmadı. Veri sözleşmesi veri README'sinde açıklandı.
5. Python `>=3.11,<3.14`, `src/` düzeni, standart kütüphane TOML/dataclass ve
   unittest seçildi. Runtime bağımlılığı yoktur. Paketleme için mevcut test
   ortamıyla eşleşen setuptools 84.1.0 sabitlendi; resmi
   [PyPI wheel SHA-256](https://pypi.org/project/setuptools/84.1.0/) kilide eklendi.
   Bu kilit işletim sistemi/Python/pip veya gelecekteki model ortamını kapsamaz.
6. Üç tam ve bağımsız TOML planı yazıldı; inheritance/env override eklenmedi.
   Dört sorgu hücresinin dengesi, benzersiz kimlikler, türler, desteklenen
   kapsam ve her hücrede tam sayı tekrar seçimi doğrulanır. Aşamaların üretim
   seedleri ayrıdır. `planning_only` sabit, eğitim kapalı, modeller UNSELECTED.
7. Yalnız `validate` ve `manifest` komutları eklendi. Plan manifesti ham ve
   normalize config hashleri, dosya hashleri/ağaç özeti, Python/paket/platform,
   varsa commit/dirty bilgisi içerir. Dataset/template null; model sonucu yok.
   Secret/environment dökümü okunmaz; kaynak symlinkleri reddedilir; var olan
   manifestin üzerine yazılmaz. Config baytları bir kez okunarak hash ve
   normalize ayarın aynı girdiye dayanması sağlandı.
8. Ana `R_all` ile ortak valid destekli R/D/U ayrıldı. Parser ve bütün
   sunumların tekrar kontrolü yeni proje kararları olarak etiketlendi.
   TR/EN karşılaştırması aynı canonical/converse çiftlerini kullanır;
   Türkçenin ek varyantları dil karşılaştırmasına katılmaz. Ortak bilingual
   sistem mesajı altında kullanıcı problemi dili estimandı belirtildi.
9. Türkçe converse sadece yönlü ifade dönüşümüdür; source P/L'deki bileşik
   reshuffle kopyalanmaz. Ayrı premise-shuffle/syntax kolları ve ortak
   morph_joint müdahalesi açıklandı. Saf biçimbilimsel nedensellik iddiası yok.
10. Bağımsız metodoloji incelemesiyle kaynak ayrımları, paydalar ve bütçeler
    kontrol edildi. Sonradan RQ1 eşit K, Holm aileleri, kaynak parser sınırı,
    yeni repeat uzantısı ve ileri token bütçesi açıklamaları netleştirildi.
    Konfigürasyonun hatalı sunum türünde TypeError sızıntısı düzeltildi ve
    regresyon testiyle doğrulandı.

## Oluşturulan dosyalar — tam liste

Hepsi yenidir; önceden var olan kullanıcı dosyası değiştirilmedi. Aşağıdaki
yollar proje köküne göredir; 18 kaynak/belge/konfigürasyon dosyası vardır.

| Dosya | İşlev |
|---|---|
| `README.md` | Kurulum, mevcut/gelecek modül sınırları, plan özeti ve sınırlar |
| `EXPERIMENT_PROTOCOL.md` | Belirsizlikler, kaynak ayrımı, hipotez/değişkenler, deney matrisi ve analiz |
| `REPRODUCIBILITY.md` | Ortam, config/hash, veri/ham yanıt sözleşmesi ve tekrar üretim sınırları |
| `PROGRESS.md` | Bu aşamanın işlemleri, kararları, dosya listesi, testler ve eksikler |
| `pyproject.toml` | Paket, Python aralığı, build pin ve CLI girişi |
| `requirements-build.lock` | Build paketinin sabit sürümü ve wheel SHA-256 değeri |
| `.gitignore` | Ortam/cache/build/secret ve üretilmiş artefakt dışlamaları |
| `configs/pilot.toml` | Küçük pilot taslağı |
| `configs/baseline.toml` | Temel değerlendirme taslağı |
| `configs/advanced.toml` | Yalnız RQ5 prompting için ileri taslak; eğitim kapalı |
| `src/turkish_cbe/__init__.py` | Paket ve sürüm |
| `src/turkish_cbe/__main__.py` | Ağsız plan komutları ve hata çıkışları |
| `src/turkish_cbe/config.py` | Katı konfigürasyon doğrulaması ve bütçe hesabı |
| `src/turkish_cbe/provenance.py` | Plan manifesti, hash ve ortam kaydı |
| `tests/test_config.py` | 24 konfigürasyon/bütçe invariant testi |
| `tests/test_provenance.py` | 10 manifest/hash/çıktı koruma testi |
| `tests/test_cli.py` | 6 CLI kapsam/hata/plan testi |
| `data/README.md` | Gelecek veri ve sonuç artefaktları için sözleşme |

Teslimde ayrıca `outputs/turkish-cbe.zip` kaynak arşivi bulunur. Geçici wheel,
kurulum hedefi/sanal ortam ve smoke-test plan manifestleri üst çalışma
klasöründeki `work/` altında tutulur; kaynak arşivine katılmaz. Paketleme ve
testlerin oluşturduğu build/egg-info/bytecode kalıntıları kaynak teslimden
temizlenir.

## Doğrulamalar ve gerçek sonuçları

Bu tablo **yazılım doğrulamasıdır**; LLM performans ölçümü değildir.

| Kontrol | Ortam | Sonuç |
|---|---|---|
| `PYTHONPATH=src python3 -m unittest discover -s tests -v` | Python 3.13.1 | 40/40 geçti |
| Aynı test takımı | Python 3.12.14 | 40/40 geçti |
| Üç TOML için `validate` | Python 3.13.1 | Hepsi geçerli; execution_ready=false |
| Bütçe | Pilot/temel/ileri | 240 / 55.440 / 63.360 planlı çağrı; hiçbiri gönderilmedi |
| Ağsız wheel üretimi | Python 3.12.14, setuptools84.1.0, pip26.2.2, wheel0.48.0 | Başarılı; build izolasyonu kapalı, mevcut exact backend kullanıldı |
| Wheel yerel hedefe kurulumu + kurulu CLI | Python 3.12.14 | Başarılı; pilot bütçesi 240 |
| Editable kurulum + console komutu | Python 3.12.14, system-site-packages gören geçici venv | Başarılı; pilot bütçesi 240 |
| Plan manifesti ve kaynak hashleri | Python 3.13.1 | Plan üretimi; null dataset/no model results ve kaynak hash kontrolü |
| Son teslim denetimi | 18 kaynak dosyası ve ZIP | Manifest hashleri kaynakla eşleşti; yerel belge bağlantıları, arşiv bütünlüğü ve dosya hashleri doğrulandı |

Kapsanan önemli hatalar: yanlış tür/bilinmeyen alan; yinelenen seed/sunum/model
yuvası; dengesiz dört hücre; hücre içinde kesirli repeat; NaN/sonsuz/bool
sayısal değer; eğitim açma; ileri dışı reasoning; yorum değişiminde raw/resolved
hash ayrımı; kaynak mutasyonu; generated/env dışlama; dış symlink; mevcut
manifesti koruma; bozuk TOML/eksik dosya/olmayan run komutu.

Python 3.11 denenmedi. Temiz ortamda PyPI indirmesi/hash kilidiyle kurulum
çalıştırılmadı; resmi wheel hash metaverisi doğrulandı. Kaynak kontrolü için
Git repository başlatılmadı; commit yokluğu manifestte açıkça gösterilir.
Bu doğrulamalar üretici determinismi, oracle doğruluğu, Türkçe anlam
eşdeğerliği, güç veya gerçek API tekrar üretimi kanıtı değildir.

## Eksikler, riskler ve hazır olma değerlendirmesi

**Sonraki geliştirme için temel hazır; gerçek pilot çalıştırmak için hazır değil.**

- Üretici/oracle/renderer/parser, veri şeması doğrulayıcısı, runner ve metrics
  sonraki aşamada uygulanacak. Şimdiki konfigürasyon doğrulaması veri
  doğrulamasının yerine geçmez.
- Türkçe şablon/minimal çift ve isim ekleri incelemesi yapılmadı. İki bağımsız
  değerlendirici onayı, kayıtlı uzlaşma ve otomatik gold korunum testleri gerekli.
  Onaylanmamış morph_joint kolu temel 7-sunum planını bloke eder.
- Model ve tokenizer snapshotları, veri sürümü ve API yetenekleri seçilmedi.
  Uzak model nondeterminismi kalır; seed aynı cevabı garanti etmez.
- Uç nokta kısa yolları `n` ile çıkarım derinliğini karıştırabilir. İç düğüm
  uzaklık kolu ayrı doğrulanmalı; nedensel kısa yol elemesi iddia edilmemeli.
- Örneklem öneridir; güç/duyarlılık hesabı, lisanslar, bütçe ve dış önkayıt
  tamamlanmadı. Sıklık derlemi ve tokenizasyon özellikleri için kaynak yok.
- Arşiv kaynak koduyla replikasyon kontrolü, veri/model/derlem lisanslarının
  sabitlenmesi ve son yayın paketinin bağımsız yeniden üretimi henüz yapılmadı.

RQ3-uzaklık, RQ4 ve fine-tuning için yürütülebilir konfigürasyon yoktur.
İleri RQ5 dosyası da yalnız plan doğrulamasıdır. Sonraki promptun kapsamına
geçilmedi; bu aşamanın sınırlarında duruldu.


## 2026-10-23 — Prompt 2 tamamlandı

Sürüm 0.2.0; deterministik mantıksal sıralama üretimi. Prompt 1'in çalışan
konfigürasyon/plan/provenance mimarisi korundu. Üç deney TOML dosyası ve eski
CLI davranışı değiştirilmedi. Kullanıcının yeni klasör sözleşmesi gereği
önerilen logic.py yerine src/data_generation/ eklendi. Runtime bağımlılığı
hâlâ yoktur; model/eğitim/rendering aşamasına geçilmedi.

### Uygulanan üretim kararları

- Tam n-1 komşu kenarlı strict total order, random isim ataması ve sıra,
  bağımsız öncül permütasyonu. Dilden bağımsız before/after enum'ları kullanılır.
- Seed/n/örnek/akış adından SHA-256 türetme; names/order/premise/query/fileorder
  akışları ayrıdır. Salted hash(), global random veya zamana dayalı seed yok.
- Pozisyon oracle'ı ve isim/öncül envanterinden bütün zinciri iteratif çözen
  bağımsız grafik oracle'ı. Döngü, çelişki, fork, kopukluk, n-1/adjacency
  ihlali, unknown entity ve öz-sorgu reddedilir. LLM kullanılmaz.
- Her n için 200 temel sıra; dört direction×gold hücresinde 50. Direction,
  sorgu subject'inin latent konumunun forward/reverse olmasıdır; gold ile aynı
  değişken değildir. Her isim grubu içinde de dört hücre eşittir.
- Endpoint 0,n-1; Internal yalnız 1..n-2; Distance-controlled aynı iç konumlarda
  d<=n-3. Varsayılan d iki kuvvetleriyle 512'ye kadar; özel pozitif tam sayılar
  CLI ile seçilebilir. Geçersiz d'ler manifestte görünür ve üretilmez.
- Aynı 200 sıra kol/d düzeyinde kullanılır. ordering_id soyut temel çekilişi,
  instance_id sorgu/adlandırma koşulunu tanımlar. ordering_id isim etiketini
  dışlar: başka adlandırma/çeviri de aynı split grubuna bağlanmalıdır.
- İsim grupları üç eşlenik alias olarak çarpılmadı. Her n'de 68/68/64 gruplama
  dört hücreyi korur; küçük grup dönerek toplamda 600/600/600 temel sıra verir.
  Türkçe 48×32 ad-soyad havuzu=1536; harmonic CVCV=5184; sembolik E0000..E2047.
  Havuzlar sabit/NFC, örnek içi tam isimler eşsizdir. Sıklık derlemi değildir.
- JSONL'de kullanıcının 13 alanına ordering_id, sample_index, query_direction
  eklendi. UTF-8, NFC, sıralı anahtar, sabit ayraç ve LF serileştirme kullanılır.
- Streaming kayıt/blok/dataset doğrulaması; toplam seed, kimlik benzersizliği,
  dört hücre, bağımsız oracle eşleşmesi ve kollar arası temel sıra eşitliği.
- Veri manifesti plan/ortam/kod ve isim havuzu hashleri, geçerli/atlanmış d,
  dosya bytes/SHA-256/sayı/sınıf dağılımı/doğrulama içerir. Timestamp yoktur.
  Aynı yerleşim/ortamda manifest deterministik; çapraz ortamda veri hashleri
  karşılaştırılır. Plan manifesti veriye bağlanmaz; ayrı dataset manifesti var.
- CLI: python -m data_generation generate / verify; kurulu giriş
  turkish-cbe-generate. Mevcut veri/manifest üzerine yazılmaz. Hata halinde
  sadece çağrının oluşturduğu kısmi dosyalar geri alınır.
- Bağımsız incelemede .. içeren yolun containment kontrolünü aşması ve bazı
  isim/mesafe manifest metadatasının doğrulanmaması bulundu; yollar resolve
  edilerek ve ortak tasarım-metaveri kontrolü eklenerek düzeltildi. Her iki
  hata regresyon testine alındı. İncelenmiş kodla bütün veri yeniden üretildi;
  ilk ve son tam üretimin 71 JSONL hashinin tümü aynı kaldı.

### Üretilen gerçek envanter

Seed 11, n=4,8,16,32,64,128,256,512,1024. Her n'de 200 temel sıra;
1.800 temel sıra, 71 JSONL, **14.200 kayıt**, **155.538.017 byte** JSONL.

| Kol | Dosya | Kayıt | True | False | Her direction×gold hücresi |
|---|---:|---:|---:|---:|---:|
| Endpoint | 9 | 1800 | 900 | 900 | 450 |
| Internal | 9 | 1800 | 900 | 900 | 450 |
| Distance-controlled | 53 | 10600 | 5300 | 5300 | 2650 |
| Envanter toplamı | 71 | 14200 | 7100 | 7100 | 3550 |

Her dosya 200 kayıt ve hücre başına 50 içerir. Envanter toplamı bağımsız
örneklem veya birleştirilmiş deney istatistiği değildir. Mesafe kolunda isim
kayıtları: Türkçe 3.532, sentetik 3.520, sembolik 3.548; her türde true/false eşit.
Ayrıntılı n/d ızgarası ve isim kuralları data/README.md'dedir.

### Yapılan kontroller

| Kontrol | Gerçek sonuç |
|---|---|
| Tam unittest takımı, Python 3.13.1 | 87/87 geçti |
| Tam unittest takımı, Python 3.12.14 | 87/87 geçti |
| Önceki aşamanın testleri | 40/40 korunarak geçti |
| Yeni testler | entities 9 + generator 17 + oracle 8 + dataset 13 = 47 geçti |
| Exhaustive oracle | n=2..5,152 permütasyonun bütün farklı ordered pair sorguları ve iki relation; ayrıca n=1024 |
| Tam veri doğrulaması | 14200/14200 kayıt; n-1/adjacency/acyclic/mesafe/gold/oracle kontrolleri geçti |
| Hücre ve grup denetimi | 71/71 blok, her hücre50; ortak sıra/benzersiz instance kontrolleri geçti |
| Aynı seed, iki tam üretim | 71/71 JSONL SHA-256 aynı |
| Python 3.13.1 verify --replay | 71 dosya ve 14200 kayıt aynı byte hashleriyle doğrulandı |
| Python 3.12.14 verify --replay | Aynı teslim verisinin 71/71 hashleri tekrar üretildi |
| v0.2.0 wheel build/yerel kurulum | Ağsız başarılı; Python 3.12.14 ve sabit setuptools84.1.0 |
| Kurulu veri console komutu | seed23 küçük fixture:24 temel sıra,96 kayıt,8 dosya; generate+verify --replay geçti |

Python 3.11 hâlâ test edilmedi. Smoke-test fixture'ı work/ altındadır ve bu
araştırma verisine eklenmedi. Veri/hash doğrulaması model sonucu değildir.
Tam kaynak+veri ZIP'i güncellendi; belge bağlantıları ve arşiv dosya hashleri
son teslim kontrolüne dahildir. Paketleme/bytecode kalıntıları teslimden temizlenir.

### Oluşturulan ve değiştirilen dosyalar

Yeni 12 kod/test/manifest dosyası + 71 JSONL; önceki 9 dosya güncellendi.
Toplam proje envanteri 101 dosyadır (30 kaynak/belge/manifest + 71 JSONL).
Aşağıdaki yollar proje köküne göredir. Geçici test/kurulum dosyaları üst work/
klasöründedir; teslim ZIP'ine alınmaz. outputs/turkish-cbe.zip güncellendi.

Yeni kod/test/manifest:

- `src/data_generation/__init__.py`
- `src/data_generation/__main__.py`
- `src/data_generation/entities.py`
- `src/data_generation/generator.py`
- `src/data_generation/oracle.py`
- `src/data_generation/validation.py`
- `src/data_generation/dataset.py`
- `tests/test_entities.py`
- `tests/test_generator.py`
- `tests/test_oracle.py`
- `tests/test_dataset.py`
- `dataset_manifest.json`

Güncellenen mevcut dosyalar:

- `README.md`
- `EXPERIMENT_PROTOCOL.md`
- `REPRODUCIBILITY.md`
- `PROGRESS.md`
- `data/README.md`
- `pyproject.toml`
- `.gitignore`
- `src/turkish_cbe/__init__.py`
- `src/turkish_cbe/provenance.py`

Üretilen JSONL dosyaları — tam liste:

- `data/latent/endpoint/n_0004.jsonl`
- `data/latent/endpoint/n_0008.jsonl`
- `data/latent/endpoint/n_0016.jsonl`
- `data/latent/endpoint/n_0032.jsonl`
- `data/latent/endpoint/n_0064.jsonl`
- `data/latent/endpoint/n_0128.jsonl`
- `data/latent/endpoint/n_0256.jsonl`
- `data/latent/endpoint/n_0512.jsonl`
- `data/latent/endpoint/n_1024.jsonl`
- `data/latent/internal/n_0004.jsonl`
- `data/latent/internal/n_0008.jsonl`
- `data/latent/internal/n_0016.jsonl`
- `data/latent/internal/n_0032.jsonl`
- `data/latent/internal/n_0064.jsonl`
- `data/latent/internal/n_0128.jsonl`
- `data/latent/internal/n_0256.jsonl`
- `data/latent/internal/n_0512.jsonl`
- `data/latent/internal/n_1024.jsonl`
- `data/latent/distance_controlled/d_0001/n_0004.jsonl`
- `data/latent/distance_controlled/d_0001/n_0008.jsonl`
- `data/latent/distance_controlled/d_0002/n_0008.jsonl`
- `data/latent/distance_controlled/d_0004/n_0008.jsonl`
- `data/latent/distance_controlled/d_0001/n_0016.jsonl`
- `data/latent/distance_controlled/d_0002/n_0016.jsonl`
- `data/latent/distance_controlled/d_0004/n_0016.jsonl`
- `data/latent/distance_controlled/d_0008/n_0016.jsonl`
- `data/latent/distance_controlled/d_0001/n_0032.jsonl`
- `data/latent/distance_controlled/d_0002/n_0032.jsonl`
- `data/latent/distance_controlled/d_0004/n_0032.jsonl`
- `data/latent/distance_controlled/d_0008/n_0032.jsonl`
- `data/latent/distance_controlled/d_0016/n_0032.jsonl`
- `data/latent/distance_controlled/d_0001/n_0064.jsonl`
- `data/latent/distance_controlled/d_0002/n_0064.jsonl`
- `data/latent/distance_controlled/d_0004/n_0064.jsonl`
- `data/latent/distance_controlled/d_0008/n_0064.jsonl`
- `data/latent/distance_controlled/d_0016/n_0064.jsonl`
- `data/latent/distance_controlled/d_0032/n_0064.jsonl`
- `data/latent/distance_controlled/d_0001/n_0128.jsonl`
- `data/latent/distance_controlled/d_0002/n_0128.jsonl`
- `data/latent/distance_controlled/d_0004/n_0128.jsonl`
- `data/latent/distance_controlled/d_0008/n_0128.jsonl`
- `data/latent/distance_controlled/d_0016/n_0128.jsonl`
- `data/latent/distance_controlled/d_0032/n_0128.jsonl`
- `data/latent/distance_controlled/d_0064/n_0128.jsonl`
- `data/latent/distance_controlled/d_0001/n_0256.jsonl`
- `data/latent/distance_controlled/d_0002/n_0256.jsonl`
- `data/latent/distance_controlled/d_0004/n_0256.jsonl`
- `data/latent/distance_controlled/d_0008/n_0256.jsonl`
- `data/latent/distance_controlled/d_0016/n_0256.jsonl`
- `data/latent/distance_controlled/d_0032/n_0256.jsonl`
- `data/latent/distance_controlled/d_0064/n_0256.jsonl`
- `data/latent/distance_controlled/d_0128/n_0256.jsonl`
- `data/latent/distance_controlled/d_0001/n_0512.jsonl`
- `data/latent/distance_controlled/d_0002/n_0512.jsonl`
- `data/latent/distance_controlled/d_0004/n_0512.jsonl`
- `data/latent/distance_controlled/d_0008/n_0512.jsonl`
- `data/latent/distance_controlled/d_0016/n_0512.jsonl`
- `data/latent/distance_controlled/d_0032/n_0512.jsonl`
- `data/latent/distance_controlled/d_0064/n_0512.jsonl`
- `data/latent/distance_controlled/d_0128/n_0512.jsonl`
- `data/latent/distance_controlled/d_0256/n_0512.jsonl`
- `data/latent/distance_controlled/d_0001/n_1024.jsonl`
- `data/latent/distance_controlled/d_0002/n_1024.jsonl`
- `data/latent/distance_controlled/d_0004/n_1024.jsonl`
- `data/latent/distance_controlled/d_0008/n_1024.jsonl`
- `data/latent/distance_controlled/d_0016/n_1024.jsonl`
- `data/latent/distance_controlled/d_0032/n_1024.jsonl`
- `data/latent/distance_controlled/d_0064/n_1024.jsonl`
- `data/latent/distance_controlled/d_0128/n_1024.jsonl`
- `data/latent/distance_controlled/d_0256/n_1024.jsonl`
- `data/latent/distance_controlled/d_0512/n_1024.jsonl`

### Eksikler, metodolojik riskler ve sonraki aşama

Latent veri üretimi/oracle/JSONL doğrulama altyapısı sonraki geliştirmeye hazır.
Doğal dil renderer, Türkçe ek/şablon doğrulaması, insan eşdeğerlik onayı,
model revision/runner/parser ve model ölçütleri henüz yoktur. Gerçek model
pilotu için hazır olduğu iddia edilmez; sonraki prompt başlatılmadı.

- 14200 sorgu 1800 temel sıraya bağlıdır; bağımsız gözlem sayısını şişirmeyin.
  Endpoint, Internal ve d kollarını kontrolsüz havuzlamayın. Her çeviri/sorgu/
  adlandırma türevi aynı ordering_id ile aynı splitte kalmalıdır.
- Internal çiftler uniform seçilir; d dağılımı n ile değişir. Kontrollü d
  kolu ayrı kullanılmalı; sonlu 200 çekilişte bütün konumlar tam dengeli değildir.
- Endpoint kaynak/son kısa yollarına açıktır; n−1 gerçek çıkarım derinliği
  değildir. İç sorgular bütün olası kısa yolların yok edildiğini kanıtlamaz.
- İsim grupları aynı latent sıraya eşlenik üç takma ad değildir; bağımsız koşul
  gruplarıdır. Tam ad/tek sözcük/sembolik uzunluk ve bilinirlik karışabilir.
  Sentetik ses yapısı elle incelenmedi; gerçek sözcükle çakışma mümkündür.
- Seed 11 geliştirme verisidir; train/dev/test ayrımı, dış önkayıt ve model
  test kilidi yapılmadı. Baseline/advanced deney TOML'leri bu veriyle otomatik
  ilişkilendirilmedi. Yayın öncesi veri/model/lisans/güç kapıları devam eder.
- Her tam sıra zinciri graf izomorfisi bakımından aynıdır; rastgele isimler
  veya farklı seedler held-out mantık ailesi/evrensel yeterlik kanıtı değildir.


## 2026-10-23 — Prompt 3 tamamlandı

Paket/renderer sürümü 0.3.0; latent generator sürümü 0.2.0 olarak korundu.
Türkçe yüzey, morfoloji, gerçek metinden normal forma doğrulama ve insan
kontrol raporu uygulandı. Önceki üç deney TOML'ü, latent üretici/oracle/veri
şeması, 71 JSONL ve dataset_manifest.json değiştirilmedi. Sonraki prompta,
model inference'a, eğitim veya model-sonuç analizine geçilmedi.

### Renderer öncesi ve uygulama sırasında alınan kararlar

1. Renderer kodundan önce protokole §12 eklendi. Dokuz ilişki, sabit eksen,
   görev/kişi seçimi, P3 odak etkisi, P4 çoklu müdahalesi, P6 ispatı, sembolik
   adların okunuşu ve insan onayı sınırı yazıldı. Özgün CBE yöntemiyle yeni
   Türkçe deneyler ayrı kalır; bu uygulama kaynak protokolün replikasyonu değildir.
2. İlişkiler: ön/arka, sağ/sol, üst/alt; geliş ve tamamlanma zamanı; boy, hız,
   ağırlık; üst/alt sıra. Ortak before/after normal form kullanılır. Mekânsal
   referans ekseni sabit, üst/alt yalnız düşey konumdur ve temas dışlanır.
   Tamamlanmada etiket bir görev adıdır; kişi kendisi tamamlanmış sayılmaz.
3. P0 kanonik. P1 yalnız isim/öncül listeleri. P2 her n-1 öncülde argüman
   takası ve converse; kısmi rewrite uygulanmadı. P3 aynı sözcük/eklerle devrik
   tamamlayıcı–yüklem–özne sırası. P4 iyelikli özellik yapısı: morphology,
   syntax ve lexical olarak açık çoklu dönüşüm. P5 P1+P2+P4. P6 varsayılan
   kapalı; raporda include-negation ile açıkça etkinleştirildi.
4. P6 ispatı: farklı a,b için strict total order tam olarak before veya
   after sağlar; not after(a,b) iff before(a,b). Olayların gerçekleşmesi,
   karşılaştırılabilirlik ve eşitlik yokluğu prompt bağlamında zorunludur.
   Serbest olumsuzlama veya farklı kip/tie durumuna genellenmez.
5. Morfoloji: proper-name ilgi -(n)In, ayrılma -DAn, ünlü uyumu, n tamponu,
   sert ünsüz sonrası t; özel ad yazımı korunur, ASCII kesme standarttır.
   Ön/arka/sağ/sol/üst/alt iyelik+bulunma+bildirme biçimleri kapalı sözlüktür.
   Ağırlık→ağırlığı gibi ortak ad biçimleri sabit şablonlarda yer alır.
   Türkçe I/İ/ı/i ve NFC desteklenir. Edddd etiketi harf ve rakamlar tek tek
   okunarak çekilir; okunuş yönergesi promptta görünür. Genel Türkçe analiz
   veya sınırsız özel ad telaffuzu iddiası yoktur; Alp yerel istisnası, âîû için
   açık pronunciation gerekliliği belgelendi.
6. TDK büyük/küçük ünlü uyumu ve kesme işareti resmi kaynakları kontrol edildi.
   Kaynaklar protocol ve insan rehberinde bağlıdır; şablon onayı değildir.
7. Yüzey cümleleri renderer'dan ayrı yazılmış kapalı gramerle parse edilir;
   kendine ait pozitif/ters/iyelik/olumsuz şablon tablosu vardır. Gerçek metin,
   ek, kaynak normal form ve değişmeyen sorgu kontrol edilir. İki exact oracle
   doğru cevabı yeniden hesaplar; stored normal-form/gold iddialarına güvenilmez.
8. Varyant stili, argüman yönü, iki listenin sürüm/seed/kaynak kimliğiyle
   bağımsız hesaplanan tam permütasyon planı, prompt bileşimi, kaynak/prompt/
   sunum hashleri ve factor_axes/transformations metadata'sı doğrulanır.
   Yanlış faktör etiketi ve izin verilmeyen sıralama değişimleri reddedilir.
9. Otomatik sunumlar human_review_status=pending ve
   eligible_for_confirmatory_evaluation=false kalır. Visible prompt yalnız
   bağlam, karışık isim listesi, öncüller ve sorguyu içerir; gold/latent order/
   sample_index/kimlik metadatası modele gösterilmez.
10. İnsan kontrol tasarımı: önceki veri manifestinin altı hash-kontrollü
    n=4/8 endpoint/internal/distance-d1 bloğundan 36 farklı kaynak sıra;
    dokuz ilişki × dört query-direction/gold hücresi. Her kaynak P0–P6 ile
    eşlenir. Üç isim ve üç sorgu türü dengelidir. Bu, ilişkiler arası başarı
    ölçmek için oluşturulmuş tam çapraz deney değildir.
11. CSV çift prompt/öncül/sorgu, otomatik normal-form hash ve oracle sonucunu
    içerir. İki değerlendiricinin dilbilgisi, doğallık, eşdeğerlik, morfoloji,
    temas belirsizliği; uzlaşma ve not alanları boştur. İnsan incelemesi yapılmış
    gibi gösterilmedi. Yeniden üretim komutu ve metadata manifesti eklendi.
12. Entegrasyon testleri semantic-parser'daki spatial soru parçacığının tüm
    konumlarda midir olarak sabitlenmesini yakaladı; önünde/üstünde midir ve
    arkasında/sağında/solunda/altında mıdır ayrımı düzeltildi. Bağımsız soru eki
    fixture'ı eklendi. İncelemede P0 izinsiz isim sırası ve P1/P5'te eksik
    permütasyonun kabulü bulundu; bağımsız liste planı doğrulaması ve regresyon
    testleri eklendi. Yayımlanacak rapor son kod hashleriyle yeniden üretildi;
    CSV byte hash'i değişmedi.
13. Önceki model-planı TOML etiketleri yeni P0–P6'ya sessizce bağlanmadı:
    eski premise_shuffle yalnız öncülü, yeni P1 her iki listeyi değiştirir.
    İlk taslaktaki uniform kısmi converse ile yeni tüm-öncül P2 ayrımı protokole
    yazıldı. Inference planı eşlemesi sonraki aşamanın ayrı kararıdır.

### Gerçek çıktılar ve doğrulamalar

252 tam Türkçe prompt çifti, 36 benzersiz kaynak ordering_id, 9 ilişki,
4 ilişki ailesi, 7 varyant, 3 isim türü, 3 sorgu kolu. Her ilişki 28, her
varyant 36, her isim türü ve sorgu kolu 84 satırdır. Dört yön×gold hücresinin
her biri 63 satır; true 126/false 126 ve n=4/n=8 126'şar satırdır.
**Tamamlanmış insan incelemesi: 0.**

| Kontrol | Gerçek sonuç |
|---|---|
| Tam unittest, Python 3.13.1 | 149/149 geçti |
| Tam unittest, Python 3.12.14 | 149/149 geçti |
| Önceki aşama testleri | 87/87 korunarak geçti |
| Yeni testler | morphology 13 + renderer 16 + semantic 22 + report 11 = 62 geçti |
| Renderer matrisi | 378 ilişki×varyant×isim×gold kombinasyonu normal form ve iki oracle ile geçti |
| Büyük zincir | n=1024 P0/P2 üretimi ve normal forma dönüşü geçti |
| Morfoloji | Bütün mevcut isim havuzları, Türkçe karakter/NFC/I, ses uyumu, buffer/sertleşme, özel ad yazımı ve 10 rakam sonu fixture'ları geçti |
| Faktör ayrımı | P1 yalnız listeler; P2 yön; P3 sözcük multiset'i; P4/P5 çoklu etiket; P6 opt-in/strict şartı geçti |
| Bozuk veri | Yanlış ek, fazla/eksik/ters öncül, sorgu/bağlam/tie/kimlik/hash/faktör/liste planı değişimleri reddedildi |
| CSV gerçek metni | 252/252 aynı normal form ve değişmeyen gold; tüm insan alanları boş/pending |
| Tekrar üretim | Python 3.13.1, Python 3.12.14 ve kurulu wheel console CSV byte hash'i aynı |
| v0.3.0 wheel | Ağsız build + yerel kurulum + yeni review console 252 satırla geçti |
| Önceki veri | 71/71 latent JSONL SHA-256 aynı; generator ve dataset manifesti korunuyor |

CSV SHA-256: e98e4081647d9304878b16f0e96a2d978206375a44b95c5e599c42cfaf74b1f4.
Rapor manifesti CSV/kaynak veri manifesti/altı kaynak JSONL/renderer-normalizer
kod hashleri, seed 31, ortam ve dağılımları içerir. İnsan kararları veya gerçek
model sonucu içermez. Python 3.11 henüz test edilmedi; v0.3.0 editable kurulum
ayrıca tekrarlanmadı. Geçici smoke/kurulum/ilk rapor dosyaları üst work/ altında.

### Yeni ve güncellenen dosyalar — tam liste

13 yeni dosya, 8 güncellenen mevcut dosya. Önceki 71 veri dosyası dahil toplam
proje 114 dosyadır; tamamı güncellenen outputs/turkish-cbe.zip içindedir.
Paketleme/bytecode kalıntıları teslimden temizlenir. Yollar proje köküne göredir.

Yeni:

- `src/renderers/__init__.py`
- `src/renderers/morphology.py`
- `src/renderers/turkish.py`
- `src/renderers/review.py`
- `src/validation/__init__.py`
- `src/validation/semantic_equivalence.py`
- `tests/test_morphology.py`
- `tests/test_turkish_renderers.py`
- `tests/test_semantic_equivalence.py`
- `tests/test_review_report.py`
- `data/validation/README.md`
- `data/validation/turkish_samples.csv`
- `data/validation/turkish_samples_manifest.json`

Güncellenen:

- `README.md`
- `EXPERIMENT_PROTOCOL.md`
- `REPRODUCIBILITY.md`
- `PROGRESS.md`
- `data/README.md`
- `pyproject.toml`
- `src/turkish_cbe/__init__.py`
- `src/turkish_cbe/provenance.py`

### Eksikler, metodolojik riskler ve hazır olma

Türkçe renderer ve otomatik kontrol altyapısı sonraki geliştirmeye hazırdır.
**Gerçek model değerlendirmesine henüz hazır değildir.** Model inference,
İngilizce renderer, cevap parserı/ölçüt, fine-tuning ve sonraki prompt uygulanmadı.

- İnsan doğallık/dilbilgisi/ek/eşdeğerlik incelemesi ve gerekçeli çift
  değerlendirici onayı bekleniyor. 252 örneğin hazırlanması, her son deney
  hücresinden en az 20 örnek inceleme kapısını veya insan onayını tamamlamaz.
- P3 gramatik devrik cümle ile odak/pragmatik yükü değiştirir. P4 ve P5 saf
  morphology etkisi değildir; özellik adları ve sözdizimi de değişir.
- Üstünde/altında fiziksel temas okuması bağlamda dışlansa da insan özellikle
  bu çağrışımı değerlendirmelidir. Olayın gerçekleşmesi/tie varsayımları
  kaldırılırsa P6 ispatı geçersizdir.
- Kapalı metin grameri genel Türkçe çözümleyici değildir. Renderer ve parser
  proper-name ekleri için aynı morphology fonksiyonlarını kullanır; bağımsız
  beklenen-biçim fixture'ları ve insan kontrolü ortak hata riskini azaltmalıdır.
  Destek dışı ad/okunuşlara serbest genelleme yapılmaz.
- Edddd digit okunuşu ek yönerge ve token uzunluğu getirir; isim grubu etkisi
  saf morphology/dil etkisi gibi yorumlanmamalıdır. Sentetik adların sözcük
  olma ihtimali ve iki sözcüklü tam ad karıştırıcıları devam eder.
- Rapor amaçlı 36 kaynak bağımsız model deneyi değildir; yedi varyant kaynak
  sıra düzeyinde bağımlıdır. ordering_id bütün çeviri/isim/sorgu türevlerinde
  aynı split grubunda tutulmalıdır. Model revision, önkayıt, güç/lisans/bütçe
  ve kesin renderer-plan eşlemesi sonraki model aşamasında dondurulmalıdır.

## Aşama 4 — İngilizce kontrol ve eşleştirilmiş veri (2026-10-23)

### Kapsam ve tasarım kararları

Prompt 4 uygulandı; model inference, puanlama, fine-tuning ve sonraki prompt
başlatılmadı. Koddan önce EXPERIMENT_PROTOCOL.md §13'e belirsizlikler ve
alınan kararlar yazıldı. Önceki generator 0.2.0, morphology/TR renderer ve
semantic validator 0.3.0 değişmedi; yeni dataset/English katmanı 0.4.0'dır.

- `latent_instance_id=ordering_id` bütün sorgu/dil/isim/sunum ailesinin split
  anahtarı, eski `instance_id` ayrı `query_instance_id` oldu. Eğitim ve test
  alias veya sözcük değiştirerek aynı sırayı paylaşamaz. Seedler 401–404.
  Daha önce açık seed 11 yalnız development guard'ıdır, frozen test değildir.
- Test 200/n, dokuz n, pilot 50/n, altı n; tanımsız training/development büyüklüğü
  50/n, dokuz n olarak seçildi. 50'lik örneklerde iki marjin 25/25, hücreler 12/13;
  testte her hücre 50. Bütün seçilen generator sample_index değerleri saklandı.
- Kaynak adlar yalnız Türkçe tam ad; native_shared iki dilde aynı Unicode,
  en_transliterated EN ASCII/TR özgün, symbolic_shared iki dilde aynı Edddd.
  Semboller 2.048 kimliklik havuzdan örneklenir ve alfabetik kaynak ada atanır; ranka göre
  numaralanmaz. Transliterasyon çakışması reddedilir. Sentetik isimler eski
  üreticide desteklenir; bu matched konfigürasyonda ek deney yapılmadı.
- Kaynağa yakın endpoint/front kontrolü A–D `reference_structural` olarak
  kodlandı. İngilizce 31 ilişkinin tamamı kopyalanmadı; mevcut TR ile eşdeğer 9 ilişki
  uygulandı. Yeni equivalent, protokol P'ye yakın uniform konumlu kısmi converse
  maskesi ve iki liste karıştırmasıdır; önceki P2 bütün kenarları çevirir.
  Liste identity ise rotation vardır; permütasyonlar tam uniform değildir.
- E, endpoint/native_shared TR P1–P5'tir; P4/P5 morphology, syntax ve lexical çoklu
  dönüşüm olarak belgelenir. F, aynı sıra/öncül/converse planında farklı
  sorgu karmaşıklığıdır; eşdeğerlik/tutarlılık çifti değildir. Geçerli bütün
  2’nin kuvveti olan d değerleri kullanılır; uçlar interior, d<=n−3.
- Ana tam deney kolu yalnız front/behind; dokuz ilişkinin tamamı pilot endpoint/native
  uzantısındadır. Üretilmeyen Cartesian hücreler açıkça matris dışındadır.
- Grup, source query, gerçek tam prompt ve karşılaştırma ayrı JSONL bankalarıdır.
  A–E aynı sorgu/gold; C/D aynı paylaşılan plan; F query/gold'un değişmesine izin
  verir ve her cevap ayrı oracle ile kontrol edilir. İki dilde gerçek text
  ayrıştırılır; Türkçe Reader eski bağımsız grameri bir inventory kontrolüyle
  kullanır. Bu performans uyarlaması eski dosyalara müdahale etmez.
- Model/tokenizer revision seçilmedi. Unicode referans token, boşluk tokenı,
  karakter ve UTF-8 boyutları ölçülür; model_token_count=null kalır.
  Python 3.12.14 / UCD 15.0.0 dataset ortamıdır. UCD değişirse hashes-only portable,
  deep replay kayıtlı ortam gerektirir. E yalnız TR olduğundan ham dil ortalamaları
  language effect gibi yorumlanmaz; aynı C/D hücresi üzerinden analiz gerekir.
- Exclusive oluşturma, hash, canonical JSONL, beklenen matris tamlığı, source
  seed replay, alias politika/plan replay, iki oracle ve split kontrolleri birlikte
  uygulanır. Kendi yarım çıktıları rollback ile kaldırılır; eşzamanlı başkasının
  manifesti korunur. Frozen durumu insan onayı anlamına gelmez.

### Dosya listesi

Yeni kaynaklar:

- `src/renderers/english.py`
- `src/experiments/__init__.py`
- `src/experiments/dataset_builder.py`
- `src/experiments/names.py`
- `src/experiments/pairing.py`
- `src/experiments/surface_reader.py`
- `src/experiments/token_lengths.py`

Yeni testler:

- `tests/test_english_renderers.py`
- `tests/test_matched_names.py`
- `tests/test_pairing.py`
- `tests/test_surface_reader.py`
- `tests/test_token_lengths.py`
- `tests/test_dataset_builder.py`

Yeni protokol/veri dosyaları:

- `configs/matched_dataset.toml`
- `EXPERIMENT_MATRIX.md`
- `DATASET_CARD.md`
- `experiment_dataset_manifest.json`
- `data/training/groups.jsonl`
- `data/training/latent.jsonl`
- `data/training/prompts.jsonl`
- `data/training/cells.jsonl`
- `data/training/split_manifest.json`
- `data/development/groups.jsonl`
- `data/development/latent.jsonl`
- `data/development/prompts.jsonl`
- `data/development/cells.jsonl`
- `data/development/split_manifest.json`
- `data/pilot/groups.jsonl`
- `data/pilot/latent.jsonl`
- `data/pilot/prompts.jsonl`
- `data/pilot/cells.jsonl`
- `data/pilot/split_manifest.json`
- `data/test/groups.jsonl`
- `data/test/latent.jsonl`
- `data/test/prompts.jsonl`
- `data/test/cells.jsonl`
- `data/test/split_manifest.json`

Güncellenen:

- `.gitignore`
- `pyproject.toml`
- `src/turkish_cbe/__init__.py`
- `README.md`
- `EXPERIMENT_PROTOCOL.md`
- `REPRODUCIBILITY.md`
- `data/README.md`
- `PROGRESS.md`

### Test ve kurulum kanıtları

223 birim testi hem Python 3.13.1 hem Python 3.12.14 üzerinde geçti. Önceki 149 test
korundu; yeni 74 test: English 10, names 11, pairing 13, reader 8, lengths 10,
builder 22. Builder testleri yalnız küçük başarıyı değil, rehash sonrası bozuk
semantic gold, eksik hücre, grup metadatası, alias ve plan, seed ayrımı, deterministik byte
hashleri, dosya üzerine yazmama ve 3 rollback ve eşzamanlı manifest durumu içerir.

Ağsız wheel build ve temiz yerel hedefe kurulum başarılı. Kurulmuş bütün .py
kaynakları proje ile byte aynıdır. Dört giriş komutu çalıştı: planning validate,
eski rapor 252 satır, matched 16 grup / 48 sorgu / 400 prompt / 400 hücre oluşturma ve 20 dosyalı
derin doğrulama. 304 farklı görünür metin vardır; farklı ad kontrollerinde aynı TR metin
tekrarı aynı splitte izinlidir. Bu örnek 400 bağımsız model deneyi değildir.

Wheel 0.4.0: 62.427 bayt; SHA256
`5b291c462515a01c59d825eef8c3dce85874c4961bb3909069d8b686cc7154ca`.
Önceki 252 satırlık kontrol CSV’si byte aynı, SHA256
`e98e4081647d9304878b16f0e96a2d978206375a44b95c5e599c42cfaf74b1f4`.
Seed 11’in 71 JSONL / 14.200 kaydı Python 3.12.14 ile tam replayden geçti.

Kaynak hash envanteri yalnız projenin beş paketini kapsayacak biçimde
sınırlandı. Kurulu ortamın ilgisiz bağımlılıkları hashleri değiştirmez;
eksik paketler ve symlink kaynakları reddedilir. Kurulu hedefe ilgisiz
bir modül eklenerek mini export hem kurulu CLI hem kaynak CLI üzerinden
derin doğrulandı. Bu, kurulum yerinden bağımsız kaynak kimliğini doğrular.


### Kesin veri sonuçları ve son kontrol — 2026-10-24

Üretim Python 3.12.14 ile tamamlandı. Aynı ortamda tüm veri için derin
seed/alias/plan/yüzey/oracle/hücre/split doğrulaması başarılıdır. Python
3.13.1 ile ayrıca portable hash kontrolü başarılıdır. İlk derin doğrulama
bir dosya okuma zaman aşımı (Errno60) nedeniyle kesildi; aynı kaynak ve
veri değiştirilmeden yeniden çalıştırılınca tamamı geçti. Bu bir model
çalışması veya model sonucu değildir.

| Split | Latent sıra | Kaynak sorgu | Tam prompt | Karşılaştırma | Prompt gold TRUE / FALSE |
|---|---:|---:|---:|---:|---:|
| training | 450 | 3.550 | 20.050 | 20.050 | 10.025 / 10.025 |
| development | 450 | 3.550 | 20.050 | 20.050 | 10.025 / 10.025 |
| pilot | 300 | 1.900 | 33.100 | 33.100 | 16.550 / 16.550 |
| test | 1.800 | 14.200 | 80.200 | 80.200 | 40.100 / 40.100 |

Toplam: **3.000 sıra, 23.200 kaynak sorgu, 153.400 tam prompt ve 153.400
karşılaştırma**. Kaynak sorgularda 11.600 TRUE / 11.600 FALSE; promptlarda
76.700 TRUE / 76.700 FALSE vardır. Testin her n/query/d bloğunda dört hücrenin
her biri 50'dir. Pilot/training/development hücreleri 12/13, iki marjin 25/25'tir.
A–F ve bütün tekil hücrelerin gerçek sayıları EXPERIMENT_MATRIX.md ve
manifestin comparison_cell_counts/query_cells/prompt_cells alanlarındadır.

Derin verify: valid=true, files=20, orders=3000, prompt_records=153400,
**145.602 farklı görünür metin**. 7.798 satır, başka bir deney etiketindeki
metni tekrarlar; bu tekrarlar yalnız aynı split içindedir. Özellikle
EN-only transliterasyonun TR tarafı ve n=4 internal/d1 sorguları buna yol açar.
Bu etiketler bağımsız gözlem veya farklı metin gibi değerlendirilmez.

Bütün cümleler normal forma geri eşlendi; iki bağımsız oracle aynı gold'u
verdi. C/D'de ortak sunum planı aynıdır; A–E'de sorgu ve gold eşleşir. F'de
iki farklı sorgu ayrı doğrulanır. Bütün split grupları, alias sıralamaları
ve görünür prompt hashleri splitler arasında ayrıdır; seed11 guard yeniden
denetlendi. Eksik/tekrarlı kimlik, beklenmeyen hücre veya bozuk referans yoktur.

Yeni dondurulmuş veri dosyaları toplam **3.053.164.852 bayt**; manifest 24
sahip olunan Python kaynak dosyasının hashini de içerir. Önceki teslimdeki
106 dosya byte aynı, 8 dosya değişmiş, silinen dosya yoktur. Oluşturulan 37
ve değiştirilen 8 proje dosyasının tam listesi yukarıdadır. Build, egg-info
ve bytecode ara dosyaları teslim ağacından kaldırılmıştır.

Kök matched manifest SHA-256:
`e016f612eebbe582de83792935047ab8ad87e371d62e5dec24bd834f97b89669`

Dosya envanteri birleşik SHA-256:
`3948f5f75145da019a8d45e712b374dc704fb8840a3cbd96a34876ad0900254b`

Çözümlenmiş üretim planı SHA-256:
`ef9677019e10550d10d78cb971eb027277ca0bd776946ac0873bf56f9c567fe4`

Bütün dondurulmuş dosyaların ayrı hashleri (kök manifestte byte/satır
sayılarıyla birlikte aynı değerler kayıtlıdır):

| Dosya | SHA-256 |
|---|---|
| `data/training/groups.jsonl` | `d2e75a63a417113306a786333632ac47252a1f8ebd6b40482a72bc6ef7811984` |
| `data/training/latent.jsonl` | `685aa21b7bc732ba9b70bafeadc1d6900802633e2d6a4e77bb6aa61630b8586b` |
| `data/training/prompts.jsonl` | `8eeac8922ad484bb9b2783cdbb928d66131168164631edea342279717f1bf54b` |
| `data/training/cells.jsonl` | `c10d20e4f5029f8c51bdb7b985799e1fa6acf893cf04fcfa01afbd3b65aaf79b` |
| `data/training/split_manifest.json` | `2bbf81a92a110e3eb8d803400faa4546cb744974237a51ea2514ee31debb683c` |
| `data/development/groups.jsonl` | `1e82eefb7115546dde4bea9cadae749d66413911b6cb73d417d06dff1d62db07` |
| `data/development/latent.jsonl` | `87925d8d4242651c8ee8a0f9b618637589915752a7ada9a16bcc635f77fda361` |
| `data/development/prompts.jsonl` | `0d6d6094d01518324f88558eef06f512ba27d6c5210f724c72c62e401559a8a7` |
| `data/development/cells.jsonl` | `2b55b12c4b54cf9bfd14be9337cda3bab6a0c2f7a496e1dd6e10185afdc1de4b` |
| `data/development/split_manifest.json` | `e15b30e24f7402be3caa0a5fb6ffb26ed47a7cdf1be5eee7524de28e2c53988c` |
| `data/pilot/groups.jsonl` | `f6ae2538f01c848e957c5b73d72f1364fd7e3450141b86619808a504abfddee9` |
| `data/pilot/latent.jsonl` | `505091306b326d336b4229f3c71a92aab1885d34d0fc4d14b7621c8917bcf4e8` |
| `data/pilot/prompts.jsonl` | `55ba1d73f3fd78e09d4b1d742eea242d09e612d3235875c2ddf6d27bb1b94a8b` |
| `data/pilot/cells.jsonl` | `3ceae98e7009fd6f294f81b9dec25a05a6b1c232b3e97355c2d3a4f9c58ddaac` |
| `data/pilot/split_manifest.json` | `5edaad9ad5d872e45eb06b24e8e7fa7d9b1513e8629e1e64fe3b10ead7f410b6` |
| `data/test/groups.jsonl` | `94027aaa493164594c3bd801b3f4d4f212a24640d6b7895ef2eca9d7c2b9a304` |
| `data/test/latent.jsonl` | `4fadfd5fcfafc263fccd7c5220c5e243223cf16e721d85849bd769e151393ded` |
| `data/test/prompts.jsonl` | `9ba05adb16fa256992ed6d4eb4288ad5e87c80f3d3268d6134b79b4dc73f149e` |
| `data/test/cells.jsonl` | `85ad003700c545891865792d279600a1fdf8736705ab563524ce1ab43f4d9a27` |
| `data/test/split_manifest.json` | `c62946836a02d2604987d0519af5d3102ddab092d003c169616b755962ed3ab0` |

### Eksikler, riskler ve sonraki aşamaya hazır olma

Eşleştirilmiş veri hazırlama ve doğrulama altyapısı hazırdır. İnsan dilsel
incelemesi tamamlanmamıştır; bütün matched promptlarda pending/eligible=false
korunur. Eski 252 örneklik rapor yeni EN/TR koşullarının onayı değildir.
Model ve tokenizer revision seçilmediğinden model_token_count=null kalır;
referans Unicode ölçümü BPE veya sağlayıcı chat maliyeti sayılamaz. Gerçek
model değerlendirmesi için insan onayı, model/tokenizer, runner/cevap parserı
ve önkayıtlı analiz planı gerekir.

Kaynağa yakın kol yapısal karşılaştırmadır; isim havuzu, ek bağlam, seed ve
identity-rotation farkı nedeniyle makalenin birebir performans replikasyonu
iddia edilmez. P4/P5 çoklu faktördür. F ve aynı sıranın dil/isim/varyant
kopyaları ayrı bağımsız örnekler olarak toplanmaz. Dokuz ilişkinin tümü pilot
uzantısında bulunur; tam testte bütün faktörlerin Cartesian çaprazı üretilmedi.

Hiçbir inference, eğitim/fine-tuning veya sonraki prompt çalıştırılmadı.


## Prompt 4 ek tamamlama — v0.4.1 — 2026-10-24

Kullanıcı önceki iki eksikliğin tamamlanmasını istedi: gerçek model BPE
token sayımı ve tam testte bütün ilişki/isim/sorgu faktörlerinin çaprazlanması.
Bu ek iş Prompt 4 kapsamında kalır; inference, eğitim veya sonraki prompta
geçilmedi. Yöntem kararları koddan önce protokol §14'e yazıldı.

### Tasarım kararları

- Eski 24 Python kaynağı ve dondurulmuş veri değişmeden korunur. Yeni katman
  ayrı `complete_grid` paketidir; dağıtım/add-on sürümü 0.4.1, eski core 0.4.0,
  generator 0.2.0, Türkçe renderer 0.3.0 olarak kalır.
- Aynı 3.000 kaynak sıra ve 23.200 sorgu kullanılır; her sorgu üç isim
  kontrolü × dokuz ilişki × dokuz geçerli dil/sunum koşuluna açılır. P6 ayrı
  isteğe bağlı koldur. İngilizceye Türkçe morfoloji müdahalesi icat edilmez.
- Yeni E bütün sorgu/isim/ilişki koşullarına genişletilir; eski endpoint/native
  sözleşmesi değiştirilmez. F sorgu karmaşıklığı karşılaştırmasıdır. Kaynağa
  yakın temel kol A–D + spatial_front + endpoint filtresiyle ayrı tutulur.
- Uzun öncülleri tekrar saklamak yerine SQLite'ta kaynak, bileşen ve ölçüm
  kayıtları tutulur. Her koordinatın metni gerçekten üretilir/ölçülür ve
  JSONL'ye kayıpsız açılabilir. 5.637.600 sayısı fiziksel tam-metin JSONL
  satırı veya farklı görünür metin sayısı değildir.
- cl100k_base ve o200k_base gerçek tiktoken BPE sayıları ölçülür. İlgili
  model eşlemeleri tokenizer kontrolüdür; inference modeli seçimi değildir.
  Tam görünür düz metin ölçülür; chat/rol/sistem/araç ek yükü dahil değildir.
- Vocabulary, regex/special-token kuralları, paket sürümleri ve bütün
  bağımlılıklar hashlerle sabitlenir; çalışma zamanı internete bağlanmaz.
  Güvenli prefix/suffix BPE birleşimi tam encode testleriyle doğrulanır;
  güvenli olmayan sınırda tam metin encode edilir.
- İzole süreçlerle paralel hesaplama yapılır; yalnız parent SQLite'a yazar.
  Çıktı sırası kaynak sırasıdır, bekleyen iş kuyruğu sınırlıdır. Worker 1/2
  aynı mantıksal ölçüm hashlerini üretmiştir.

### Oluşturulan dosyalar

- `src/complete_grid/{__init__,grid_design,model_tokens,surface_reader,text_lengths,parallel,dataset_builder}.py`
- `tests/test_grid_design.py`, `tests/test_model_tokens.py`,
  `tests/test_grid_surface_reader.py`, `tests/test_grid_text_lengths.py`,
  `tests/test_grid_parallel.py`, `tests/test_complete_grid.py`
- `requirements-tokenizers.lock`
- `tokenizers/encodings.json`, `tokenizers/cl100k_base.tiktoken`,
  `tokenizers/o200k_base.tiktoken`, `tokenizers/LICENSE.tiktoken`
- `vendor/tokenizers/wheels.json` ve hashleri bu kayıtta bulunan yedi wheel:
  certifi, charset_normalizer, idna, regex, requests, tiktoken, urllib3.
- `data/complete_grid/grid.sqlite`, `data/complete_grid/coverage.json`,
  `data/complete_grid/manifest.json`, `data/complete_grid/pilot_preview.jsonl`.

### Değiştirilen dosyalar

`README.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md`,
`EXPERIMENT_MATRIX.md`, `DATASET_CARD.md`, `PROGRESS.md`, `data/README.md`,
`pyproject.toml`, `requirements-build.lock` (yalnız kapsam yorumu), `.gitignore`.
Önceki 151 teslim dosyasının 141'i byte düzeyinde aynı, 10'u yukarıdaki
belgelendirme/dağıtım dosyalarıdır; 30 yeni dosya eklenmiştir. Silinen dosya
yoktur. Eski kaynak, test, konfigürasyon, latent ve matched veri dosyaları
değişmedi. Ara build/egg-info/bytecode dosyaları teslim kapsamı değildir.

### Tamamlanan otomatik ve kurulum kontrolleri

- Python 3.12.14: **284 test geçti, 0 atlandı** (16,288 saniye).
- Python 3.13.1: **284 test geçti, 0 atlandı** (30,069 saniye).
- Yeni 61 test; tasarım/hücre sayıları, bağımsız yüzey grameri, hızlı
  referans uzunlukları, gerçek BPE, n1024, Unicode/sınır durumları, paralel
  sıra, bozuk gold/ölçüm/SQL görünümü, manifest/schema/hash, güvenli hedef
  sahipliği, JSONL export ve tam replay senaryolarını kapsar. Önceki 223
  test korundu.
- CPython 3.12/macOS arm64 vendor wheel'leri `--no-index --require-hashes`
  ile ayrı boş hedefe ağsız kuruldu. CPython 3.13 için uyumlu bağımlılıklar
  ayrı ortamda kurularak yukarıdaki gerçek BPE testleri çalıştırıldı.
- v0.4.1 wheel ayrı hedefe kuruldu; 31 kurulu Python dosyası kaynakla byte
  eşit, önceki 24 kaynak frozen hashlerle eşittir. Kurulu CLI mini üretimi
  worker 2 ile 16 sıra/48 sorgu/11.664 ölçüm/15.120 karşılaştırma üretti;
  derin replay geçti. 243 exportun her biri tam metin BPE ile doğrulandı.
- Wheel SHA256: `8351d9c2fa307831375dc907ab829db65a34258b3df43d8ee0952aece500bb07`
- Mini mantıksal satır digest'i: `6c799cd026e4187f1ba6918395f184b2faf532919a1ad39d32bf4784ddf244da`
- cl100k_base sözlük SHA256: `223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7`
- o200k_base sözlük SHA256: `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`

### Tam üretim ve son doğrulama

Dört worker ile tam üretim tamamlandı. Gerçek envanter:

| Split | Sıra | Kaynak sorgu | Bileşen | Ölçülmüş prompt | Karşılaştırma |
|---|---:|---:|---:|---:|---:|
| training | 450 | 3.550 | 109.350 | 862.650 | 1.197.450 |
| development | 450 | 3.550 | 109.350 | 862.650 | 1.197.450 |
| pilot | 300 | 1.900 | 72.900 | 461.700 | 634.500 |
| test | 1.800 | 14.200 | 437.400 | 3.450.600 | 4.789.800 |
| Toplam | 3.000 | 23.200 | 729.000 | 5.637.600 | 7.819.200 |

Kaynak sorgularda TRUE/FALSE 11.600/11.600; ölçülmüş promptlarda
2.818.800/2.818.800'dür. Her sorguda 243 tasarım, her grupta 243 bileşen vardır.
A/B/C/D ayrı ayrı 626.400; E 3.132.000; F 2.181.600 karşılaştırmadır.
Bütün hücrelerin sayıları coverage.json ve EXPERIMENT_MATRIX.md içindedir.

Ortam Python 3.12.14 / SQLite 3.53.1 / UCD 15.0.0. Tam matris dosya kimlikleri:

| Dosya | Bayt | SHA256 |
|---|---:|---|
| grid.sqlite | 776.835.072 | `4b47113284862a553086ea402d5755dd08fb6b1bbd99074e85ec05554ba579bd` |
| coverage.json | 573.502 | `60c3b415e67b233e11d07432fa44a0cd271ebfcd9f5d71400071d557af10c8f9` |
| manifest.json | 42.632 | `08e45961a12a1e3a1731fdc29538c6883e41edaf8d6521224507d910e1c59c8f` |
| pilot_preview.jsonl | 535.373 | `f2ce1c316809e1ee9b64045f09d2ab84bbdc390c6be2fe83d3702e06422e53fb` |

Mantıksal ölçüm satır digest'i:
`1bb2f131d3f62422a14244baf91bcefa15286866bd2a66995bbf7b2058c8ece7`.
Farklı platformlarda fiziksel SQLite byte dizisinden ayrı karşılaştırılır.

Pilot önizlemesi tek kaynak sorgunun 243 koşulunu içerir: dokuz ilişki,
üç isim kontrolü, dokuz dil/sunum koşulu. Her satırın tam metni ve iki gerçek
BPE sayısı dışa aktarım sırasında tekrar doğrulandı.

Eski 71 latent dosyanın 14.200 kaydı ve 1.800 sırası bağımsız oracle/denge/
komşuluk/seed replay kontrolünden yeniden geçti. Eski 20 matched dosyanın
portable hash kontrolü geçti; eski 24 kaynak dosya byte düzeyinde aynıdır.

Tam matrisin **derin replay kontrolü başarılıdır**: valid=true, deep=true,
3.000 sıra, 23.200 sorgu, 5.637.600 prompt ölçümü ve 7.819.200 karşılaştırma.
Kaynak gruplar/aliaslar/planlar, bütün sorgular ve iki oracle, bileşenlerin
bağımsız gramerden normal formu, her tam promptun hash'i, iki gerçek BPE
sayımı ve referans uzunlukları yeniden hesaplandı ve tamamı aynı çıktı.
Mantıksal satır digest'i yukarıdaki değeri aynen verdi. Splitler arası
latent/alias/görünür metin hash sızıntısı bulunmadı.

Python 3.13.1 ile bağımlılıksız portable hash/SQL şema/bütünlük/yabancı
anahtar/hücre kontrolü ayrıca geçti: valid=true, deep=false, 5.637.600 kayıt.
Bu kontrol BPE replay yerine sunulmaz; gerçek replay Python 3.12.14'te yapıldı.

Teslim paketi `outputs/turkish-cbe.zip` güncel 181 proje dosyasını içerir;
eski 151 dosya yolu korunur. Paketleme frozen veri/kod hashlerini, dosya
envanterini ve her ZIP girdisinin CRC'sini denetleyerek eski arşivi atomik
günceller. Arşiv SHA256 kimliği ayrı `outputs/turkish-cbe.zip.sha256`
dosyasındadır; kendi hash'ini arşivin içine koyma döngüsü oluşturulmaz.

İlk güncel ZIP paketleme denemesi, önceki teslim verisinin bir dosyasını
okurken filesystem Errno60 zaman aşımıyla durdu. Eski arşiv atomik değiştirme
adımına gelinmediği için korundu; yarım ZIP temizlendi. Veri veya kaynak
değiştirilmedi. Paketleme yardımcı betiğinde aynı byte konumundan sınırlı
okuma tekrarları eklendi; her dosyanın tam SHA256 ve arşiv CRC kontrolü
korundu. Önceki dosyalarda okuma zaman aşımı tekrarlandığı için, değişmediği
daha önce byte/hash kontrolüyle doğrulanan 141 girdi önceki SHA256 kimliği
doğrulanmış frozen ZIP'ten alınır; 40 yeni/değişmiş girdi güncel dosyalardan
okunur. Bu, aynı frozen byte dizisini tekrar sunan bir paketleme kurtarma
yoludur; araştırma verisini veya kodu değiştirmez. Girdiler yine beklenen
SHA256 değerlerine ve son arşiv bütünlük kontrolüne tabidir. Bu hata,
başarılı veri replay/test sonuçlarının yerine raporlanmaz. Paketlemede
kilit dosyasının filesystem mtime değişimi de gözlendi; yalnız mtime farkı
byte değişimi sayılmaz. Böyle bir farkta dosyanın boyutu ve tam SHA256
içeriği arşive yazılan içerikle yeniden karşılaştırılır; gerçek byte farkı
yine reddedilir.

### Kalan metodolojik sınırlar ve hazır olma

İstenen iki eksik tamamlandı: tam çapraz matris ve gerçek tokenizer
ölçümleri hazırdır. Prompt 4 veri hazırlama altyapısı sonraki prompt için
hazırdır; sonraki prompt kendiliğinden çalıştırılmadı.

İnsan dilsel incelemesi pending ve confirmatory eligibility=false kalır;
otomatik doğrulama insan onayı değildir. Python 3.11 test edilmedi; vendor
wheel'leri CPython 3.12/macOS arm64'e özgüdür, diğer ortamlar hash kilidindeki
uyumlu dosyaları indirir. BPE sayımları düz metne aittir; chat/sistem/araç
ek yükü yoktur. P4/P5 çoklu faktör, F ayrı sorgu karşılaştırmasıdır.
Tekrarlanan metin ve aynı latent sıranın türevleri bağımsız gözlem değildir.
Model runner/cevap parserı, model seçimi ve önkayıtlı analiz sonraki
değerlendirme kapsamındadır; inference, eğitim veya model sonucu yoktur.


## Prompt 5 — tamamlanan inference altyapısı ve dry-run — 2026-10-24

Kullanıcı önceki oturumlarda yarım kalan işi sürdürmemi istedi. Açık tercih:
önce dry-run ve altyapı tamamlanacak, gerçek pilot modeli sonra seçilecek.
Önceki parser/config/provider/storage parçaları tamamlandı; eksik cohort
seçimi, plan/bütçe, runner, rate/cache, CLI ve script bağlantıları eklendi.
Gerçek model çağrısı, ağırlık indirmesi, pilot/full, fine-tuning veya sonraki
prompt çalıştırılmadı. Tasarım sözleşmesi koddan önce protokol §15'e yazıldı.

### Prompt 4 koruma kontrolü

Frozen grid manifesti ve 777MB SQLite/coverage dosyaları byte/hash bağlarıyla
planlama sırasında yeniden doğrulandı. Eski 24 core  + 7 complete_grid Python
kaynağının 31'i de frozen hashleriyle aynı; seed, generator, renderer, oracle,
mevcut split ve veri dosyası değiştirilmedi. Önceki tam derin replay sonucu
5.637.600 ölçüm / 7.819.200 karşılaştırma ile valid=true idi; bu devamda
yeniden 20 dakikalık üretim/replay yapılmadı. Aynı frozen artefaktın güncel
dosya ve kaynak hashleri korunuyor.

### Oluşturulan dosyalar — 33

- `src/inference/__init__.py`, `__main__.py`, `config.py`, `dataset.py`,
  `planning.py`, `prompts.py`, `parser.py`, `providers.py`, `storage.py`,
  `runtime.py`, `runner.py`, `cli.py`
- `configs/models.yaml`
- `scripts/run_pilot.py`, `scripts/run_full.py`
- `tests/test_inference_config.py`, `test_inference_parser.py`,
  `test_inference_prompts.py`, `test_inference_providers.py`,
  `test_inference_storage.py`, `test_inference_planning.py`,
  `test_inference_runtime.py`, `test_inference_runner.py`, `test_inference_cli.py`
- `requirements-inference.lock`
- `vendor/inference/pyyaml-6.0.3-cp312-cp312-macosx_11_0_arm64.whl`,
  `vendor/inference/pyyaml-6.0.3-cp313-cp313-macosx_11_0_arm64.whl`, `wheels.json`
- `results/raw/README.md`, `results/normalized/README.md`
- `logs/README.md`, `logs/pilot_dry_run.json`, `logs/full_dry_run.json`

### Değiştirilen dosyalar — 6

`README.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md`, `PROGRESS.md`,
`pyproject.toml` ve `.gitignore`. Önceki 181 proje yolunun hiçbiri silinmez;
175'i değişmeden korunur ve 33 yeni dosyayla toplam 214 proje dosyası teslim
edilir. Paket sürümü/inference 0.5.0, frozen grid 0.4.1 ve önceki bileşen
sürümleri ayrı tutulur. Build/egg-info/bytecode ara dosyaları teslim değildir.

### Model ve istemci tasarımı

Altı YAML örneği kapalıdır: OpenAI tarihsel GPT-4o snapshot, Qwen2.5
0.49/1.54B, Llama3.2 1.23/3.21B ve vLLM Qwen1.54B. Modeller kodda
sabitlenmez; her spec kopyalanır/hashlenir. Model, tokenizer, immutable
revision, template SHA/tarihi, serving, dtype/device, seed, sampling,
reasoning, context/output ve fiyat metadatası kaydedilir. Bilinmeyen veya
desteklenmeyen alan/ayar/model sessizce düzeltilmez ve değiştirilmez.

OpenAI Responses ve vLLM istemcileri tek denemeli stdlib HTTP kullanır.
OpenAI'nin count endpoint'i chat dahil input sayısını verir; Responses seed
desteklemediği için null zorunludur. HF/vLLM gerçek chat şablonunu kullanır;
tarih açıkça sabitlenir. HF rendered input metni ve token ID/hashleri, vLLM
actual tokenize cevabı saklanır. Context taşması kırpılmaz.

OpenAI örneği deprecated olarak işaretli ve erişimi test edilmemiştir.
Resmî karttaki standart tarife 2026-10-24 için USD 2.5 input / 10 output / 1.25
cached input per 1M olarak yapılandırıldı. Bu hesap ücretlendirmesi veya
model kullanılabilirliği garantisi değildir. HF/vLLM fiyat/donanım maliyeti
bilinmiyor, null kalır. İki küçük Qwen tokenizer_config metadata dosyası
yalnız template SHA doğrulaması için alındı; Llama metadata 401/gated idi.
Hiçbir ağırlık indirilmedi.

PyYAML 6.0.3 kilidi 73 resmî release hash'i içerir. Her iki vendor wheel hash'i
resmî PyPI metadata ile doğrulandı, --no-index --require-hashes kurulumları
CPython 3.12/3.13 boş hedeflerinde geçti. Kilit SHA256:
`837e0bb00e1109a701c4820c11a5ff195310a61822c30ba081a48c7582b10d9f`.
Torch/Transformers/vLLM gerçek servisleri kurulu değildir; HF optional extra
uyumluluk aralığı tam GPU/platform kilidi sayılmaz.

### Prompt, parser ve ortak cohort

Kaynak context/entity/premise bloğu ve soru byte düzeyinde korunur. Eski
TR YES/NO cevap talimatı istenen EVET/HAYIR talimatına çevrilir ve iki dilde
sistem mesajı eklenir. source_prompt_sha256 ile yeni messages_sha256 ayrı
kaydedilir; gold ve latent metadata modele gönderilmez.

Kesin parser yalnız dış boşlukları atılmış büyük harfli EVET/HAYIR veya
YES/NO kabul eder. Açıklama, JSON, noktalama, çelişki, başka cevap sözcüğü
dili, refusal, kesilmiş veya boş yanıt invalid kalır. Dili tanımlanan cevap
sözcükleri sınıflanır; serbest metin için genel dil tespiti iddia edilmez.
Invalid cevap modelden tekrar istenmez ve paydadan çıkarılmaz.

Bütün modeller aynı kaynak group/query/design ID'lerini alır. Dört answer
hücresi korunur; endpoint/internal/distance tabakaları ayrı raporlanır.
Planı elle re-hash etmek bütçe sınırını değiştirmeye yetmez: gerçek yürütme
planı/cohort'u/çağrı bütçesini yeniden üretir ve ortam/kodla karşılaştırır.
Kurulu inference kodu proje koduyla farklıysa planlama/yürütme reddedilir.

### Değişmez kayıt ve tekrar çağrı koruması

Her attempt UUID kimliği alır: durable inflight checkpoint → fsynced raw
request → provider → exclusive raw response/error → normalized outbox ve
terminal checkpoint. Raw dosyalar overwrite edilmez. HTTP metadata/count
çağrıları ayrı UUID ile saklanır; auth header/key değerleri yazılmaz.

Resume tamamlanmış cevabı tekrar çağırmaz. Crash sonrası tamamlanmış ham
yanıt varsa tekrar normalize edilir. Sonucu belirsiz çağrı ambiguous olarak
durur; açıklamalı ve kalıcı retry/abandon çözümü gerekir. Sadece kesin
reddedilmiş geçici hatalarda sınırlı exponential backoff/Retry-After vardır.
Retry sınırı dolunca ham hata korunup permanent_failed kaydı üretilir.

Normalized JSONL sadece eklenir; kesintide eksik satır suffix'i outbox'tan
eklenebilir. Hız rezervasyonları endpoint/key-adı scope'unda kalıcıdır;
üretim output rezervi de TPM bütçesine dahildir. Cache varsayılan kapalı,
hash/provenance bağlı, fixture/live namespace'leri ayrıdır. Cached cevap yeni
bağımsız model sonucu değildir; tekrar ölçümü cache ile çoğaltılamaz.

Full için tamamlanmış gerçek, cache kullanılmamış pilot ve açık kullanıcı
onay kaydı gerekir. İlgili model/prompt/parser/deney eksenleri, pilot ham
manifest/normalized hashleri ve gerçek insan review artefaktı bağlanır.
Fixture/dry-run onay kapısını geçemez. Bu teslimde onay üretilmedi.

### Gerçek dry-run sonuçları

| Plan | Latent grup | Prompt/model | Model örneği | Çağrı planı | Gerçek çağrı |
|---|---:|---:|---:|---:|---:|
| Küçük pilot |8|32|6|192|0|
| Temel full |1.800|7.200|6|43.200|0|

Pilot yeni chat input tahmini toplam 52.914, maksimum generated token 6.144;
full tahmini 203.173.686, maksimum 1.382.400'dür. Bu input sayıları kaynak
BPE +64/çağrı heuristiğidir; chat/model tokenizer ölçümü değildir. Gerçek
input/usage alanları tahminden ayrıdır. Bütün altı modelin toplam parasal
maliyeti bilinmiyor. Sadece OpenAI örneği için planlanan token/output-cap
tahmini pilot USD 0.0322875, full USD 86.9597025'tir; gerçek fatura değildir.

Pilot blocker selected_models_disabled; full ayrıca call_budget_exceeded ve
completed_real_pilot_and_explicit_approval_required ile engellidir. Örnek
gruplar bütçeye sığdırmak için azaltılmadı. Test/frozen splitler ayrıdır.

Plan parmak izleri, JSON'un kanonik içeriğinden kendi `plan_sha256` alanı
hariç hesaplanır; dosya byte hash'inden farklıdır.

Pilot kanonik plan parmak izi: `048a05a57685f01d51e964bd25b27f45bfd7516eb4e3e2a39a54da41860aceee`

Full kanonik plan parmak izi: `1c427c2775187b910a252e0e5803ce97efad300579cdb92694741dac2c5964af`

Pilot JSON dosyası SHA256: `5dd1af7b5a527fbd0638b009bd0c9ec830b8c0e6cc35daa77a4183291f033f96`

Full JSON dosyası SHA256: `2484112f8675c73b8bb614cb25a4a49587ea7234e285b868add8e1dd695b9b7a`

### Test ve kurulum sonuçları

- **460 test** Python 3.12.14'te 37.329 saniyede geçti; 0 skip.
- **460 test** Python 3.13.1'de 39.292 saniyede geçti; 0 skip.
- Önceki 284 test korundu; yeni 176 test: config 25, parser/prompts 20,
  providers 39, storage 30, planning 21, runtime 13, runner 22, CLI 6.
- Gerçek frozen önizlemenin 243 tasarımı üzerinde yeni mesajların bağlam/
  öncül/soru korunumu doğrulandı. Gerçek BPE ile n1024 ve önceki bütün
  testler çalıştı.
- Provider testleri HTTP/Transformers fixture'larıdır; gerçek servis
  uyumluluğu veya model performansı sonucuna dönüştürülmedi.
- Runner 22 testinde gerçek frozen prompt/oracle/BPE, geçici proje ve sahte
  provider kullanıldı. Resume, belirsiz çağrı çözümü, 429/backoff/exhaustion,
  cache, language/invalid/refusal/truncation, cohort ve onay kapısı denetlendi.
  Fixture çıktıları geçici klasörlerdeydi; teslim results altında bulunmaz.
- v0.5.0 wheel ağsız kuruldu. Kurulu 43 Python kaynağı proje ile byte
  düzeyinde aynı; önceki 31 kaynak v0.4.1 ile de aynı kaldı. Kurulu CLI,
  pilot 192/full 43.200 planlanan çağrıyı sıfır gerçek çağrıyla doğruladı.
  Ağ bağlantısı test sırasında engellendi. Plansız `--execute` reddedildi.

### Kalan sınırlar ve sonraki aşamaya hazır olma

Prompt 4 veri ve Prompt 5 altyapı/dry-run kapsamı tamamdır. Gerçek pilot
kullanıcının model ve erişim seçimini bekler. API/GPU model uyumluluğu,
Llama gated metadata/template onayı, gerçek vLLM engine/model deployment
pinleri ve platforma özgü ağır bağımlılık kilidi henüz doğrulanmamıştır.
İnsan dilsel review pending, confirmatory eligibility=false kalır.

Uzak API için exactly-once veya seed ile byte eşitliği garanti edilmez;
uncertainty saklanır. Maliyet input tahmini, tarihli tarife, output-cap ve
sağlayıcı usage kapsamına bağlıdır; donanım ve bilinmeyen ek ücretler dahil
değildir. Sonraki model analizi, prompting/fine-tuning veya yeni promptun
kapsamına kendiliğinden geçilmedi.

## Prompt 6 — CBE metrikleri ve istatistik altyapısı — 2026-10-24/25

### Kapsam ve korunmuş önceki aşama

Prompt 6 kodu, kaynak §3/A/B ve protokol §16 kararlarıyla tamamlandı. Gerçek
model çağrısı, model sonucu, fine-tuning veya araştırma performans grafiği yoktur.
Önceki 43 Python kaynağı, frozen veri ve iki Prompt 5 dry-run planının SHA
değerleri korunur. Dağıtım/analiz 0.6.0, inference 0.5.0, grid 0.4.1 ayrıdır.

### Oluşturulan dosyalar — 16

- `src/analysis/__init__.py`, `__main__.py`, `io.py`, `metrics.py`,
  `statistics.py`, `pipeline.py`, `reporting.py`
- `tests/test_analysis_cli.py`, `test_analysis_io.py`, `test_analysis_metrics.py`,
  `test_analysis_statistics.py`, `test_analysis_pipeline.py`, `test_analysis_reporting.py`
- `requirements-analysis.lock`, `ANALYSIS.md`
- `results/analysis/no_model_results/analysis_manifest.json`

### Değiştirilen dosyalar — 5

`README.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md`, `PROGRESS.md`,
`pyproject.toml`. Yeni entry point `turkish-cbe-analyse`, optional analysis
bağımlılıkları ve sürüm 0.6.0 eklendi. Mevcut mimari veya üretici değiştirilmedi.
Önceki 214 yol korunur; 209 değişmez dosya ve 16 yeni dosyayla teslim 230 yoldur.

### Tasarım kararları ve uygulama

1. İstenen 12 ölçüt uygulanır. R/D/U, mean accuracy, raw invariance ve q/kappa
   yalnız ortak valid-pair desteğindedir. Marjinal accuracy ve strict R bütün
   planlı çiftleri içerir; invalid/eksik/provider hatası yanlış sayılır.
   Coverage ve invalid paydaları ayrı döner. V=0 ve q=0/1 tanımsızlıkları null
   ve reason ile korunur; R+D+U=1 ve diğer eşitlikler sentetik olarak tarandı.
2. B=2000, %95 CI, sabit seed ve SHA-türetilmiş alt seed. Tüm bağlı gözlemler
   latent_instance_id kümeleriyle birlikte, n tabakaları korunarak örneklenir.
   Tek kümeli herhangi bir tabaka veya tanımsız tekrarlar CI=null üretir;
   defined_replicates gizlenmez. R için ayrıca Wilson ve bağımlılık etiketi vardır.
3. McNemar discordant≤25 exact, daha büyükse continuity-corrected chi-square.
   Tek latentten çok gözlem varsa sıradan McNemar p-değeri verilmez. Holm
   tasarım/kontrast ailesindeki modeller arasında uygulanır; null testler aile
   büyüklüğünde kalır. Aile tanımı keşifseldir, dış önkayıt iddiası değildir.
4. Model/sürüm, split, isim kontrolü, ilişki, query türü, n/distance ve varyant
   tabakaları korunur. Relation-family özeti cluster bootstrap kullanır.
   TR−EN accuracy ve strict-R, language×presentation/complexity, internal−endpoint
   kontrastları ve betimsel token–invalid korelasyonu ayrı tablolardır.
   TR-only koşullardaki bulunmayan dil kontrastları açık listelenir.
5. Planlı tasklar frozen banktan yeniden kurulur; hiç başlanmamış çağrılar dahil
   eksikler korunur. Input manifest/plan/source, read-only checkpoint outbox,
   ham request/response/error/resolution, terminal/current-attempt ve normalize
   parser tutarlılığı denetlenir. Cache/fixture/çakışan task/cohort reddedilir.
   Ana analiz replicate=0; diğer tekrarlar bağımsız örnek olmaz.
6. CSV/LaTeX tablolar ve altı şekil ailesi PDF/SVG/PNG olarak dışa aktarılır.
   Bilimsel eksenler sabittir; test tanıları facet oluşturmaz. Endpoint n
   eğrisinde d=n−1 yapısal bağı etiketlenir; distance grafiğinde n sabittir.
   McNemar örneklem büyüklüğü `mcnemar_n` adındadır; problem büyüklüğü `n`
   üzerine yazılmaz. Bu iki entegrasyon hatası final kontrolde düzeltilip
   gerçek pipeline çıktısıyla regresyon testine bağlandı.
7. Matplotlib 3.10.7 ve 11 transitive paket, 485 resmî PyPI artifact hash'iyle
   sabittir. 24 CP312/313 macARM wheel SHA denetimi ve aktif bağımlılık kapanışı
   geçti. Temel metrik ve istatistikler stdlib ile bağımsız çalışır; diğer
   platformlar ve gerçek model/GPU ortamı bu doğrulamaya dahil değildir.

### Doğrulama ve gerçek sonuç durumu

- Yeni testler: metrics 15, statistics 33, pipeline 20, IO 33, reporting 22,
  CLI 14; toplam 137. Önceki 460 test korunur.
- Final Python 3.12.14: **597 test geçti, 0 atlandı** (77.116 saniye).
- Final Python 3.13.1: **597 test geçti, 0 atlandı** (80.909 saniye).
- v0.6.0 wheel ağsız kuruldu: 50 kaynak proje ile, önceki 43 kaynak frozen
  v0.5.0 ile byte düzeyinde aynı. Kurulu analiz CLI yalnız no_model_results
  üretti; kurulu pilot/full dry-run 192/43.200 planlanan, sıfır gerçek çağrı
  verdi. Ağ guard denemesi 0. Wheel SHA256:
  `fb2a58250a9f057a47a7f2a6a97c1fff4122510f69c55e8909f49c01809ba708`.
- Gerçek frozen promptlarla 33 IO testi; bütün model yanıtları geçici fixture.
  Sonuçlar gerçek inference kanıtı olarak yayımlanmadı.
- Gerçek build_tables→reporting sentetik smoke: 32 planlı yanıt, 8 profil,
  20 CSV/LaTeX ve altı ailede 72 grafik dosyası doğrulandı. n4/n8 endpoint
  eğrileri birleşir; n8 internal d1/d2 eğrileri n sabit kalır. Geçici sentetik
  çıktılar silindi; kaynak veri/ham cevaplar değiştirilmedi.
- Boş gerçek kayıt klasöründeki CLI çalışması no_model_results, 0 planlı/
  gözlenen/invalid/eksik yanıt, boş tablo ve dosya listesi verdi. Teslimde
  gerçek performans tablosu, p-değeri veya araştırma grafiği yoktur.

### Sınırlar ve hazır olma

Prompt 6 analiz altyapısı hazırdır. Gerçek çıkarım, insan dilsel onayı ve
confirmatory eligibility halen bekler. Null/az kümeli aralıklar ve bağımlı
McNemar sınırlamaları açık raporlanır. Token korelasyonları nedensel değildir;
pointwise aralıklar eşzamanlı güven bantları sayılmaz. Tarihsel protokol taslağı
ve kaynak makale arasında birebir benchmark replikasyonu iddia edilmez.

## Prompt 7 — lexical sıklık, tokenizasyon ve önyargı — 2026-10-25

### Prompt 6 önce tamamlandı

Kullanıcının son talimatıyla önce Prompt 6 finalleştirildi: iki Python'da
597 test, kurulu v0.6.0 wheel, gerçek pipeline→altı grafik ailesi sentetik
entegrasyon kontrolü ve 230 dosyanın SHA/CRC arşiv doğrulaması tamamlandı.
Sonra Prompt 7'ye geçildi; protokol §17 yeni lexical koddan önce kaydedildi.

### Oluşturulan dosyalar — 45

- Kod (7): `src/analysis/lexical_bias/__init__.py`, `__main__.py`, `catalog.py`,
  `corpus.py`, `tokenization.py`, `bias.py`, `statistics.py`.
- Test (7): `tests/test_lexical_catalog.py`, `test_lexical_corpus.py`,
  `test_lexical_tokenization.py`, `test_lexical_bias.py`, `test_lexical_statistics.py`,
  `test_lexical_pipeline.py`, `test_lexical_cli.py`.
- Belge/kilit (2): `LEXICAL_ANALYSIS.md`, `requirements-lexical.lock`.
- Talep edilen katalog klasörü (2): `analysis/lexical_bias/README.md`,
  `analysis/lexical_bias/relation_vocabulary.json`.
- Korpus (6), `data/corpora/ud_turkish_boun/r2.18/` altında:
  `README.md`, `LICENSE.txt`, `corpus_manifest.json`, `tr_boun-ud-train.conllu`,
  `tr_boun-ud-dev.conllu`, `tr_boun-ud-test.conllu`.
- Tokenizer manifesti ve iki snapshot (11): `tokenizers/hf/manifest.json`;
  `models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775/`
  ve `models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/`
  klasörlerinin her birinde `LICENSE`, `tokenizer_config.json`, `tokenizer.json`,
  `vocab.json`, `merges.txt`.
- Sonuç kapsamı (1): `results/lexical/README.md`.
- Betimsel baseline (8), `results/lexical/baseline/` altında:
  `lexical_manifest.json`, `relation_vocabulary.json`; `tables/` altında
  `corpus_frequencies.csv/.tex`, `tokenization_vocabulary.csv/.tex`,
  `tokenization_prompt_preview.csv/.tex`.
- Grafik durum açıklaması (1): `figures/lexical/README.md`.

### Değiştirilen dosyalar — 5

`README.md`, `EXPERIMENT_PROTOCOL.md`, `REPRODUCIBILITY.md`, `PROGRESS.md`,
`pyproject.toml`. Dağıtım/lexical 0.7.0, yeni turkish-cbe-lexical CLI ve
opsiyonel tokenizer-only bağımlılıklar eklendi. Önceki 50 Python kaynağı,
CBE 0.6.0/inference 0.5.0 ve frozen grid 0.4.1 değişmedi. Önceki 230 yolun
hiçbiri silinmedi; 225'i aynı, 45 yeni dosyayla toplam teslim 275 dosyadır.

### Gerçek ölçüm ve tasarım kararları

1. Bağımsız korpus UD Turkish-BOUN r2.18, revision
   `0def94a229dcbc28d32192b0138b560c075c6d87`. CC BY-SA 4.0 lisansı/atfı ve
   orijinal README/LICENSE tutulur; üç train/dev/test parçası yalnız lexical
   sayımdır. Beş kaynak dosya toplam 9.924.171 byte; SHA'ları manifesttedir.
2. Gerçek sayım 9.761 cümle, 121.835 yüzey token, 125.212 syntactic word ve
   3.374 MWT ile yayımlanan korpus envanteriyle eşleşir. Noktalama hariç payda
   surface 101.052, lemma 104.424'tür; nonpunctuation eksik lemma 0.
   Yüzey MWT bir kez sayılır; lemma bileşenleri ayrı katmandadır. Cümle,
   noktalama ve eksik lemma sınırları korunur. Türkçe I/İ + NFC kullanılır.
3. Dokuz ilişki/18 yön, 14 benzersiz kök. Canonical/query/P4 literal predicate
   surface sayımları gerçekten 0; bütün canonical lemma sayımları pozitiftir.
   Root çiftleri: ön124/arka53, sağ25/sol43, üst95/alt81, önce201/sonra390,
   uzun84/kısa34, hızlı10/yavaş34, ağır32/hafif11. Üst/alt ve önce/sonra birden
   fazla relation hücresinde paylaşılır. Zero log ve inverse-zero oran null;
   smoothing yoktur. Canonical lemma P4 alternatif kök frekansı diye sunulmaz.
4. Katalog gerçek renderer'dan üretilir; TR/EN ifadeler, roots, P4/query
   varyantları, ek parçaları, Türkçe karakterler ve ses kuralları kaydedilir.
   Genel morfolojik çözümleme veya tamamlanmış insan onayı iddiası yoktur.
5. Tokenizer ölçümü kayıtlı model kimliği/revision'a bağlıdır. OpenAI donmuş
   BPE düz metindir; API chat framing sayısı null. Public Qwen tokenizer-only
   dosyaları Git blob/LFS ve SHA-256/template hashleriyle yerelde doğrulanır.
   HF gerçek chat template sayar; vLLM satırı local-template/not-serving-attested
   olarak etiketlenir. Llama iki erişim HTTP401; başka tokenizerla değiştirilmez.
6. Altı model yapılandırması için 216 sözlük ve 1.458 prompt ölçüm satırı vardır.
   Dört yapılandırmada 972 plaintext prompt ölçümü, üç Qwen yapılandırmasında
   729 local-chat ölçümü yapılabildi. Kaynak 243 n=4 pilot önizlemedir; full
   bankanın her modelde yeni chat sayımı iddia edilmez. Aynı query/design/model
   ve isim kontrolü dil oranlarında korunur; transliterasyon ayrı etiketlidir.
   Asset provenance manifestte tekilleştirilir, satırda SHA referansı tutulur.
7. Soru lexical yönü, latent query yönü ve öncül converse sunumu ayrı eksenlerdir.
   Gold denge/coverage ile valid EVET/HAYIR payları, .90/.10 betimsel tanı,
   canonical/P2 ve canonical/equivalent R/D/U/accuracy/invariance ayrımı hazırdır.
   Sadece soru ilişki sözcüğünü değiştirip gold değişen sorgular CBE çifti yapılmaz.
8. Frequency–accuracy/R, token–invariance ve kontrollü OLS/Spearman hazırdır.
   Regresyon model/n/family/gerçek prompt tokenı/yönü kontrol eder; model×vocab
   ayrı fixed-effects tasarımıdır. 18 tasarım hücresi ve 14 lemma bağımsızlık
   sayımı ayrıdır. Missing/rank/few-vocabulary uyarıları, conditional valid
   destek ve dışlanan invalid sayısı açık kalır. IID p/SE/CI üretilmez.
9. Tokenizer-only ortam 20 paket/674 resmî PyPI artifact hash'iyle kilitlendi;
   40 CP312/313 wheel doğrulandı. Model ağırlığı veya Torch kurulmadı. Kaynak
   edinimi ağdan yapıldı; son yerel preprocessing socket/urllib guard altında
   **0 ağ denemesi, 0 LLM çağrısı** ile tamamlandı.

### Test ve doğrulama sonuçları

- Python 3.12.14: **706 test geçti, 0 atlandı**, 98.482 saniye.
- Python 3.13.1: **706 test geçti, 0 atlandı**, 101.305 saniye.
- Önceki 597 test korundu; 109 yeni test: catalog 8, corpus 19, tokenizer 20,
  statistics 20, bias 16, lexical pipeline 13, CLI 13.
- Token ölçümünün iki Python payload'u byte düzeyinde eşit:
  `8524f9a26eb8dede3af106a77c572eb3e82ffb1034872852b96c0a3106842167`.
  Bu test payload'u fullprovenance içerir; teslim tabloları aynı ölçümleri
  manifest referansı ile daha küçük saklar.
- Gerçek offline CLI/corpus/tokenizer kontrolü; bütün çıktı SHA'ları, corpus
  sınırları, bozuk asset/upstream/template pinleri, repeat-frequency rankları,
  eksik/invalid paydalar ve protected input/çıktı immutability doğrulandı.
- Model sonucu yoktur. Preprocessing manifesti
  `preprocessing_only_no_model_results`, contains_model_results=false verir;
  beş performans analizi unavailable'dır. Bias/regresyon katsayısı, performans
  tablosu ve araştırma grafiği oluşturulmadı. Test modelleri geçici dizinlerdeydi.
- Final v0.7.0 wheel ağsız kuruldu: 57 kaynak proje ile, önceki 50 kaynak
  v0.6.0 ile byte düzeyinde aynı. Kurulu lexical CLI 18/216/1458 satır ve
  sıfır model sonucu verdi; core analiz no_model_results. Ayrı korunmuş eski
  ortamda pilot/full saved-plan dry-run 192/43.200 planlanan, sıfır gerçek
  çağrı verdi. Network guard 0 deneme. Wheel SHA256:
  `b4b833c998a0f72ef015b9404847de8987fc88763331a4e4749a044d63a96367`.
- Final teslim 275 dosyalık açık envanterdir; bütün ZIP girdileri yayınlama
  öncesi SHA-256, CRC ve byte uzunluğuyla doğrulanır. Arşiv hash
  yan dosyası self-reference olmaması için ZIP dışında tutulur.

### Sınırlar ve sonraki aşama

Prompt 7 altyapısı ve erişilebilir kaynakların ölçümleri hazırdır. İnsan
linguistic review, gerçek pilot/model seçimi ve Llama tokenizer erişimi bekler.
Korpus dar ve çok anlamlı kökleri sense-disambiguate etmez; gerçek eğitim
verisi veya genel Türkçe sıklığı sayılmaz. Formal surface sıfırları aşinalık
yokluğu değildir. Token kapsamı/plaintext/local-chat ayrımı ve vLLM serving
sınırı korunur. Lexical sonuçlar korelasyoneldir; nedensel frekans etkisi veya
model iç ağırlıklarının doğrudan ölçüldüğü iddia edilmez. Prompt 8, model
inference ve fine-tuning kendiliğinden başlatılmadı.


## Prompt 8 — reasoning prompting ve native effort deney altyapısı — 2026-10-25

**Kapsam tamamlandı: altyapı, dry-run ve testler. Gerçek model değerlendirmesi
çalıştırılmadı.** Kullanıcının “Önce dry-run ve altyapıyı tamamla; pilot modelini
sonra seçelim” tercihi devam ediyor. Prompt 6/7'nin çalışan metrik/lexical
uygulamaları korundu; model seçimi, API/model indirmesi, pilot/full execution,
eğitim/fine-tuning veya bir sonraki promptun uygulamasına geçilmedi.
Dağıtım/reasoning sürümü 0.8.0; diğer bileşen sürümleri aynen kalır.

### Tasarım kararları ve yapılan işlemler

1. Yeni uygulama kodundan **önce** EXPERIMENT_PROTOCOL.md §18 yazıldı. CBE
   çekirdeği ile yeni prompting kolları ayrıldı. Aynı latent/soru/gold/yön/
   mesafe/dil/sunum koordinatları bütün etkin kollarda korunur; hiçbir armda
   bağlam veya destek yüzünden sessiz örnek düşürülmez.
2. A/direct önceki mesajlarla byte düzeyinde aynı kaldı. B/zero-shot reasoning
   açıklama sonrası tek `FINAL:` satırı ister. C normalize et, yönü belirle,
   geçişleri değerlendir ve sonucu üret adımlarını iki dilde belirtir. D aynı
   direct mesajıyla yalnız native effort değiştirir. B/C, provider'ın iç
   reasoning metnine erişim veya bunun güvenilir açıklaması olarak yorumlanmaz.
3. Strict parser B/C'de yalnız tek son `FINAL: EVET|HAYIR` / `FINAL: YES|NO`
   satırını kullanır. Açıklamadan keyfî cevap çıkarmaz; duplicate marker,
   wrong-language, ek metin, refusal ve truncation invalid kalır. Truncation
   ve refusal hem provider flag'i hem terminal finish_reason ile tanılanır.
   Son cevap doğruluğu esastır; açıklamanın içeriği puanlanmaz.
4. `configs/models.yaml` değişmedi. Yeni TOML ayarı ortak 2048 output cap,
   etkin stratejiler ve exact model/sürüm capability beyanını içerir. Base ve
   effective spec ayrı korunur/hashlenir. Temperature/top_p/seed otomatik
   değiştirilmez; native effort sampling kısıtıyla çelişirse tüm plan durur.
   Provider default `none` kabul edilmez. Mevcut altı kapalı örnekte capability
   doğrulanmadığından none/low/medium/high toplam 24 D hücresi unavailable,
   her biri sıfır çağrıdır. HF/vLLM boolean thinking ile effort seviyeleri
   birbirine eşitlenmez. Üç sağlayıcıda A/B/C mevcut adaptörleri kullanır;
   D taşıması desteklenen OpenAI adaptöründe capability koşuluyla hazırlanır.
5. OpenAI Docs becerisi ve güncel resmi reasoning/deployment belgeleri
   kullanıldı. Exact model/effort desteği genellenmedi, başka model seçilmedi.
   Sentetik HTTP taşıma testi none/low/medium/high payload'larını mevcut
   OpenAI adaptöründe aynen doğruladı; bu test gerçek model desteği veya erişim
   kanıtı değildir. Ağ/model çağrısı yoktu.
6. İlk entegrasyon testi, `experiments/` altına yeni Python dosyası eklemenin
   eski builder'ın özyinelemeli kaynak bağını bozduğunu yakaladı. Hash koşulunu
   gevşetmek veya eski veri bankasını yeniden üretmek yerine yalnız yeni
   uygulama **src/reasoning/** paketine taşındı. İstenen
   src/experiments/reasoning/README.md bu gerekli yerleşim farkını açıklar.
   Önceki **57 Python kaynağı byte düzeyinde aynı** kaldı; frozen 24 kaynak,
   v0.4.1 bankası, manifestler, lexical varlık/çıktılar ve eski P5 planları
   değişmedi. Runtime monkeypatch veya kaynak denetiminde istisna yoktur.
7. Reasoning journal ayrı results/reasoning namespace'inde, eski immutable
   RunStore/rate/backoff/cache/provider primitifiyle çalışır. Unique task/attempt,
   raw SHA ve durable outbox doğrulanır; resume bitmiş çağrıyı yinelemez,
   belirsiz timeout/crash açık resolution olmadan tekrar gönderilmez. Namespace
   symlink ile eski sonuçlara yönlendirilemez. Full için tamamlanmış gerçek
   uncached pilot, veriyle bağlı insan dilsel onayı ve açık onay makbuzu gerekir.
   Fixture veya sadece bir dry-run bu kapıyı açamaz; makbuz üretilmedi.
8. Analiz her armı önce ayrı CBE'ye verir; n, query türü/mesafesi, dil, isim,
   relation/family ve model/sürüm korunur. Invalid/missing marjinal accuracy'de
   yanlış kalır; R/D/U/kappa yalnız ortak valid presentation çiftlerini
   kullanır. Strateji−direct farkları ve **aynı dört geçerli cevap** desteğinde
   duyarlılık analizi birlikte raporlanır. Böylece coverage değişmesiyle
   görünen artış saklanmaz. TR−EN ve dil×strateji, token coverage/usage ile
   truncation/invalid tanıları eklendi. B=2000, seed=20261025, %95 bootstrap,
   uygun McNemar, Holm ve önceki R/Wilson yolunu kullanır. Native effort ile
   açık CoT aynı müdahale diye havuzlanmaz. CSV/LaTeX ve arm etiketli altı
   şekil ailesinin PDF/SVG/PNG dışa aktarımı hazırdır.
9. Tam testin ilk turunda eski lexical CLI testinin “lexical dışındaki bütün
   kaynaklar 50 adettir” sayım varsayımı yakalandı. Yalnız bu test güncellendi:
   asıl 50 pre-lexical kaynak şartı korunur, yeni 8 reasoning dosyası ayrıca
   hash snapshot'ına alınır. Üretim kodu veya veri doğrulama kuralı değiştirilmedi.
10. README, ayrıntılı REASONING_EXPERIMENTS.md, yeniden üretim rehberi,
    protokol ve bu günlük güncellendi. Yeni bağımlılık veya run_*.py scripti
    eklenmedi; yeni CLI `turkish-cbe-reasoning` / `python -m reasoning`.
    Eski run_*.py fingerprint'i ve kaydedilmiş P5 planları aynen çalışır.

### Üretilen dry-run kapsamı ve sonuç yokluğu

Bunlar **altı kapalı örnek konfigürasyonun kapsam hesaplarıdır**, gerçek model
seçimi veya çalıştırılmış deney değildir. Her planda A/B/C için 18 etkin arm,
24 unavailable D armı ve bütün etkin kollarda ortak 2048 output cap vardır.

| Plan | Latent sıra | Kaynak sorgu | Arm başına prompt | Planlanan üretim çağrısı | Gerçek çağrı |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot, n=[4,8] | 8 | 32 | 128 | 2.304 | 0 |
| Full, dokuz n seviyesi | 1.800 | 14.200 | 56.800 | 1.022.400 | 0 |

- Pilot her query-answer hücresinde bir latent sıra; full hücre başına 50,
  her n için 200 kaynak sıradır. Endpoint/internal/distance strata ayrıdır.
- Offline input proxy toplamı pilot 1.117.350, full 6.854.692.842 token;
  output üst sınırları 4.718.592 / 2.093.875.200. Bunlar gözlenmiş tüketim
  değildir. Chat framing ve model tokenizer'ı için exact sayım değildir.
- Bilinen fiyatlı OpenAI altkümesi pilot $8.3298825, full $6345.9140175
  tahminidir (maksimum output ve proxy input ile). Diğer fiyatlar, donanım ve
  yardımcı maliyetler bilinmediği için **genel maliyet null** kalır. Maliyet
  varsayımı/as-of ve kaynak her armda saklanır; sıfır maliyet sayılmaz.
- Generation/attempt/auxiliary HTTP/preflight sayıları ayrıdır. HTTP adaptöründe
  runner ve generate iki kez count metodu kullanır, exact-last-payload cache
  ikinci sayım HTTP'sini tekrar göndermez. Retry/resume sınırları açık
  etiketlidir. Disabled/call-limit engelleri ve full pilot-onay gereği saklanır.
- Pilot plan SHA: `b32a13bde51fcaf06a2af8f5792c78f70d48630dc1192fbd52d2c74c9da5722e`.
  Full plan SHA: `2ea547726c23e2792c9cb8ca818a2de4f0454b8b97dd5f6400b6fc13968d39ec`.
  Plan dosya/cohort/source hashleri ve hücre dengeleri dry_run_summary.json'dadır.
- No-model analysis manifesti: sıfır planlı/gerçek inference gözlemi,
  `contains_model_results=false`, `status=no_model_results`, boş table_rows/files.
  Teslimde sahte accuracy, p-değeri, performans grafiği veya açıklama yoktur.

### Doğrulama ve test kanıtları

- **Python 3.12.14: 799 test PASS, 105.506 s, sıfır skip.**
- **Python 3.13.1: 799 test PASS, 110.554 s, sıfır skip.**
- Önceki 706 teste 93 yeni reasoning testi eklendi. 243 frozen sunum
  koordinatında problem frame'i ve dört stratejinin gold/latent kullanmaması,
  parser sınırları, capability/sampling/tamper kontrolleri, görev eşleştirme,
  bütün planlı paydalar, bootstrap/duyarlılık/etkileşim, truncation ve
  crash/resume/cache/approval korumaları doğrulandı. Sentetik figürler test
  geçici dizinlerinde açık sentetik etiketiyle PDF/SVG/PNG olarak üretildi.
- Bağımsız analiz fixture smoke'u 128 planlı observation, dört strateji,
  endpoint/internal ve ayrı n/distance hücrelerinde 32 profil, 192 strateji
  metrik kontrastı, 72 ortak-dört-valid duyarlılık satırı, 24 dil×strateji,
  64 usage/64 truncation satırıyla tüm paydaları doğruladı. Bunlar **unit test
  kanıtıdır**, araştırma bulgusu değildir ve deliverable sonuçlarına konmadı.
- Ağsız wheel 0.8.0 üretimi/kurulumu 65 kaynak SHA'sı, önceki 57 kaynak,
  README UTF-8, CRC ve dokuz entry point ile doğrulandı. Wheel SHA:
  `64e7489b9d5657d210042030b5b774ed711896a30eba6619552f58217a648429`.
  Kurulu yeni reasoning pilotu 144 planlı/0 gerçek çağrı ve aynı saved planın
  tekrar yüklenmesiyle geçti. Eski P5 192/43.200 planları ve boş CBE CLI de
  ağsız geçti. Ağ denemesi/model çağrısı/ağırlık indirmesi **0**.
- Paket envanteri 298 dosya: önceki ZIP'ten 269 değişmemiş girdi, 23 yeni ve
  altı güncellenen dosya. Bütün girdi SHA/CRC/boyut denetimi atomik teslim
  yayımlamasının koşuludur; ZIP SHA256 ayrı sidecar'dadır (kendi içeriğine
  hash yazan öz referans oluşturulmaz). Büyük donmuş JSONL'ler önceki doğrulanmış
  ZIP'ten aktarılır; yeniden okunma/yeniden üretim gerekmez.
- Ayrıntılı çalışma kanıtları dış work/ klasöründeki stage8-final-tests312/313,
  stage8-wheel-smoke, stage8-analysis-verification, stage8-native-adapter-review,
  stage8-package-verification ve stage8-archive kayıtlarında tutulur. Bunlar
  kullanıcıya teslim edilen gerçek model sonuçları değildir.

### Dosya envanteri

Yeni dosyalar (23):

- `REASONING_EXPERIMENTS.md`
- `configs/reasoning.toml`
- `figures/reasoning/README.md`
- `results/reasoning/README.md`
- `results/reasoning/dry_run_summary.json`
- `results/reasoning/no_model_results/analysis_manifest.json`
- `results/reasoning/plans/full.json`
- `results/reasoning/plans/pilot.json`
- `src/experiments/reasoning/README.md`
- `src/reasoning/__init__.py`
- `src/reasoning/__main__.py`
- `src/reasoning/analysis.py`
- `src/reasoning/io.py`
- `src/reasoning/parser.py`
- `src/reasoning/planning.py`
- `src/reasoning/prompts.py`
- `src/reasoning/runner.py`
- `tests/test_reasoning_analysis.py`
- `tests/test_reasoning_cli.py`
- `tests/test_reasoning_parser.py`
- `tests/test_reasoning_planning.py`
- `tests/test_reasoning_prompts.py`
- `tests/test_reasoning_runner.py`

Güncellenen dosyalar (6):

- `EXPERIMENT_PROTOCOL.md`
- `PROGRESS.md`
- `README.md`
- `REPRODUCIBILITY.md`
- `pyproject.toml`
- `tests/test_lexical_cli.py`

Dış outputs/turkish-cbe.zip ve checksum sidecar aynı teslim için güncellenir;
bu iki arşiv dosyası proje içi 298 dosya envanterinin parçası değildir.
Çalışma scriptleri/teşhis logları work/ içinde kalır; araştırma koduna veya
sonuçlarına kullanılmayan dosya eklenmedi.

### Eksikler, metodolojik riskler ve sonraki aşamaya hazırlık

- **Altyapı ve dry-run hazırdır.** Gerçek pilot model/erişim seçimi ertelendi;
  gerçek model doğruluğu, reasoning kazanımı, truncation/invalid oranı ve
  token tüketimi henüz ölçülmedi. Model GPU/serving ortamı denenmedi.
- D için gerçekten desteklenen, sürümü sabitlenmiş model/effort/sampling
  capability kontrolü gerekiyor. Mevcut unavailable hücreleri tamamlanmış
  gerçek D deneyleri olarak saymak doğru olmaz.
- 2048 ortak cap bir yeterlilik garantisi değildir. Pilotla sonradan ayarlanırsa
  bütün karşılaştırılabilir kollarda yeni plan/run gerekir. Aynı üst sınır,
  eşit üretilen token veya eşit dahili hesaplama demek değildir.
- Türkçe insan dilsel incelemesi pending; confirmatory eligibility false
  kalır. Otomatik testler insan değerlendirmesinin yerine geçmez. Model/API
  sürümleri ve yerel weights erişimi gerçek yürütmeden önce yeniden doğrulanır.
- Koşullu R/invariance artışı düşük coverage veya Shared Wrong ile gelebilir;
  accuracy, strict R, coverage, ortak-dört-valid duyarlılığı ve dil/query
  tabakaları birlikte okunur. Sabit çalışma sırası ve provider drift'i gibi
  koşullar gerçek pilot/run kayıtlarıyla değerlendirilmelidir.
- Açık CoT ve native effort ayrı müdahalelerdir. Açıklama metni modelin iç
  sürecinin kanıtı veya puanlama hedefi değildir. Benchmark başarısı bütün
  problem ailesinde yetkinliğin ispatı olarak yorumlanamaz.
- Önceki çalışan kod/mimari gereksiz değiştirilmedi. Sonraki prompta
  kendiliğinden geçilmedi; bu aşama yalnız Prompt 8 kapsamındadır.


## 2026-10-25 — Prompt 9: isteğe bağlı eğitim altyapısı tamamlandı, eğitim çalıştırılmadı

### Kapsam ve tasarım kararları

- Uygulama kodundan önce protokol §19'da model/donanım, yayımlanmış tarif,
  JS guard, GRPO agreement ve normalization, bütçe eşitliği, held-out sözlük,
  reproducibility ve sonuç yokluğu belirsizlikleri yazıldı. Orijinal CBE
  B.3/Ek G referansı ile yeni Türkçe tasarım kararları ayrı belgelendi.
- Yerel donanım gerçekten incelendi: Apple M1, 7 Metal çekirdeği, 16 GiB
  birleşik RAM, NVIDIA/CUDA yok. PyTorch/PEFT/Accelerate ve ağırlıklar yok;
  tokenizer için Transformers 4.57.1 bulunuyor. Bellek CUDA VRAM sayılmadı.
  BF16 ağırlık alt sınırı 3.08 GB, toplam eğitim belleği/throughput/GPU saatleri
  ve maliyet null; tahmin uydurulmadı. Gerçek eğitim başlatılmadı.
- Altı kol: base, SFT-LoRA, paired SFT, CE+JS, GRPO correctness ve GRPO
  correctness+agreement. Model koddan değil mevcut kapalı registry'den alınır.
  Qwen2.5-1.5B immutable snapshot bir çokdilli adaydır; daha büyük aile
  üyelerinin Türkçe ölçümü bu 1.5B snapshot için yeterlilik kanıtı sayılmadı.
- LoRA r16/alpha32/dropout .05 attention+MLP; SFT lr2e-4/epoch1/batch16/
  BF16/1536 ve GRPO lr1e-5/beta0/G4/600 adım başlangıç referansıdır. JS .1,
  detached iki-sunum gold>.5 guard'ı, loss ortalaması ve tek on-policy update
  yerel seçimlerdir; yayımlanmamış makale ayrıntıları tahmin edilmedi.
- Agreement yalnız diğer sunumlarla, valid instance YES oranının chance
  referansı üzerinde pozitif katkıdır: correctness + .5×bonus. Invalid
  reward0; sürekli aynı yanlış cevapta bonus0. Correctness öncelikli ayrı
  izlenir. Prompt-normalized binary advantage'da bonusun etkisizliği/epsilon
  ölçeklemesi tanısı ve instance-centered ayrı kontrol modu vardır. Pozitif
  clipped alternatif, paper'ın signed grpo-ig deneyiyle aynı sayılmadı.
- Veri yalnız frozen training split, Türkçe ve symbolic_shared, n≤128;
  comparison height/speed/weight training ve spatial front/right/above holdout.
  Endpoint/internal/distance kontrol metadata'sı, gold ve dört sunum korunur.
  İleri havuz 300 latent/1900 sorgu/5700 unit; canonical5700/paired11400/
  dört-sunum22800. GRPO n16..128 uygun4500'den600 seçer; dört query×gold
  hücresinde150'şer unit. Pilot8latent/32sorgu/96unit/384sunum,8GRPOunit.
- Training/test latent ve görünür prompt hash kesişimleri sıfır. Dört test
  cohort'u in-domain, heldout-vocabulary, larger-n ve ikisi birlikte şeklinde
  ayrıdır. Yüklem/soru sözlükleri temiz olsa da EN context higher/lower
  örtüşmesi raporlu: full-context holdout false, human review pending.
- Pilot iki optimizer adımı; referans esinli SFT357/paired713/GRPO600; ayrı
  kontrol tüm eğitim kollarında600 adım planlar. Referans yönetim cap1000,
  paired bir epoch'u713'te bitirebilmek içindir; makale ayarı değildir.
  Sequence exposure, token muhasebesi, unique units, parametre×adım bütçesi,
  zero-gradient adımları, CUDA event ve loop wall zamanı ayrı alanlardır.
  Eşit adım/equal exposure, eşit compute/token/GPU süresi iddiası değildir.
- Çalıştırma kapıları: explicit enable/execute, exact library pinleri,
  sabit model/tokenizer commit ve template, yerel ağırlık manifesti hashleri,
  native CUDA BF16 ve boş VRAM. CPU/MPS/model fallback/indirme yoktur.
  Exact token preflight, assistant-only multi-token target masking,
  paired whole-batch, deterministic seed-before-LoRA, GRPO dropout kapalı,
  on-policy old likelihood ve beta0 policy loss uygulandı. Limit aşımında
  truncation/drop yok; çalıştırma reddedilir. GPU kodu henüz çalıştırılmadı.
- Ham step/rollout/plan/execution/adapter kayıtları immutable yeni dizindir.
  Training resume yok; yarım run aborted kalır. Complete adapter/plan/evidence
  hashleri değerlendirmeden önce denetlenir. Evaluate dry-run varsayılan,
  2000 call cap bütün cohort'u sessiz küçültmez. Context-overflow invalid
  tutulur; eksik evaluation tamamlanmış gibi puanlanmaz. R/D/U/kappa/coverage
  ve transfer tabloları mevcut CBE analizinden gelir. Son incelemede Holm
  ailesinin kollar birlikte işlenmesi düzeltildi ve regresyon testi eklendi.
- Kod/raporlama ayrı modüllerdir. Gerçek training/model verisi yokken rapor
  yalnız no_training_or_model_results durum JSON'udur. Araştırma bulgusu,
  başarı grafiği veya tamamlanmış adapter oluşturulmadı. Şimdiki kolların
  tamamı planlı/çalıştırılmamış durumdadır; gerçek optimizer adımı **0**.

Son bağımsız kod incelemesinde iki sorun giderildi: rollout temperature/top-p
ile örneklerken old/current logp ham dağılım kullanıyordu. Artık ikisi aynı
softmax(logits/temperature), top_p1/top_k0 tam destekli policy'dir. Fresh
GenerationConfig ve use_model_defaults=false ile snapshot warper/penalty
mirası engellenir. JS sıcaklık1'de ayrı kalır. Son adım wall bütçesini aşınca
completed yazılmasını önlemek için adım sonrası ve rollout arası kontrol
eklendi. Altı ek saf/GenerationConfig testi ve bir opsiyonel Torch policy
logp-gradient testi yazıldı; ikincisi backend yokluğunda çalışmadı. Düzeltmeler
32 hedefli test ve ikinci salt-okuma incelemesiyle teyit edildi. Source hash'i
değiştiğinden üç plan/no-results kaydı, wheel ve tam testler yeniden üretildi;
önceki plan durum kayıtları dış work/ içinde korunur, ham model sonucu yoktur.

### Doğrulama ve teslim kanıtı

- Önceki 799 teste **117 yeni test** eklendi. Son tam takımlar:
  Python3.12.14 **916/916**, 127.472s; Python3.13.1 **916/916**, 133.272s.
  Kayıtlı hata/skip sıfır. Veri ayrımı/replay/hash, reward/advantage,
  masking/sequence schedule, donanım/weights kapıları, CLI/report ve sentetik
  CBE/transfer+Holm yolları sınandı. Sentetik çıktılar geçici dizinlerdedir,
  araştırma sonucu olarak teslim edilmedi.
- **Yedi opsiyonel Torch sayısal/gradient testi çalışmadı:** Torch yokken
  kayıtlı testlere eklenmez. Sıfır skip, bu yedi testin çalıştığını göstermez.
  Gerçek autograd/CUDA/LoRA/model generation doğrulanmadı. Ağ izni sonrası
  CPU Torch kurulumu shell ve Node PyPI DNS/bağlantı hatası aldı. Başarısız
  kurulum sürümün mevcut olmadığı anlamına gelmez. İndirilen model/ağırlık0.
- Doğrudan aday pinleri Torch2.9.0/Transformers4.57.1/PEFT0.17.1/
  Accelerate1.11.0 resmî metadata, PEFT signature'ları upstream source ile
  kontrol edildi. Tam transitive/CUDA training hash kilidi ve çalıştırılmış
  stack compatibility henüz yoktur. Planlar bütün kurulu dağıtım sürümlerini
  ayrıca saklar; bu metadata eksik bağımlılığı doğrulanmış hale getirmez.
- Üç final planın config/model/dataset/source75 fingerprint'i ve engelleri
  doğrulandı. Pilot JSONL96grup export hashleriyle denetlendi; test verisi
  eğitime girmez. Summary file/plan hashlerini ve herkol bütçesini taşır.
  No-results raporunda gerçek metrik/CSV/LaTeX/p-değeri/grafik bulunmaz.
- Dağıtım0.9.0 wheel ağsız oluşturulup kuruldu;75kaynak,65eski byte-eşit
  kaynak, UTF-8 README/CRC ve10entrypoint doğrulandı. Altı kolun dry-run'ı,
  base evaluation dry-run, pilot plan/export/emptyreport geçti. Eski P5
  192/43200 ve P8 2304/1022400 çağrılık planlar ile yeni144çağrılık
  reasoning planı/reload ve boş CBE analizi aynı kurulumda geçti.
  Ağsız smoke'ta ağ denemesi/model çağrısı/ağırlık indirmesi0.
- Final wheel SHA256: `d6cd56893933367c3ba7b83229bdc976c1c0cf3f5baada9fa630751d398ecad0`; boyut 253272 byte.
- Önceki65 `.py`, frozen data/manifests/SQLite, registry, lexical assets ve
  eski inference/reasoning logları korunur. Eski lexical testte yalnız
  kaynak koruma envanteri yeni training modüllerini kapsayacak biçimde
  genişletildi; production hash doğrulaması gevşetilmedi.
- Teslim envanteri328dosya:30yeni,6güncellenen,292değişmemiş eski ZIP girdisi.
  Final yayımlama her girdi SHA/CRC/boyutunu denetler ve atomiktir. Arşiv
  SHA256 dış sidecar'dadır; kendine referans veren hash kaydı yapılmaz.
  Büyük frozen JSONL'ler eski doğrulanmış ZIP'ten aktarılır, üretilmez.
- Ayrıntılı çalışma kanıtları dış work/ içinde stage9-final-tests312/313,
  stage9-wheel-smoke, stage9-data-verification, stage9-package-verification
  ve stage9-archive dosyalarındadır. Bunlar gerçek eğitim sonucu değildir.

### Dosya envanteri

Yeni dosyalar (30):

- `TRAINING_COMPARISON.md`
- `configs/training/matched_steps.toml`
- `configs/training/pilot.toml`
- `configs/training/reference_inspired.toml`
- `results/training/README.md`
- `results/training/backend_validation.json`
- `results/training/data/pilot/manifest.json`
- `results/training/data/pilot/training_units.jsonl`
- `results/training/dry_run_summary.json`
- `results/training/hardware.json`
- `results/training/plans/matched_steps.json`
- `results/training/plans/pilot.json`
- `results/training/plans/reference_inspired.json`
- `results/training/reports/no_results/training_report.json`
- `src/training/__init__.py`
- `src/training/__main__.py`
- `src/training/config.py`
- `src/training/data.py`
- `src/training/engine.py`
- `src/training/evaluation.py`
- `src/training/hardware.py`
- `src/training/objectives.py`
- `src/training/planning.py`
- `src/training/reporting.py`
- `tests/test_training_config.py`
- `tests/test_training_data.py`
- `tests/test_training_engine.py`
- `tests/test_training_hardware.py`
- `tests/test_training_objectives.py`
- `tests/test_training_workflow.py`

Güncellenen dosyalar (6):

- `EXPERIMENT_PROTOCOL.md`
- `PROGRESS.md`
- `README.md`
- `REPRODUCIBILITY.md`
- `pyproject.toml`
- `tests/test_lexical_cli.py`

Dış `outputs/turkish-cbe.zip` ve `.zip.sha256` aynı teslim için güncellenir;
proje içi328dosya envanterine dahil değildir. Scratch script/log/wheel/deneme
paketleri dış work/ altında kalır; kullanılmayan araştırma özelliği eklenmedi.

### Eksikler, riskler ve hazır olma değerlendirmesi

- **Tasarım, konfigürasyon, veri ayrımı ve dry-run altyapısı hazırdır.**
  CUDA eğitim/evaluation backend'i çalıştırılarak doğrulanmadı; gerçek pilot
  için uygun GPU, yerel exact snapshot, Torch gradient testleri ve stack
  doğrulaması gerekir. Bu tamamlanmadan eğitim başarısı söylenemez.
- 16GiB birleşik Mac belleği CUDA değildir; declared16GiBfree eşiği fit kanıtı
  değildir. Activations/optimizer/cache ve süre/maliyet null'dır. Loop wall
  limiti adım sınırında yumuşaktır; model yükleme/preflight dahil değildir.
- 1.5B modelin Türkçe yeterliliği, insan dilsel kontrolü ve tam-context
  vocabulary izolasyonu açık kalır. EN higher/lower örtüşmesi transfer
  yorumunu sınırlar. 1536 token eski referans esinlidir; mevcut uzun Türkçe
  girdiler sınırı aşarsa yeni ortak koşullu plan gerekir, sessiz kesilmez.
- Parametre güncelleme adımı eşitliği gerçek token/GPU bütçe eşitliği
  sağlamaz. Prompt-normalized GRPO bonusu etkisiz kalabilir. Chance-corrected
  raw reward'un artması öğrenme katkısı veya bütün problem ailesinde
  competence ispatı değildir. R/SharedWrong/coverage/accuracy birlikte okunur.
- Aynı orijinal eğitim ayarlarıyla replikasyon tamamlandığı veya altı kolun
  eğitildiği iddia edilmez. Önceki çalışan mimari korunmuştur. Prompt10'a
  kendiliğinden geçilmedi; bu teslim yalnız Prompt9 kapsamındadır.


## 2026-10-25 — Prompt 10: İngilizce raporlama altyapısı ve sonuç bekleyen makale taslağı

### Önceki aşamanın denetimi ve kapsam

- Prompt9'un75kaynak dosyası önceki doğrulanmış ZIP hashleriyle karşılaştırıldı;
  tamamı aynı. Gerçek raw/normalized model kayıtları ve tamamlanmış eğitim
  yoktur. Prompt9'un kalan7Torch testi/CUDA smoke'u uygun backend'e, gerçek
  pilot ise ertelenmiş model/erişim seçimine bağlıdır. Bunlar çalıştırılmış
  gibi gösterilmedi; bu makaleye sahte sonuçla tamamlanmadı. Kullanıcının
  sonuç yoksa yalnız yer tutucu üretme talimatıyla Prompt10'a geçildi.
- EXPERIMENT_PROTOCOL§20 kanıt kabulü, bağımsız örnek birimi, RQ-tablosu,
  eksik/undefined metrikler, görsel ayrımı, kaynak/uzantı, immutable yayım ve
  hazır olma sınırlarını kaydeder. Bu dosya harici önkayıt değildir.
- Yeni paket reports/publication altında ayrı tutuldu. Altımodül:
  evidence, tables, figures, manuscript, CLI ve sürüm. `src/`75dosyaya dokunmak
  veya frozen plan kaynak bağını gevşetmek gerekmedi. Setuptools source roots
  src+reports, namespacesfalse; dağıtım0.10.0 ve11entrypoint. Yeni dependency
  eklenmedi, önceki hash kilitleri değişmedi. Testlerdeki eski source-count
  varsayımlarını da değiştirmek gerekmedi.

### Uygulama ve tasarım kararları

- Evidence reader native inference/reasoning okuyucuları ve completed training/
  adapter/evaluation hash kanıtlarını kullanır. Synthetic fixture, arbitrary
  CSV/tahmin enjeksiyonu, hash mutasyonu, duplicate/overlapping task, symlink,
  kaçan path ve farklı cohort reddedilir. Yarım veya geçersiz run sessizce
  çıkarılmaz. Recorded provenance, provider'ın bağımsız attestation'ı değildir.
- Frozen bankanın dosya/source/binding/coverage SHA'sı ve SQL split×n/query
  dağılımı çapraz denetlendi. Gerçek toplam3000latent,23200sorgu,5637600prompt,
  7819200comparisoncell. Prompt/sunum/karşılaştırma sayıları bağımsız örnek
  sayılmadı. Table1 33split×n satırıdır; humanreviewpending korunur.
- Native CBE/lexical/reasoning/training analizleri yeniden kullanılır; reporter
  yeni estimand/p-değeri düzeltmesi/icad edilmiş istatistik üretmez. Ortakvalid
  RDU ve kappa, invalidyanlış marjinalaccuracy, coverage/strictR, n/d/query/
  family/language/model/arm ayrımı korunur. Bootstrap2000/seed20261025/%95,
  clusterlatent/nstrata ve mevcut McNemar/Holm/Wilson yorumları kaydedilir.
- Yedi kayıpsız CSV ve okunabilir LaTeX/Markdown görünümleri hazırlandı.
  Kompakt metrik satırları stabil CSVrowID ile tam koşul/modelrevision/nativeCI
  kaydına bağlanır. N,CI,test,padjusted alanları bulunur; NA bilinmiyor demektir.
  Wilson etiketi/dependence uyarısı gizlenmez. LaTeX linewidth'e ölçeklenir ve
  uzun metric adlarına satır kırma izni verir; derlenmiş journaltemplate yoktur.
- Table2/3/4/6/7 birer not_run satırı. Table5 236satır: ön işleme inventory1,
  korpus18,tokenizer216 ve model ilişkileri için not_run1. Features gerçek
  preprocessing'dir; doğruluk, invariance veya bias ilişkisi bulunmuş sayılmaz.
  Missing model N/score/CI/p null'dır, hiçbiri0başarı olarak doldurulmaz.
- RQ1→T3,RQ2→T4,RQ3→T6,RQ4→T5,RQ5→T7; bütünRQlar results_pending.
  Sayısal p-değerinden otomatik hipotez onayı yoktur. Figure1architecture ve
  Figure2generationpipeline PDF/SVG/PNG üretildi. Figure3profiles,4language
  difference,5RDU,6vocabularybias yalnız açık eksik-sonuç Markdown'larıdır.
  Synthetic test figürleri yalnız geçici dizinlerde/etiketli kaldı.
- İngilizce Introduction,RelatedWork,Methodology,ExperimentalSetup,Results,
  Discussion,Limitations,Conclusion ve birleşik manuscript/BibTeX oluşturuldu.
  Önceki Türkçe morphology benchmark'ları nedeniyle first-ever iddiası yok.
  CBE,Jiang/Zhai;CLUTRR;MGSM;CheckList;Ismayilzada2025;TurBLiMP2025;
  Başar/Bisazza2026;Segura2016;TurkishUD birincil kaynakları kontrol edildi.
  Önceki makalelerin bulguları Turkish-CBE sonucu olarak ithal edilmedi.
- Reporter iki yeni hedef paper/ ve reports/current/ oluşturur.75girdi,
  82kaynak/pyproject SHA, config/environment ve bütün çıktı hashleri bir
  manifestte tutulur; öz-referans hash yoktur. Rendering sırasında girdi/kod
  değişirse yayın durur. Eski raw/normalized/data/plan/sunumlar ezilmez.
  Yeni rapor yeni hedef ister. Raporlama model indirme/inference/eğitim yapmaz.

### Testler ve çıktı doğrulaması

- Önceki916teste81yeni test eklendi: evidence23,table/figure28,manuscript20,
  CLI/sınır10. **Python3.12.14 997/997 (131.631s)** ve
  **Python3.13.1 997/997 (151.570s)** geçti; kayıtlıhata/skip0.
  Native fixture-rejection, mutation, provenance, eşleşme/payda, kaynak
  koruma, escaping, açıkNA, formatdeterminism, compactCI ve atomic/no-overwrite
  sınırları doğrulandı.7opsiyonelTorch testi bu997'ye dahil değildir; CUDA
  backend'e erişim olmadığından önceki doğrulanmamış durum aynen sürer.
- İki yöntem PNG'si görsel incelendi: metin/oklar okunaklı ve sığar, insan
  incelemesi pending olarak yazılıdır. Gerçek performans eğrisi uydurulmadı.
- Ağsız wheel0.10.0 üretimi/kurulumu81kaynak,75eski kaynak,UTF8README,CRC ve
  11CLI ile geçti. Kurulu reporter75girdiyi tekrar doğruladı ve41paperdosyasını
  kaynak çalışmasının **aynı byte'larıyla** yeniden üretti. BeşRQpending,
  contains_model_resultsfalse,submission_readyfalse korundu.
- Aynı kurulumda eskiP5 192/43200 çağrılık planlar,P8 2304/1022400 planları,
  yeni144çağrılık reasoning plan/reload,boşCBE/reasoning analizi ve altıtraining
  kolu dry-run/baseevaluation/no-resultsreport çalıştı. Ağ denemesi/model
  çağrısı/weightsdownload/optimizerstep **0**. Model/GPU backend test edilmedi.
- Final wheel SHA256 `667377cfd2d9bd090ab7063dd0fc79e501cca4cce4f4c50eaa97e40353a7a51b`; boyut 289010 byte.
- Teslim385dosya:57yeni,5güncellenen,323değişmemiş eskiZIP girdisi. Eski328
  yolun tamamı korunur;75eskiPython kaynağı frozen veri/loglarla aynıdır.
  Final ZIP her girdi SHA/CRC/boyutuyla kontrol edilip atomik yayımlanır;
  checksum sidecar ayrı tutulur. Büyük JSONL'ler önceki doğrulanmış arşivden
  stream edilir, yeniden üretilmez veya yerel dosyalarda değiştirilmez.
- Çalışma kanıtları dış work/ altında stage10-final-tests312/313,
  stage10-wheel-smoke,stage10-report-verification,stage10-package-verification
  ve stage10-archive kayıtlarındadır. Scratch test/kurulum/packing dosyaları
  araştırma sonuçları arasına konmadı.

### Dosya envanteri

Yeni dosyalar (57):

- `paper/conclusion.md`
- `paper/discussion.md`
- `paper/experimental_setup.md`
- `paper/figures/figure_1_architecture.pdf`
- `paper/figures/figure_1_architecture.png`
- `paper/figures/figure_1_architecture.svg`
- `paper/figures/figure_2_generation_pipeline.pdf`
- `paper/figures/figure_2_generation_pipeline.png`
- `paper/figures/figure_2_generation_pipeline.svg`
- `paper/figures/figure_3_accuracy_robustness_profiles.md`
- `paper/figures/figure_4_turkish_english_difference.md`
- `paper/figures/figure_5_rdu_distribution.md`
- `paper/figures/figure_6_relation_vocabulary_bias.md`
- `paper/introduction.md`
- `paper/limitations.md`
- `paper/manuscript.md`
- `paper/methodology.md`
- `paper/references.bib`
- `paper/related_work.md`
- `paper/results.md`
- `paper/tables/table_1_dataset_statistics.csv`
- `paper/tables/table_1_dataset_statistics.md`
- `paper/tables/table_1_dataset_statistics.tex`
- `paper/tables/table_2_overall_model_performance.csv`
- `paper/tables/table_2_overall_model_performance.md`
- `paper/tables/table_2_overall_model_performance.tex`
- `paper/tables/table_3_turkish_english_comparison.csv`
- `paper/tables/table_3_turkish_english_comparison.md`
- `paper/tables/table_3_turkish_english_comparison.tex`
- `paper/tables/table_4_morphosyntactic_robustness.csv`
- `paper/tables/table_4_morphosyntactic_robustness.md`
- `paper/tables/table_4_morphosyntactic_robustness.tex`
- `paper/tables/table_5_lexical_relation_analysis.csv`
- `paper/tables/table_5_lexical_relation_analysis.md`
- `paper/tables/table_5_lexical_relation_analysis.tex`
- `paper/tables/table_6_complexity_inference_distance.csv`
- `paper/tables/table_6_complexity_inference_distance.md`
- `paper/tables/table_6_complexity_inference_distance.tex`
- `paper/tables/table_7_reasoning_training.csv`
- `paper/tables/table_7_reasoning_training.md`
- `paper/tables/table_7_reasoning_training.tex`
- `reports/README.md`
- `reports/current/REPORT_STATUS.md`
- `reports/current/evidence.json`
- `reports/current/report_manifest.json`
- `reports/current/rq_traceability.csv`
- `reports/publication/__init__.py`
- `reports/publication/__main__.py`
- `reports/publication/evidence.py`
- `reports/publication/figures.py`
- `reports/publication/manuscript.py`
- `reports/publication/tables.py`
- `tests/test_publication_cli.py`
- `tests/test_publication_evidence.py`
- `tests/test_publication_figures.py`
- `tests/test_publication_manuscript.py`
- `tests/test_publication_tables.py`

Güncellenen dosyalar (5):

- `EXPERIMENT_PROTOCOL.md`
- `PROGRESS.md`
- `README.md`
- `REPRODUCIBILITY.md`
- `pyproject.toml`

Dış outputs/turkish-cbe.zip ve checksum sidecar aynı teslim için güncellenir;
proje içi385dosya envanterine dahil değildir.

### Eksikler, metodolojik riskler ve hazır olma değerlendirmesi

- **Raporlama modülü ve İngilizce taslak hazırdır; deneysel makale yayına hazır
  değildir.** Gerçek model/pilot sonuçları, uygun CUDA stack doğrulaması ve
  Türkçe insan dilsel incelemesi beklenir. Sonuç toplanınca yeni doğrulanmış
  native run'lar üzerinden yeni çıktı hedefinde rapor üretilebilir.
- Karşılaştırılabilir model/sampling/output/context koşulları ve gerçek
  token/usage bilgisi yokken dil veya prompting etkisi çıkarılamaz.7Torch
  testi/gerçek GPU çalıştırması yerine safPython testleri geçirilmiş sayılmaz.
- P4/P5 çoklu dönüşüm, endpoint d=n−1, dependent sunum/isim kopyaları,
  corpus kapsamı ve farklıchat/plaintext token ölçümleri karıştırıcıdır.
  Englishhigher/lower contextörtüşmesi full lexical holdout'u sınırlar.
  Frekans ilişkileri nedensel veya dahili model ağırlıklarının kanıtı değildir.
- Koşullu invariance/R başarı kanıtı sayılmaz: SharedWrong,accuracy,coverage,
  strictR ve paydalar birlikte gerekir. Undefined kappa/CI/testlernullkalır;
  finite benchmark bütün problem ailesinde competence ispatlamaz.
- Altyapının tamamlanması; hipotezlerin desteklendiği, eğitim kollarının
  çalıştırıldığı, orijinal tarifin tekrarlandığı veya bir derginin yayın
  standardının karşılandığı anlamına gelmez. Önceki mimari korundu.
  Prompt11'e ve yeni deney kapsamına kendiliğinden geçilmedi.


## Prompt 11 — Son bilimsel ve teknik kalite denetimi (2026-10-25)

### Kapsam ve karar

18 istenen kontrol uygulandı; bağımsız salt-okunur veri/dil, yürütme ve
istatistik/yayın incelemeleri önce yapıldı. Araştırma yazılımı dağıtımı0.11.0.
**Ampirik makale yayına hazır değildir.** Somut mevcut Critical
leakage/oracle/metric hatası saptanmadı; bu sınırlı kapsamlı denetim, hiç
hata bulunamayacağının veya bütün ailede competence sağlandığının ispatı
sayılmaz. Dört Major yazılım sorunu giderildi. Gerçek insan dilsel
incelemesi, model verisi ve isteğe bağlı CUDA/Torch doğrulaması açık
önkoşul olarak tutuldu. Model çağrısı0, ağırlık indirmesi0, optimizer adımı0.

### Bulgu, düzeltme ve tasarım kararları

1. M01: GPU/call gate sağlandığında, training evaluation511.200 çağrı/kol
   için review olmadan model yüklemeye ulaşabiliyordu. Pre-fix sentinel,
   yalnız GPU kapısını mock edip yükleme sınırını gözlemledi; hiç model
   yüklenmedi/çağrılmadı. Büyük pilot ismi de review atlama yolu açıyordu.
   Yeni ortak validator toplam planned calls>2.000 için inference,
   reasoning ve training evaluation'da provider/GPU/modelden önce durur;
   full pilot-approval aynı validator'ı kullanır.
2. Review yalnız approved JSON bayrağı değildir: exact dataset SHA,
   iki ayrı yeterliği belirtilmiş Türkçe incelemeci, gerçek prompt metni
   ve grid ID/hash, dört değerlendirme boyutunun tamamında pass ve
   approved resolution gerekir. En az100 farklı TR prompt, seçili her
   relation×presentation×name template hücresinde20 örnek ve her n/query
   türü için kapsama zorunludur. Bu, tam çapraz n×query hücre başına20
   veya bütün veri üzerinde insan incelemesi diye yorumlanmaz. Fixture
   receipt canlı çalıştırmayı açamaz. Gerçek onaylı receipt üretilmedi.
3. M02: Training evaluation fresh neutral greedy GenerationConfig ve
   use_model_defaults=False kullanır. Repo defaults'tan beam/penalty/forced
   token devralınmaz. Sampling, seed ve hash her çağrıda/manifestte tutulur.
   Büyük evaluation receipt+CSV bytes da kökene eklenir.
4. M03: training_arm_contrasts, trained-minus-base eşlenmiş etkilerini
   native rapora ve gelecekteki publication kanıtına ekler. Accuracy,
   strictR ve coverage bütün planlı koordinatlar üzerindedir; invalid
   yanlış sayılır. R/invariance/kappa common-four-valid destek üzerinde
   duyarlılık analizidir. Kolların/ortak desteğin sayıları,2.000 tekrarlı
   latent-cluster bootstrap CI, uygun exact/paired McNemar ve5 declared
   arm üzerinden Holm (eksikler yuva tutar) eklenir. Bu çalışma çapında
   FWER değildir. Eksik/eşleşmeyen/gold-query değiştiren koordinatlar
   reddedilir; eksik kol başarı verisi olamaz.
5. M04: Nonbase native training raporu artık verilen completed training
   evidence ile full adapter identity eşitliğini şart koşar; publication
   collector ile köken sözleşmesi tutarlı hale getirildi.
6. Bağımlı R profillerinde grafik önceliği cluster-bootstrap CI oldu;
   Wilson betimsel etiketlenir. Endpoint derece kısa yolu tüm3.000
   örnekte doğrulandı. n girdi büyüklüğü ve d latent grafik mesafesidir;
   ikisi de modelin hesaplama derinliği ölçüsü olarak sunulmaz.
7. Kaynak/şema katman sürümleri değişmedi; dağıtım0.11.0 ve exact kaynak
   SHA birlikte kullanılır. Frozen bankaya bağlı generator/renderer/
   complete_grid/experiments/core/validation kaynakları ve tüm mevcut
   data/log/result/paper byte'ları korundu. Gerekli yürütme/raporlama
   düzeltmeleri nedeniyle eski source-bound planları güncel kaynakla
   tüketmek reddedilir; yeni plan zorunludur. Kaynak doğrulama gevşetilmedi.
8. Prompt10 paper/reports/current ve önceki planlar tarihsel0.10 kayıtlar
   olarak kaldı. 81 önceki packaged kaynak dosyası, tüm eski Python/test/
   script dosyaları ve pyproject ayrı geçici kopyada migration patch ters
   uygulanarak byte düzeyinde geri elde edildi. Çalışan proje geri alınmadı.
   source_migration.json önce/sonra hashlerini saklar. Kaynak delta22
   code/test/pyproject yolunu içerir; dokümanlar bu ters kaynak patch'ine
   dahil değildir.

### Veri ve bilimsel kontrol sonuçları

- 3.000 source order ve23.200 query seed/rank/gold/direction/distance
  tam tekrar üretildi;18.000 alias order denetlendi. Splitler training450,
  development450,pilot300,test1.800; splitler arası exact source-order,
  alias-order ve görünür prompt hash örtüşmesi0.
-251 query stratumda test dört hücresi50/50/50/50, diğer split hücreleri
  12/13; maksimum hücre farkı1. Endpoint/internal/distance ayrı tutuldu.
-729.000 component normal form oracle ile uyumlu; her query243 design.
  Stratified33 split×n grubu×243=8.019 gerçek component,
  1.543.293 cümle ve24.057 prompt hash tekrar üretildi. Digest
  aae42adbcb6ffa630a70a2a924b59530efa2ad31b2a3a8fbb53532a5ca228c15.
  Bütün5.637.600 promptun bu aşamada yeniden render edildiği iddia edilmez.
-252 mevcut dilsel CSV satırı parse edildi;36 kısa context n4/8.
  Başlangıç AI okuması216 örneği kapsadı; kesin dilbilgisi hatası
  saptanmadı fakat P3/postverbal ve spatialP4 doğal kullanım değerlendirmesi
  insana bırakıldı.12 bağımsız suffix fixture geçti. İnsan onayı hâlâ yok.
-1.458 exhaustive Boolean/invalid/gold case bağımsız CBE oracle ile
  eşleşti;960 exact McNemar tablosu integer-binomial oracle ile eşleşti.
  Bağımlı ordinary McNemar reddi, eksik Holm yuvası ve2.000 tekrarlı
  latent-cluster bootstrap doğrulandı. R+D+U ortak geçerli destek üzerinde1.
-44 tarihsel yayın output hash/size doğrulandı;7 tablo evidence'dan
  yeniden kuruldu. Gerçek performans dosyası yok; RQ1–RQ5 pending ve
  tüm model bulguları NA/null. Yöntem sayımları model sonucu değildir.

### Test, kurulum ve dry-run

- Python3.12.14: **1.027/1.027 geçti**,139.846s.
- Python3.13.1: **1.027/1.027 geçti**,159.984s.
- Atlanan kayıtlı test0; buna karşın7 optional Torch testi Torch olmadığı
  için kaydolmadı/çalışmadı ve geçilmiş sayılmadı. Python3.11 ve CUDA
  execution test edilmedi.
-30 yeni regression test var. İlk tam test turunda yeni endpoint-shortcut
  test fixture'ının eski generator imzasına yanlış keyword vermesi
  düzeltildi; iki final suite bundan sonra tamamıyla geçti. Daha önceki
  997 test korunur; provenance eklenince bir eski synthetic training
  fixture gerçek kanıt sözleşmesine uygun biçimde adapte edildi.
- Ağsız0.11.0 wheel üretildi ve ayrı hedefe kuruldu;87/87 Python dosyası
  current source ile byte eşleşti,12 CLI help kontrolü geçti. Wheel SHA
  7944e38a643ea0a0cd4167f8e8e43f5ae3848b02778efed2cbb7fd4230ab72ae.
- Kurulu CLI ile inference/reasoning pilot/full smoke planları yeni ayrı
  hedefe kaydedilip tekrar yüklendi. Bunlar yalnız n4,4 group smoke
  planlarıdır: inference16,reasoning144 çağrı planı; gerçek/full deney
  çalıştırması veya kapsamlı final full plan olarak yorumlanmaz.
- Üç güncel training bütçe planı kaydedildi; altı kolun pilot run'ı ve
  base evaluation dry-run kaldı. Pilot plan base evaluation10.224
  çağrı planlıyor; çıktısı linguistic_review_required=true, budget ve
  hardware/model blockers açıkça kayıtlıdır. Başlatılmadı.
- Eski inference/reasoning planları kurulu güncel kaynakta beklenen
  source-fingerprint hatasıyla reddedildi. Installed audit tekrarında
  istatistik/checklist kaynak çalıştırma ile birebir aynı;87 source ve
  51 input bağlandı. Ağ girişimi0, model çağrısı0, optimizer adımı0.
  Smoke sonuçları54.786s içinde tamamlandı. İlk harness çalıştırmasında
  audit CLI'nin progress satırlarını JSON diye okumaya çalışması
  düzeltildi; mevcut doğrulanmış source code'da değişiklik gerekmedi.
- Tam test logları, hashli özet, before-fix üç kanıt, güncel audit JSON,
  migration kanıtı ve kurulu wheel kontrolü reports/audit altında saklanır.
  Ara iş dosyaları proje teslim ağacına eklenmedi.

### Açık Major önkoşullar ve metodolojik sınırlar

- İki gerçek Türkçe incelemecinin tamamlanmış ve kapsamı yeterli review'u
  yok. Software/AI inceleme bu açığı kapatmaz.252 örneklik eski boş
  rating'li rapor tüm n/query/template kapsamını onaylamaz.
- Gerçek model/pilot sonucu yok; kullanıcı modeli/erişimi daha sonra
  seçecek. Hipotezler desteklenmiş sayılmaz. Ground truth'dan sahte model
  cevabı ya da sentetik unit verisinden makale bulgusu üretilmedi.
- M1/16GiB uygun CUDA eğitim ortamı değildir; Torch/PEFT/Accelerate ve
 7 Torch testi doğrulanmadı. Tam transitive CUDA kilidi ve ölçülmüş
  GPU/token/parametre budget karşılaştırması bekler. Eğitim tekrarlandı
  veya optional eğitim kolları tamamlandı diye sunulmaz.
- Prompt/tokenizer uzunluğu, cevap dili, vocabulary/context örtüşmesi,
  P4/P5 çoklu dönüşümler, paylaşılan renderer/parser kodu ve isomorphic
  order ailesi confound sınırlarıdır. Bireysel latent ayrımı, model
  pretraining contamination yokluğu veya yapısal aile genellemesi değildir.
- Holm scope model/design/metric ailesidir; CI pointwise'dır. Common-four
  sensitivity tüm planlı accuracy/strictR/coverage ile birlikte okunur.
  Endpoint n/d derinlik iddiası, nedensel frequency etkisi, dahili
  reasoning sadakati ve family-wide competence iddiası yapılmaz.

### Hazır olma ve sonraki kapsam

İstenen yazılım denetimi ve doğrulanabilir düzeltmeler tamamlandı.
Güncel kaynak dry-run/raporlama ve inceleme kapıları hazır. Gerçek model
seçimi, insan review'u ve gerektiğinde CUDA backend olmadan deneysel
makale yayına hazır değildir. Önceki çalışan mimari yalnız gerekli
kapı/decoding/paired comparison/provenance noktalarında değiştirildi.
Bir sonraki prompta veya yetkilendirilmemiş gerçek deneye geçilmedi.
RESEARCH_AUDIT.md tam18 kontrolü Critical/Major/Minor ayrımıyla verir.

### Oluşturulan ve değiştirilen dosyalar

Yeni dosyalar (30):

- `RESEARCH_AUDIT.md`
- `reports/audit/baseline/data_language_before.json`
- `reports/audit/baseline/execution_before.json`
- `reports/audit/baseline/statistics_before.json`
- `reports/audit/current/audit_manifest.json`
- `reports/audit/current/checklist.csv`
- `reports/audit/current/data_checks.json`
- `reports/audit/current/statistics_checks.json`
- `reports/audit/migration_from_0_10.patch`
- `reports/audit/plans/inference_full.json`
- `reports/audit/plans/inference_pilot.json`
- `reports/audit/plans/reasoning_full.json`
- `reports/audit/plans/reasoning_pilot.json`
- `reports/audit/plans/training_matched_steps.json`
- `reports/audit/plans/training_pilot.json`
- `reports/audit/plans/training_reference.json`
- `reports/audit/source_migration.json`
- `reports/audit/test_verification.json`
- `reports/audit/tests_python312.log`
- `reports/audit/tests_python313.log`
- `reports/audit/wheel_verification.json`
- `reports/research_audit/__init__.py`
- `reports/research_audit/__main__.py`
- `reports/research_audit/data_checks.py`
- `reports/research_audit/statistics_checks.py`
- `src/inference/linguistic_review.py`
- `src/training/comparison.py`
- `tests/test_audit_execution_gates.py`
- `tests/test_audit_scientific_checks.py`
- `tests/test_audit_training_comparisons.py`

Güncellenen dosyalar (19):

- `EXPERIMENT_PROTOCOL.md`
- `PROGRESS.md`
- `README.md`
- `REPRODUCIBILITY.md`
- `TRAINING_COMPARISON.md`
- `pyproject.toml`
- `reports/README.md`
- `reports/publication/evidence.py`
- `reports/publication/tables.py`
- `src/analysis/reporting.py`
- `src/inference/cli.py`
- `src/inference/runner.py`
- `src/reasoning/__main__.py`
- `src/reasoning/runner.py`
- `src/training/__main__.py`
- `src/training/evaluation.py`
- `src/training/reporting.py`
- `tests/test_lexical_cli.py`
- `tests/test_training_workflow.py`

Proje teslimi415 dosyadır; önceki385 yolun tamamı korunur. ZIP ve
checksum sidecar bu teslim için atomik olarak güncellenir ve bu415
proje dosyasına dahil değildir. Değişmeyen tarihsel dataset içerikleri
önceki doğrulanmış ZIP'ten akıtılıp her girişin SHA/size/CRC'si yeniden
kontrol edilir; büyük eski JSONL dosyalarını yerel bulut cache'den
tekrar okumaya bağımlı değildir. Güncel frozen SQLite/veri sözleşmesi
ayrı kaynak ve kurulu audit çalıştırmalarında doğrulanmıştır.
