# Latent sıralama verisi — kurallar ve üretim raporu

Üretici `0.2.0` · Protokol `tr_cbe_latent_v0_2` · seed `11`.
Geliştirme verisidir; doğal dil promptu, LLM cevabı veya model performansı içermez.

## Gerçek üretim envanteri

Her `n∈{4,8,16,32,64,128,256,512,1024}` için 200 temel çekiliş:
toplam **1.800 temel sıralama**. Aynı sıralama sorgu kollarında ve d'lerde
kullanılır. 14.200 kayıt bağımsız 14.200 sıralama değildir.

| Kol | Dosya | Kayıt | True | False | Her yön × gold hücresi |
|---|---:|---:|---:|---:|---:|
| Endpoint | 9 | 1.800 | 900 | 900 | 450 |
| Internal | 9 | 1.800 | 900 | 900 | 450 |
| Distance-controlled | 53 | 10.600 | 5.300 | 5.300 | 2.650 |
| Toplam envanter | 71 | 14.200 | 7.100 | 7.100 | 3.550 |

Toplam yalnız envanterdir; kollar başarı istatistiğinde birleştirilmez.
Her `n × kol × (varsa d)` JSONL bloğunda 200 kayıt, 100 true/100 false ve
dört hücrede 50'şer kayıt vardır.

| n | Geçerli d | Bütün kollarda kayıt |
|---:|---|---:|
| 4 | 1 | 600 |
| 8 | 1,2,4 | 1.000 |
| 16 | 1,2,4,8 | 1.200 |
| 32 | 1,2,4,8,16 | 1.400 |
| 64 | 1,2,4,8,16,32 | 1.600 |
| 128 | 1,2,4,8,16,32,64 | 1.800 |
| 256 | 1,2,4,8,16,32,64,128 | 2.000 |
| 512 | 1,2,4,8,16,32,64,128,256 | 2.200 |
| 1024 | 1,2,4,8,16,32,64,128,256,512 | 2.400 |

## Kol, yön ve mesafe

`latent_order` ilkten sona tam etiketlerdir. `premise_edges=[a,b]`, `a≺b`
önermesidir; tam n−1 komşu kenar bağımsız permüte edilmiştir. `before/after`
kanonik ilişki enum'udur; İngilizce prompt değildir.

- Endpoint: pozisyon 0,n−1; d=n−1.
- Internal: 1..n−2 içinden iki farklı pozisyon uniform seçilir. Mesafe bu
  örneklemenin sonucudur; kontrollü mesafe deneyi değildir.
- Distance-controlled: her geçerli d için i, 1..n−d−2 aralığından uniform,
  j=i+d seçilir. İki uç da iç düğümdür; d≤n−3. Geçersiz d'ler atlanıp
  manifestte listelenir. CLI ayrıca 3,7,1021 gibi mesafeleri destekler.

`query_direction=forward`, subject'in latent pozisyonunun object'ten küçük
olmasıdır; reverse tersidir. Her yön true/false ile dengelenir; query_relation
buna göre before/after olur. Yön ve cevap aynı değişken değildir.

## İsim seçimi ve biçimleme

NFC normalizasyonu, sabit sıralı havuzlar ve tekrarsız örnekleme kullanılır.
Tam etiket aynı örnekte benzersizdir; örnekler arasında isimler tekrarlanabilir.
`entities.py` ve `NAME_RULES_VERSION=1` kuralları sabitler.

| entity_type | Havuz | Kural | Temel sıra |
|---|---:|---|---:|
| turkish_names | 1.536 | Elle yazılmış 48 ad × 32 soyad; tam ad kimliktir | 600 |
| synthetic_names | 5.184 | CVCV, ilk harf büyük, aşağıdaki ünlü uyumu | 600 |
| symbolic_ids | 2.048 | E0000..E2047; dört basamak korunur | 600 |

Tam adlar kurgusal birleşimlerdir; gerçek kişilere atıf veya sıklık derlemi
örneklemesi değildir. Aynı ad farklı soyadlarla bulunabilir; tam ad tekrarlanmaz.
Sentetiklerde ilk ğ kullanılmaz. İlk ünlü a/ı ise ikinci a/ı; o/u ise a/u;
e/i ise e/i; ö/ü ise e/ü olur. Bu sınırlı ses yapısı tasarımı dilbilimsel onay
değildir; tesadüfi gerçek sözcük/adlar olabilir.

İsim grubu dört sorgu hücresinden oluşan bloklarla atanır. Her n için
grup sayıları 68/68/64; küçük grup n ile döner. Her grupta dört hücre eşit,
dokuz n toplamında her grup 600 temel sıradır. Mesafe kolundaki kayıtlar:
Türkçe 3.532, sentetik 3.520, sembolik 3.548. Her grubun true/false sayısı
eşittir. Gruplar aynı sıranın üç eşlenik takma adı olarak çoğaltılmadı.
Tam adlar iki sözcük, diğerleri tek etikettir; uzunluk ve bilinirlik değişir.
İsim grubu farkı saf dil etkisi değildir.

## Kayıt şeması ve kimlikler

UTF-8 JSONL, sıralı anahtarlar, boşluksuz ayraçlar ve LF satır sonu kullanılır.
`gold_label` JSON Boolean'dır. Zorunlu alanlar:
`instance_id, seed, n, entity_type, latent_order, premise_edges, query_subject,
query_object, query_relation, query_type, query_distance, gold_label,
generator_version`.

Ek alanlar `ordering_id, sample_index, query_direction`.
ordering_id seed/n/çekiliş indeksi/soyut permütasyonu tanımlar; sorgu kolunu
ve isim etiketlerini içermez. Başka adlandırmayla aynı çekiliş aynı gruptur.
instance_id adlandırma koşulu, sorgu kolu/uçları/yönü ve kontrollü d'yi kapsar.
SHA-256 kullanılır. Bütün zincirlerin graf izomorfisi aynıdır; kimlikler yeni
mantık ailesi kanıtı değildir.

Train/dev/test ayrımı henüz yoktur. Bütün sorgu, d, çeviri ve adlandırma
türevleri ordering_id ile aynı splitte tutulmalıdır. Seed 11 geliştirme verisi
sonradan doğrulayıcı test seti olarak adlandırılmaz.

## Doğrulama

Bütün kayıtlar eşsiz n varlık, n−1 doğru komşu kenar, bağlı/çelişkisiz/döngüsüz
zincir, sorgu/mesafe/yön ve Boolean gold açısından doğrulanır. Pozisyon
oracle'ı ile yalnız öncül grafiğinden tam zinciri yeniden kuran bağımsız
oracle karşılaştırılır. Hücre dengesi, ortak sıraların kollar arasında
eşitliği ve küresel instance benzersizliği denetlenir.

[dataset_manifest.json](../dataset_manifest.json) plan, ortam, üretici kodu ve
isim havuzu hashlerini; geçerli/atlanmış d'leri; dosya sayısı, dağılım, byte
uzunluğu, SHA-256 ve doğrulama sonuçlarını içerir. `verify --replay` ayrıca
veriyi bellekte yeniden üretip byte hashlerini karşılaştırır. Normal verify
semantik açıdan geçerli kayıt permütasyonunu kabul edebilir; replay dosya
sırasını da zorunlu tutar. Doğal dil ve model değerlendirmesi kapsam dışıdır.

Bu latent dosyalar Prompt 3'te korunmuştur. Türkçe yüzey katmanının ayrı
[insan kontrol raporu ve rehberi](validation/README.md) hazırlanmıştır.


## Aşama 4 — matched splitler

`pilot/`, `development/`, `training/`, `test/` yeni, ayrık seedlerle üretilmiş
bilingual deney verisidir. Her biri groups/latent/prompts/cells.jsonl ve
split_manifest.json içerir. Kök experiment_dataset_manifest.json bunları
hash ve tam hücre sayılarıyla dondurur. Şema DATASET_CARD.md, matris
EXPERIMENT_MATRIX.md içindedir. Seed11 latent/ geliştirme kaynağı olarak kalır.
Model çağrısı veya insan dilsel onayı yoktur.


## Aşama 4 ek tamamlama — v0.4.1

`complete_grid/` güncel tam çapraz matristir. Aynı 3.000 sıranın bütün
23.200 sorgusu × 243 dil/isim/ilişki/sunum koordinatı = 5.637.600 ölçülmüş
prompt kaydı vardır. `grid.sqlite` gerçek iki BPE token sayısını ve prompt
hashlerini; `coverage.json` bütün hücreleri; `manifest.json` dosya/kod/plan/
tokenizer hashlerini içerir. Uzun metinler kayıpsız JSONL'ye açılır.

Yukarıdaki seed11 açıklaması yalnız tarihsel `latent/` içindir. Yeni matched
splitler ayrıdır; eski JSONL'ler v0.4.0 olarak korunur. Güncel şema ve sınırlar
[DATASET_CARD.md](../DATASET_CARD.md), tam sayılar
[EXPERIMENT_MATRIX.md](../EXPERIMENT_MATRIX.md) içindedir. Model sonucu veya
insan dilsel onayı yoktur.
