# Turkish-CBE veri kartı — güncel tam matris v0.4.1

2026-10-24 · Sentetik mantıksal problemler ve gerçek yerel tokenizasyon.
Model inference veya eğitim sonucu içermez. Kaynak latent sıra sayısı 3.000,
source sorgu sayısı 23.200'dür. Aynı kaynaklar üzerinde 5.637.600 ölçülmüş
prompt kaydı ve 7.819.200 karşılaştırma vardır; hücreler bağımsız yeni problemler
olarak sayılmaz. Matris [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md) içindedir.

## Sürümler ve dosyalar

- `data/complete_grid/grid.sqlite`: güncel tam faktör/ölçüm bankası.
- `data/complete_grid/coverage.json`: split, n, sorgu/mesafe/yön/gold tabakaları,
  243 tasarım koordinatı ve 351 karşılaştırma kuralı.
- `data/complete_grid/manifest.json`: kaynak, plan, tokenizer, ortam, dosya ve
  mantıksal satır hashleri; yeniden üretim kaydı.
- `data/complete_grid/pilot_preview.jsonl`: tek pilot sorgusunun bütün 243
  tasarımını açan tam-metin örneği; yeni bağımsız gözlem değildir.
- `tokenizers/`: iki resmî vocabulary, kurallar/model eşlemesi ve lisans.
- `requirements-tokenizers.lock`, `vendor/tokenizers/`: sabit bağımlılıklar
  ve mevcut ortam için çevrimdışı kurulum wheel'leri.

Önceki `data/{pilot,development,training,test}/` JSONL dosyaları ve kök
`experiment_dataset_manifest.json` v0.4.0 kaydıdır ve byte düzeyinde korunur.
Onların null model-token alanları tarihsel kayıttır. Güncel gerçek sayılar
v0.4.1 `measurements` tablosu ve JSONL dışa aktarımındadır; eski kayıtlar
sessizce değiştirilmez. Önceki generator 0.2.0, TR renderer 0.3.0 ve core 0.4.0
kaynakları değişmemiştir; tam matris ayrı `complete_grid` paketidir.

## Tam kapsama

Bütün splitlerde her sorgu, üç isim kontrolü × dokuz ilişki × dokuz geçerli
dil/sunum koşuluna açılır. EN canonical/equivalent; TR canonical/equivalent
ve P1..P5 kullanılır. P6 isteğe bağlı ayrı kol olarak kalır. Bütün test n
boyutlarında dokuz ilişki, üç isim kontrolü ve endpoint/internal/distance
bulunur. Eğitim/test ayrımı latent sırada yapılır; bütün türevler aynı splitte
kalır. Seedler ve dört hücre dengesi v0.4.0 ile aynıdır.

## SQLite şeması

| Tablo/görünüm | Birim ve temel alanlar |
|---|---|
| metadata | version, plan, tokenizers, reference_tokenizer, base_binding |
| designs | 243 koordinat: id, name_control, relation_id, language, condition, factor_axes |
| groups | latent sıra: id, latent_instance_id, split, n, sample_index, source_json |
| queries | id, group_id, query_instance_id, query_type/distance, subject/object/relation/direction, gold_label |
| components | (group_id,design_id): prefix SHA/byte/karakter/word/reference/BPE sayıları, normal-form ve plan hashleri |
| measurements | (query_id,design_id): group_id, tam prompt SHA, cl100k_base, o200k_base, UTF8/karakter/word/reference sayıları |
| comparison_rules | 351 arm/sol tasarım/sağ tasarım kuralı |
| comparison_cells | 7.819.200 sol/sağ sorgu ve tasarım başvurusu; A–E aynı sorgu, F ayrı sorgular |

`groups.source_json` endpoint generator kaydı, bijektif isim haritaları ve
bütün presentation planlarını saklar. Sorgu dışındaki uzun metin tekrar
saklanmaz; metin kaynaklardan aynı byte dizisiyle oluşturulur. Her ölçümün
hem kendi sorgu grubu hem de bileşen grubuna bileşik yabancı anahtarı vardır.
Yanlış latent sıraların geçerli görünen kimliklerle eşleşmesi reddedilir.

Her sorguda tam 243 ve her grupta tam 243 bileşen kontrol edilir. Eksik/ekstra
koordinat, değiştirilmiş SQL karşılaştırma görünümü, bozuk gold veya token
sayısı sadece dış hash yenilenerek geçerli hale getirilemez; derin replay
gerçek kaynak, gramer, iki oracle ve BPE sayımlarını tekrar hesaplar.

## Güncel JSONL dışa aktarımı — şema 2

Her satır tam görünür `prompt`, `prompt_sha256`, `prompt_id`, `dataset_version`,
`grid_query_id`, `grid_design_id`, `latent_instance_id`, `query_instance_id`,
`split`, `n`, sorgu türü/mesafesi/hedefi/ilişkisi/yönü ve boolean `gold_label`
içerir. `name_control`, `relation_id`, `language`, `condition`,
`presentation_plan_sha256`, `factor_axes`, `is_multi_transform` açıkça saklanır.

`model_token_counts={cl100k_base: integer, o200k_base: integer}` iki gerçek
ölçümdür. `lengths.model_token_count` varsayılan o200k_base ölçümüdür;
`model_tokenizer_status=measured` ve tokenizer kimliği eklenir. Referans
Unicode/whitespace/karakter/UTF8 ölçümleri de korunur. İnsan incelemesi
`pending`, doğrulayıcı deney için uygunluk `false` olarak kalır.

Bu sayımlar **tam görünür düz metnin gerçek BPE sayılarıdır**; sağlayıcının
chat şablonu, sistem mesajı, rol ve araç ek yükünü içermez. Yeni bir model
farklı tokenizer kullanırsa kendi tokenizer/revision ölçümü eklenmelidir.
İki mevcut kontrolün eşlemesi ve sözlükleri sabittir; model API'sine metin
veya veri gönderilmemiştir. Kapsam [resmî sayım belgesinde](https://developers.openai.com/api/docs/guides/token-counting)
açıklanan yerel düz metin sayımıdır.

## İnsan onayı ve yorum sınırları

İnsan dilsel incelemesi tamamlanmış değildir. Eski 252 örnek yeni bütün
koşulların insan onayı değildir. P4/P5 çoklu dönüşümdür; F bir eşdeğerlik
çifti değildir. İsim/çeviri/sunum kopyaları ve aynı metni üreten kontrol
koşulları bağımlıdır. Model performansı, güç analizi, fine-tuning veya sonraki
prompt çalıştırılmamıştır.

## Önceki v0.4.0 JSONL kartı — tarihsel veri sözleşmesi


Sürüm 0.4.0 · 2026-10-23 · Sentetik mantıksal problemler, model çıktısı yok.
Yöntem kararları [protokol §13](EXPERIMENT_PROTOCOL.md), bütün hücreler
[deney matrisi](EXPERIMENT_MATRIX.md), gerçek hashler
[experiment_dataset_manifest.json](experiment_dataset_manifest.json) içindedir.

### Amaç ve kapsam

Aynı strict total order ve aynı sorgu EN/TR sunulur. Latent pozitif yön,
ilişkiye göre ön/sağ/üst/erken/uzun/hızlı/ağır/üst sıra anlamına gelir;
`before` bir dil sözcüğü değil mantıksal enum'dur. Her zincirde n benzersiz
varlık, n−1 doğru komşuluk öncülü vardır. Eşitlik yoktur. Yalnız belirtilen
kapalı cümleler kullanılır. Tam adlar gerçek kişiler hakkındaki veri değildir;
sabit ad parçaları havuzundan kombinasyonlardır.

Ana kaynak Türkçe tam adlardır. Önceki seed=11 karma isimli veri korunur,
yeni frozen test olarak adlandırılmaz. Kontrol koşulları:

| name_control | İngilizce görünüm | Türkçe görünüm | Yorum |
|---|---|---|---|
| native_shared | Özgün Unicode ad | Aynı özgün ad | Birincil dil kontrolü |
| en_transliterated | ASCII transliterasyon | Özgün Türkçe ad | Yazım farkı ek faktördür |
| symbolic_shared | Edddd kimliği | Aynı Edddd kimliği | İkinci ortak isim kontrolü |

Transliterasyon ç→c, ğ→g, ı→i, ö→o, ş→s, ü→u ve büyük harf karşılıklarını
kullanır. Çakışma reddedilir; kaynak 1.536 tam adın transliterasyonları
benzersizdir. Semboller 2.048 kimlik havuzundan tekrarsız, sıradan bağımsız
seçilir ve alfabetik kaynak adlara atanır; latent rank etiketi değildir.
Türkçe sembol ekleri önceki açık rakam okunuşu kuralını kullanır.

### Splitler ve bağımsız gözlem birimi

| Split | Seed | n değerleri | n başına sıra | Bağımsız sıra |
|---|---:|---|---:|---:|
| training | 401 | 4..1024, 2'nin kuvvetleri | 50 | 450 |
| development | 402 | 4..1024, 2'nin kuvvetleri | 50 | 450 |
| pilot | 403 | 4..128, 2'nin kuvvetleri | 50 | 300 |
| test | 404 | 4..1024, 2'nin kuvvetleri | 200 | 1.800 |

Toplam 3.000 latent sıra vardır. Bütün dil, ilişki, isim, sunum ve sorgu
varyantları bu sıra üzerinden bağımlıdır; prompt sayısı bağımsız n değildir.
`latent_instance_id=ordering_id` split anahtarıdır. `query_instance_id`
sorgunun ayrı kimliğidir. Aynı sıranın endpoint/internal/distance türevleri
aynı splitte kalır. 50'lik bloklar 13/12/12/13 veya 12/13/13/12 hücreleriyle
hem gold hem query-direction marjininde 25/25; 200'lükler her hücrede 50'dir.
İsim dönüşümü sonrası tam sıralama ve görünür prompt hashleri splitler arası
tekrara karşı ayrıca denetlenir. Seed11 endpoint kaynakları da dış guard'dır.
Bu, farklı isomorfik zincirleri farklı mantık kuralları yapmaz ve modelin
ön eğitimi hakkında kontaminasyon garantisi vermez.

### Dosyalar ve JSONL şeması

Her splitte dört UTF-8, LF, sıralı anahtarlı JSONL ve bir split manifesti
bulunur. Dış `experiment_dataset_manifest.json` bunların tamamını SHA-256,
bayt uzunluğu ve satır sayısıyla dondurur. Kimlikler SHA-256 içeriklidir.
Gold boolean'dır; görünür yanıt yönergesi her iki dilde YES/NO kullanır.

`groups.jsonl` bir latent sıra başına bir kayıt:

| Alan | Tür / anlam |
|---|---|
| latent_instance_id, split | string; sıra kimliği ve partition |
| n, sample_index, seed | integer; kaynak generator parametreleri |
| latent_order | string[n]; kaynak adlarla strict order |
| name_maps | control → language → source_name:display_name; bijektif eşleme |
| presentation_plans | condition → entity_order, premise_order, converse_chain_positions, style |
| named_order_sha256 | string[]; bütün alias görünümlerinin sıralama hashleri |

`entity_order`, alfabetik kaynak ad listesinin indeksleri; `premise_order`,
kaynakta karışık öncül listesinin indeksleri; `converse_chain_positions`,
latent zincirin 0..n−2 kenar konumlarıdır. Bunlar birbirine karıştırılmaz.

`latent.jsonl` bir sıra/sorgu kolu/mesafe başına bir kayıt:

| Alan | Tür / anlam |
|---|---|
| latent_instance_id, query_instance_id, split | string; grup ve sorgu bağlantısı |
| record | Önceki generator'ın değişmeyen, doğrulanan 16 alanlı nesnesi |

İç record: instance_id, ordering_id, sample_index, seed, n, entity_type,
latent_order, premise_edges, query_subject, query_object, query_relation,
query_type, query_direction, query_distance, gold_label, generator_version.

`prompts.jsonl` bir görünür deney girdisi başına bir kayıt:

| Alan | Tür / anlam |
|---|---|
| schema_version | integer=1 |
| dataset_version | string=0.4.0 |
| prompt_id, latent_instance_id, query_instance_id, split | string; benzersiz prompt ve kaynak bağlantısı |
| n, query_distance | integer; varlık sayısı ve latent pozisyon farkı |
| query_type | endpoint / internal / distance_controlled |
| query_subject, query_object | Kaynak ad; iki dilde aynı mantıksal hedef |
| query_relation | before / after; iki dilde aynı sorgu ilişkisi |
| query_direction | forward / reverse; argümanların latent pozisyon yönü |
| gold_label | boolean; iki exact oracle'ın ortak cevabı |
| language, condition, name_control, relation_id | string; deney faktörleri |
| display_query_subject, display_query_object | Dile özgü görünür adlar |
| presentation_plan_sha256 | string; ortak liste ve converse planının hash'i |
| prompt, prompt_sha256 | string; gerçek tam prompt ve UTF-8 SHA-256 |
| lengths | Aşağıdaki ölçüm nesnesi |
| human_review_status | pending |
| eligible_for_confirmatory_evaluation | false |

`lengths`: utf8_bytes, unicode_codepoints, whitespace_token_count,
reference_token_count (integer); tokenization_id, unicode_database_version,
model_tokenizer_status (string); input_is_nfc (boolean); model_token_count
(null). Referans tokenizer, Unicode L/M/N kategorilerinin bitişik dizilerini
bir token, kalan her boşluk dışı karakteri bir token sayar. Girdi normalize
edilmez. Bu gerçek bir tanımlı referans ölçümdür; model BPE sayısı değildir.
Sistem mesajı, sağlayıcı chat şablonu ve çıktı sayılmaz. Model/revision henüz
seçilmediğinden model token analizi tamamlanmış sayılamaz. Artifact tokenizer
kimliği manifesttedir; Unicode veri tabanı sürümü de kimliğin parçasıdır.

`cells.jsonl` bir karşılaştırma başına bir kayıt:

| Alan | Tür / anlam |
|---|---|
| cell_id, arm, split, latent_instance_id | string; benzersiz karşılaştırma/grup |
| n, query_distance | integer; n ve sağ prompt'un mesafesi |
| name_control, relation_id | string; ortak faktörler |
| left_prompt_id, right_prompt_id | string; aynı split prompt bankasına başvuru |
| left_condition, right_condition, left_language, right_language | string; karşılaştırılan sunumlar |
| left_query_type, right_query_type | string; F kolu sorgu türlerini ayrı tutar |
| query_direction, gold_label | Sağ prompt'un yönü ve boolean gold'u |
| comparison_kind | same_query / query_complexity |
| study_track | reference_structural / turkish_extension |

A–E'de hedef, n, sıra, mesafe, yön, ilişki ve gold aynıdır. C/D'de iki dilin
sunum planı hash'i de aynıdır. F'de hedef ve gold değişebilir; iki prompt'un
gold'u ayrı doğrulanır. F bir eşdeğerlik/tutarlılık çifti değildir.
`reference_structural`, yalnız A–D/front/endpoint için kullanılır; tam yayın
replikasyonu iddiası değildir. Kaynak [CBE makalesinin Appendix B'sidir](https://arxiv.org/html/2610.22894v1).

### Doğrulama ve kullanım sınırları

Üretimde ve derin doğrulamada actual EN/TR cümleleri normal forma ayrıştırılır;
aliaslar geri çözülür, bütün öncüller kaynak zincire eşitlenir, sorgu üçlüsü ve
iki oracle denetlenir. Çiftlerde metadata ve ortak plan denetlenir. Dosya,
şema, kaynak, split, sayı ve hash denetimleri birlikte kullanılır.

Bu veri model performansı, dilsel insan onayı, sözcük sıklığı araştırması veya
fine-tuning sonucu içermez. Dokuz ilişki için insan incelemesi beklenir.
Önceki 252 satırlı rapor yeni İngilizce ve matched koşulların tamamının insan
onayı değildir. P3 odak/pragmatiği değiştirebilir; P4/P5 syntax ve lexical
etkiler de içerir. Frozen test yapısal olarak sabittir; bu aşamadaki geliştirme
erişimi gelecekteki önkayıt/disiplin gerekliliğini ortadan kaldırmaz.
Lisans, model sürümleri, gerçek tokenizer, değerlendirme runner'ı ve preregistered
analiz planı sonraki model değerlendirmesi öncesinde ayrıca tamamlanmalıdır.

Farklı deney etiketleri aynı görünür metni üretebilir: en_transliterated
koşulunun Türkçe metni native_shared ile aynıdır; n=4'te internal ve d=1
aynı hedefi seçer. Bu tekrarlar aynı splitte korunur ve prompt satırı sayısı
benzersiz metin sayısı olarak sunulmaz. Sonraki model planı bu bağımlılığı
ve aynı metin için çağrı/tekrar politikasını açıkça belirlemelidir.


### Tam veri doğrulaması — 2026-10-24

20 veri dosyasının hash/boyut ve derin yeniden üretim kontrolleri geçti.
3.000 grup, 23.200 sorgu, 153.400 prompt satırı, 145.602 farklı görünür metin
vardır. Splitler arasında grup, alias sıralaması veya görünür prompt tekrarı
yoktur. Gold dağılımı source sorgularda 11.600/11.600, promptlarda
76.700/76.700'dür. Testin dört hücresinin her biri her blokta 50 örnektir.
Bu kontroller insan dilsel onayı veya model performans sonucu değildir.
Kök manifest SHA-256: `e016f612eebbe582de83792935047ab8ad87e371d62e5dec24bd834f97b89669`.
