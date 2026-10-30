# Sözcük sıklığı, tokenizasyon ve önyargı — v0.7.0

Prompt 6 tamamlandıktan sonra uygulanmış korelasyonel ek analizdir. **Gerçek
LLM yanıtı veya performans/önyargı bulgusu yoktur.** Teslimdeki sayımlar,
bağımsız metin korpusu ve tokenizer ölçümleridir. Önceki 50 Python kaynağı,
frozen veri, model ayarları ve Prompt 5 dry-run planları korunur.

## Gerçek olarak ölçülen kapsam

| Katman | Teslim kapsamı |
|---|---|
| İlişki kataloğu | Dokuz ilişki, 18 forward/inverse yön hücresi, 14 benzersiz lemma |
| Bağımsız korpus | UD Turkish-BOUN r2.18; üç CoNLL-U parçası, 9.761 cümle |
| Yüzey sayımı | 121.835 toplam token; noktalama çıkarılmış payda 101.052 |
| Lemma katmanı | 125.212 syntactic word; noktalama çıkarılmış payda 104.424 |
| Morphological MWT | 3.374; yüzeyde tek, lemma katmanında bileşenler ayrı |
| Sıklık tablosu | 18 yön satırı; lemma sayımları pozitif, literal predicate biçimleri sıfır |
| Sözlük tokenizasyonu | 216 satır: altı yapılandırma×18 yön×iki dil |
| Tam prompt incelemesi | 243 donmuş pilot önizleme promptu×altı yapılandırma=1.458 satır |
| Mevcut düz metin sayımı | Dört yapılandırma için 972 prompt satırı; iki Llama için null |
| Yerel chat sayımı | Qwen HF 0.5/1.5 ve Qwen vLLM için 729 satır |
| Model yanıtı/API çağrısı | 0; beş performans/önyargı analizi henüz yapılamaz |

Kaynak [UD Turkish-BOUN](https://universaldependencies.org/treebanks/tr_boun/index.html)
CC BY-SA 4.0 lisanslıdır. Sabit revision
`0def94a229dcbc28d32192b0138b560c075c6d87`; dağıtım r2.18. Orijinal
README ve LICENSE, yazar/kurum atfı ve bütün dosya SHA'ları
`data/corpora/ud_turkish_boun/r2.18/` altında tutulur. Metinler TNC kaynaklı
haber/nonfiction seçkisidir; genel Türkçeyi veya LLM eğitim verisini temsil
ettiği varsayılmaz. Corpus train/dev/test parçaları yalnız bağımsız frekans
sayımı için birleşir; Turkish-CBE deney splitleriyle ilişkili değildir.

## İlişki ve sıklık sözleşmesi

`analysis/lexical_bias/relation_vocabulary.json` dondurulmuş EN/TR renderer’dan
üretilir. Forward/inverse lemma, canonical predicate, gerçek soru ve P4 biçimleri,
İngilizce karşılık, Türkçe karakterler, iyelik/ayrılma ekleri, kesme işareti,
ünlü/ünsüz uyumu ve positional morfemler saklanır. Örnek isimler iki dilde
aynıdır. Bu kapalı metadata genel morfolojik çözümleyici veya insan dilsel onayı
değildir. Katalog hâlâ human_review_status=pending olarak işaretlidir.

Surface katmanında CoNLL-U multiword token aralığının gerçek yüzey FORM'u bir
kez sayılır; bileşen integer satırları ikinci defa sayılmaz. Lemma katmanında
integer syntactic-word LEMMA alanları kullanılır. Noktalama/eksik lemma/cümle
ve dosya sınırları n-gram bağlantısını keser. Case normalizasyonu NFC ve
Türkçe I→ı/İ→i dönüşümüdür. Çok sözcüklü ifadelerde payda ilgili uzunluktaki
eligible n-gram fırsatlarıdır; tek sözcükte non-punctuation token paydasıdır.

Her yön için canonical surface count, her gerçek varyantın ayrı yüzey sayımı,
lemma count, fırsat paydası, per-million, log10(per-million) ve inverse yönüne
oran bulunur. Smoothing yoktur. Ölçülen sıfır count korunur; log10(0) ve
inverse frekansı sıfır olan oran null olur. Korpus eksikse bütün frekanslar
null/missing_corpus; bozuk hash veya eksik ilan edilmiş dosyada işlem reddedilir.

Bu küçük korpusta tam canonical/query/P4 predicate'lerin **ölçülmüş sayımı 0**
oldu. Bu, Türkçede kullanılmadıkları veya modelin onları bilmediği anlamına
gelmez. Ana lexical familiarity değişkeni canonical soru ilişkisinin lemma
sıklığıdır. P4'te erken/geç, yüksek/düşük, fazla/az gibi başka kökler
kullanıldığında bu kökler metadata'da ayrıdır; canonical lemma sıklığı onların
sıklığı diye sunulmaz. sağ, ağır, uzun, üst/alt gibi çok anlamlı sözcüklerde
sense/POS ayrıştırması yapılmadı. 18 hücre 18 bağımsız sözcük değildir: üst/alt
ve önce/sonra tekrarlarıyla yalnız 14 kök vardır.

## Tokenizer kapsamı ve yeniden üretim

Modeller `configs/models.yaml` içinden seçilir; kodda model adı sabitlenmez.
Ölçüm modelin kullanılabilirliğini veya inference başarısını doğrulamaz.

- OpenAI: kayıtlı snapshot'a bağlı donmuş o200k_base ile gerçek düz metin
  BPE sayımı. Yeni system/user içerikleri tam birleştirilerek ayrı sayılır.
  Rol/framing dahil OpenAI chat sayımı null kalır; count API çağrısı yoktur.
- Qwen HF: immutable revision'a ait yalnız tokenizer.json/config/vocab/merges
  ve lisans dosyaları yerelde. Upstream Git blob veya LFS SHA, yerel SHA-256,
  boyut, model/revision ve chat template hash'i sayımdan önce doğrulanır.
  AutoTokenizer local_files_only=True, trust_remote_code=False ile çalışır.
  Generation prompt dahil yerel chat sayımı gerçek tokenizer ile yapılır.
- Qwen vLLM: aynı pinli tokenizer ve yapılandırılmış chat template'i sayılır.
  Bu sonuç çalışan vLLM sunucusunun template veya sürümünü tasdik etmez.
- Llama 1B/3B tokenizer erişimi HTTP 401 verdi. Başka tokenizer seçilmedi;
  satırlar null/missing_local_tokenizer_assets kalır. Model ağırlığı alınmadı.

`tokenizers/hf/manifest.json` tokenizer dosyalarını ve lisanslarını bağlar.
Bütün dosyalar teslimdedir; çevrimdışı sayım sırasında ağ erişimi gerekmez.
Küçük tokenizer dosyaları model ağırlıkları değildir. Artifact metadata kayıtları
tekilleştirilir; her tablo satırı tokenizer_provenance_sha256 ile manifestteki
aynı provenance kaydına bağlıdır. Model ID/revision/encoding satırda da bulunur.

Forward/inverse delta `after−before`, dil oranı TR/EN'dir. Oranlar yalnız aynı
latent/query/condition/name-control/relation ve aynı model/tokenizer eşleşmesine
aittir. English-transliterated isim kontrolünde dil ve isim etkisi karışabildiği
etiketlenir; ayrıca aynı dilde native_shared referansına fark tutulur.
Bu teslim yalnız n=4 pilot önizleme kapsamını ölçer; 5.637.600 satırlık bankanın
bütün modellerde yeni chat tokenizasyonu yapılmış olduğu iddia edilmez.
Gerçek run kullanımındaki input_tokens ayrıca provider-reported kapsamındadır;
düz metin/BPE proxy ile sessizce değiştirilmez.

```sh
python -m pip install --require-hashes -r requirements-inference.lock
python -m pip install --require-hashes -r requirements-tokenizers.lock
python -m pip install --require-hashes -r requirements-analysis.lock
python -m pip install --require-hashes -r requirements-lexical.lock
# Model/servis çalıştırmaz; yeni bir dizin kullanın.
PYTHONPATH=src USE_TORCH=0 USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 python -m analysis.lexical_bias --root . --output results/lexical/my_measurement
# Gerçek inference kayıtları oluştuğunda:
PYTHONPATH=src HF_HUB_OFFLINE=1 python -m analysis.lexical_bias --root . --run RUN_ID --output results/lexical/RUN_ID
```

Kurulu entry point `turkish-cbe-lexical` aynı arayüzdür. İsteğe bağlı --model,
--corpus-manifest ve --preview açık seçimdir. Prompt ölçümü 10.000 kayıt sınırında
korunur; default 243 kayıt. Yeni sonuç klasörü zorunludur. Input/kod/config,
korpus/tokenizer provenance, ortam ve her çıktı SHA/boyut bilgisi manifesttedir.
Transformers 4.57.1/Jinja2 3.1.6 tokenizer-only ortamı 20 paket/674 resmî PyPI
artifact hash'iyle ayrı kilitlidir; PyTorch kurulmaz. NumPy/PyYAML/regex gibi
ortak pinler mevcut kilitlerle aynıdır. CPython 3.12/3.13 macOS arm64 üzerinde
40 wheel ve import sürümleri doğrulandı. Başka platformlar denenmedi.
Yeni ortam kütüphaneleri inference planının ortam kaydını değiştirebilir;
eski bir run planı bu ortamda yürütülmeden önce yeni dry-run oluşturulmalıdır.

## Yön değişimi, accuracy ve invariance nasıl ayrılır?

Üç farklı yön kavramı raporda ayrı alanlardır:

1. `query_relation=before/after`: sorunun forward/inverse lexical ifadesi.
2. `query_direction=forward/reverse`: sorgu varlıklarının latent konum yönü.
3. Canonical/P2: öncüllerde özne/nesne takası ve converse ilişki ifadesi;
   aynı latent problem, aynı soru ve aynı gold korunur.

Sadece soru relation sözcüğünü değiştirmek gold'u değiştirebilir; bu iki
cevap doğrudan anlamsal eşdeğer çift sayılmaz. CBE yön müdahalesi aynı sorguda
canonical→P2 olarak ölçülür. Canonical→equivalent ise liste sırasını da
etkileyen çoklu dönüşümdür; P2 sonucu diye yorumlanmaz. Varsayılan eski pilotta
P2 yoksa bu yokluk neden alanında raporlanır, satır uydurulmaz.

Marjinal accuracy invalid=yanlış ve bütün planlı paydadadır. R/D/U ve invariance
aynı ortak geçerli çiftlerdedir; strict R ve coverage ayrıca korunur. Accuracy
farkı küçükken karşılıklı doğru→yanlış/yanlış→doğru değişimler birbirini
silebilir; U bu kırılganlığı gösterir. Invariance yüksekken D de yüksek olabilir:
aynı yanlış yanıtı tekrarlamak güvenilir doğru çıkarım değildir. Bu nedenle
paired_direction tablosu iki accuracy, R/D/U, invariance, strict R, coverage ve
query yönünü birlikte verir; tek sayı üzerinden başarı iddiası kurulmaz.

EVET/HAYIR oranı yalnız geçerli yanıtlardan, all-planned ve valid-only gold
oranları/invalid coverage yanında raporlanır. .90/.10 aşırı eğilim eşiği
önceden belirlenmiş betimsel tanıdır; anlamlılık testi değildir. Dört latent
query-direction×gold hücresi overall düzeyde denetlenir; sabit yön altgrupları
koşullu gold dağılımını açık tutar. Tekrarlanan latentler için Wilson q aralığı
bağımlılık uyarısı taşır. Mevcut doğrulanmış bankın planlı gold dengesi değişmez.

## Gerçek veri geldiğinde yapılacak istatistikler

Frequency–Accuracy, Frequency–R ve token count–invariance Spearman özetleri,
model/n/kol/sunum/isim kontrolleriyle lexical-direction profil seviyesindedir.
Bağlarda ortalama rank kullanılır; eksik veya sabit girdide null/status döner.
İngilizce yanıtlarına Türkçe korpus frekansı atanmaz. Aynı kökün tekrarları
bağımsız sözcük sayılmaz; n_unique_lemma_units açık verilir. IID p-değeri yoktur.

Keşifsel çok değişkenli OLS, model, log2(n), relation family, log(actual prompt
input tokens) ve lexical yönü kontrol eder. Accuracy modelinde invalid yanlış
kalır; predicted-YES modeli valid yanıtlar üzerinde ayrı kapsam/eksik sayımı
kullanır. R/invariance modelleri ortak valid çiftler üzerinde dışlanan çift
sayısını bildirir. Eksik kovaryat gizlice atılmaz; fit missing_values olabilir.
Rank eksikliği, az gözlem, sabit kontrol veya eksik interaction hücresinde fit
verilmez. Linear probability yaklaşımı betimseldir; bounded causal model değildir.

Model×vocabulary ayrı fixed-effects tasarımdır; frequency aynı modele konmaz.
Vocab ID 18 relation-direction tasarım hücresini, independent_vocabulary_id
14 lemma kökünü temsil eder. Böylece family/direction tasarımda sabit tutulur,
18 bağımsız kök varmış gibi davranılmaz. Benzersiz kök sayımı tek başına
istatistiksel bağımsızlık kanıtı değildir. Az vocabulary ve tekrar bağımlılığı
uyarısı, rank/condition number ve n_latent raporlanır. Naif standard error,
p-değeri veya güven aralığı üretilmez. Bu veriyle sözcük sıklığının nedensel
etkisi veya LLM iç parametre ağırlıkları ölçüldüğü iddia edilmez.

Şu anda gerçek model yanıtı bulunmadığından bu beş ilişki için katsayı,
p-değeri, performans tablosu ve araştırma grafiği üretilmedi. `results/lexical/`
betimsel preprocessing dosyalarını, `figures/lexical/` ise bu eksik durumu
belgeler. Sentetik unit testler geçici klasörlerde kalır.
