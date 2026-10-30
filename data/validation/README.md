# Türkçe dilsel kontrol raporu

Renderer sürümü 0.3.0, renderer seed 31. Bu klasör model çıktısı içermez.
[turkish_samples.csv](turkish_samples.csv) **252 tam prompt çifti** içerir:
36 farklı latent kaynak sıra × P0–P6. Her satırda P0 referansı ile incelenecek
varyantın öncülleri, sorgusu, bağlamı ve tam promptu yan yanadır.

**İnsan incelemesi yapılmadı.** İki değerlendiriciye ait bütün karar alanları
boştur; bütün satırlar `pending` durumundadır. Otomatik başarı doğal Türkçe
veya insan tarafından onaylanmış anlamsal eşdeğerlik iddiası değildir.

## Gerçek kapsam

| Boyut | Sayı |
|---|---:|
| İlişki | 9 (mekânsal 3, zamansal 2, karşılaştırma 3, sıralama 1) |
| Her ilişki | 28 satır: dört sorgu hücresi × yedi varyant |
| Her varyant P0..P6 | 36 satır |
| Her isim türü | 84 satır |
| Her sorgu kolu | 84 satır |
| Her direction × gold hücresi | 63 satır |
| True / false | 126 / 126 |
| n=4 / n=8 | 126 / 126 |
| Tamamlanmış insan incelemesi | 0 |

Kaynaklar önceki veri manifestinin endpoint/internal/distance-d1 n=4,8
bloklarından hash kontrolüyle seçildi. 36 `ordering_id` farklıdır. Her
kaynak için P0–P6 eşleniktir. Farklı ilişkilerin inceleme kaynakları farklıdır;
bu örneklem ilişkiler arasında model başarısı karşılaştırma veri kümesi değildir.
API herhangi bir latent kaydı dokuz ilişkinin tamamında üretebilir.

[turkish_samples_manifest.json](turkish_samples_manifest.json) CSV hashini,
kaynak veri manifestini, altı kaynak JSONL hashini, 36 kaynak kimliğini,
renderer/normalizer kod hashlerini, seed ve dağılımları kaydeder.

## Faktör sözleşmesi

| Faktör | Değişim | Etiket / sınır |
|---|---|---|
| P0 | Kanonik Türkçe | Referans |
| P1 | Yalnız isim ve öncül listelerinin permütasyonu | Cümleler ve sorgu aynı |
| P2 | Bütün öncüllerde argüman takası + converse sözcüğü | Liste sırası aynı; tek ifade-yönü faktörü |
| P3 | Tamamlayıcı–yüklem–özne sırası | Aynı sözcük/ekler; devrik, odak/pragmatik işaretli |
| P4 | İyelikli özellik ve durum eki yapısı | **Morphology + syntax + lexical**, çoklu dönüşüm |
| P5 | P1 + P2 + P4 | Açık birleşik faktör |
| P6 | Karşıt atomik ilişkinin olumsuzlanması | Varsayılan kapalı; bu raporda açıkça etkin; negation + lexical |

Sorgu ve doğru cevap bütün varyantlarda aynıdır. Mekânsal eksen sabittir;
üst/alt fiziksel temas değildir. Tamamlanmada etiketler görev adıdır. P6 için
farklı a,b ve strict total order altında tam olarak biri doğrudur:
`before(a,b)` veya `after(a,b)`. Dolayısıyla `not after(a,b)` ile `before(a,b)`
eşdeğerdir. Eşitlik, karşılaştırılamazlık veya tamamlanmamış/gelmemiş olay
olasılığı varsa bu dönüşüm kullanılamaz. Bağlam bu olasılıkları açıkça dışlar.

## Morfoloji ve okunuş kuralları

Tam adın son sözcüğünün okunuşu ekin ünlüsünü belirler. İlgi/tamlayan eki
`-(n)In` dört yönlü; ayrılma eki `-DAn` iki yönlü ünlü uyumu kullanır.
Sert son ünsüzden sonra ayrılma eki t ile başlar. Özel adın yazımı korunur:
`Çelik'in`, `Çelik'ten`; yazıda kök `Çeliğ` yapılmaz. Standart kesme ASCII `'`.
Ortak adlardaki `önü-n-de-dir`, `arkası-n-da-dır`, `ağırlık→ağırlığı` gibi
biçimler kapalı ve açık şablon sözlüğünde tanımlıdır; serbest sözcük çekimi yoktur.

Edddd etiketleri E harfi ve rakamlar tek tek Türkçe okunur. E0004 son sözcük
`dört` olduğu için `E0004'ün/E0004'ten`, E0002 için `E0002'nin/E0002'den`.
Bu deneysel okunuş sözleşmesi ilgili promptta görünür. Genel özel ad sözlüğü
veya genel Türkçe çözümleyici sunulmaz; belirsiz okunuşlar açık ipucu ister.

Kurallar için [TDK kesme](https://tdk.gov.tr/icerik/yazim-kurallari/noktalama-isaretleri-aciklamalar/),
[büyük ünlü uyumu](https://tdk.gov.tr/icerik/yazim-kurallari/buyuk-unlu-uyumu/),
[küçük ünlü uyumu](https://tdk.gov.tr/icerik/yazim-kurallari/kucuk-unlu-uyumu/)
kontrol edildi (2026-10-23). Bu kaynaklar şablonların insan onayı değildir.

## İnsan incelemesi nasıl yapılmalı?

CSV'yi UTF-8 ve çok satırlı hücre desteğiyle açın. Otomatik özgün dosyayı
koruyup ayrı bir kopyada çalışın. Birinci ve ikinci değerlendirici bağımsızca:

1. `grammar`, `morphology`, `equivalence`: `yes/no/uncertain` yazsın.
2. `naturalness`: 1–5 (1 doğal değil; 5 bağlama uygun doğal) puanlasın.
3. `contact_ambiguity`: mekânsal cümlede fiziksel temas/başka yön okuması
   mümkün mü `yes/no/uncertain`, diğerlerinde `not_applicable` yazsın.
4. Sorunu `notes` ile açıklasın; anlaşmazlık ortak, gerekçeli `resolution`
   ile çözülsün. Otomatik true alanları insan kararına kopyalanmasın.

Dilbilgisi/ek/anlamda `no` veya çözümsüz belirsizlik varsa şablon model
değerlendirmesine alınmaz. P3'ün odak etkisi, P4 özellik adlarının doğallığı,
üst/altın temas çağrışımı ve isim okunuşları özellikle incelenmelidir.
İnceleme sayısı ve kişiler belirtilmeden `approved` yazılmaz.

Bu ilk kontrol örneklemi, protokolün gelecekte her deney hücresinden en az
20 örnek ve bütün şablonlarda çift değerlendirici şartını tamamlamaz.
`eligible_for_confirmatory_evaluation=false` bütün otomatik sunumlarda kalır.
İnsan kararlarını işleyen/onaylayan bir inference pipeline bu aşamada yoktur.

## Yeniden üretme

Proje kökünde, mevcut raporu koruyarak yeni hedefe:

```sh
PYTHONPATH=src python3 -m renderers.review --dataset-manifest dataset_manifest.json --output work/turkish_samples.csv --manifest work/turkish_samples_manifest.json --seed 31 --include-negation
```

`--include-negation` yoksa P0–P5 için 216 satır üretilir. Mevcut çıktıların
üzerine yazılmaz. Aynı kaynak/kod/seed aynı CSV baytlarını üretir; manifest
ortam metaverisi farklı Python/OS'ta değişebilir. Kaynak latent dosyalar
değiştirilmez ve model API'sine bağlanılmaz.
