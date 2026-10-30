# Turkish-CBE deney matrisi — güncel v0.4.1

Güncel tam faktör matrisi `data/complete_grid/` içindedir. Önceki v0.4.0
hedefli JSONL bankası değişmeden korunur; aşağıdaki son bölüm tarihsel kapsamıdır.
Yeni matris aynı 3.000 latent sıranın bütün geçerli sorgularını kullanır.

## Tam çaprazlama

Her kaynak sorgu **3 isim kontrolü × 9 ilişki × 9 dil/sunum koşulu = 243**
ölçülmüş prompt kaydı içerir. Hiçbir n, isim kontrolü, ilişki veya zorunlu
sorgu kolu örneklenerek azaltılmaz. İki dilde canonical/equivalent; Türkçede
ek olarak P1..P5 vardır. Türkçeye özgü ablation'lar için İngilizce koşul icat
edilmez. İsteğe bağlı P6 bu matrisin dışındadır.

| Faktör | Düzeyler |
|---|---|
| İsim | native_shared, en_transliterated, symbolic_shared |
| İlişki | spatial_front/right/above, temporal_arrival/completion, comparison_height/speed/weight, ranking |
| EN sunum | canonical, equivalent |
| TR sunum | canonical, equivalent, P1, P2, P3, P4, P5 |
| Sorgu | endpoint, internal, her geçerli 2'nin kuvveti d için distance_controlled |
| Boyut | Test/training/development: 4..1024; pilot: 4..128, 2'nin kuvvetleri |
| Dört hücre | query-direction forward/reverse × gold TRUE/FALSE |

| Split | Latent sıra | Sorgu | Ön bileşen | Ölçülmüş prompt | Karşılaştırma |
|---|---:|---:|---:|---:|---:|
| training | 450 | 3.550 | 109.350 | 862.650 | 1.197.450 |
| development | 450 | 3.550 | 109.350 | 862.650 | 1.197.450 |
| pilot | 300 | 1.900 | 72.900 | 461.700 | 634.500 |
| test | 1.800 | 14.200 | 437.400 | 3.450.600 | 4.789.800 |
| **Toplam** | **3.000** | **23.200** | **729.000** | **5.637.600** | **7.819.200** |

Testte her n ve kaynak sorgu bloğunda dört hücrenin her birinde 50 sıra
bulunur. Diğer splitlerde 12/13 hücre ve 25/25 iki marjin korunur. Internal
sorgunun gözlenen mesafesine göre ayrıca tabakalama yapılabilir; bu kolun
her gözlenen d hücresinde 50 örnek varmış gibi yorumlanmaz. Distance_controlled
kolunda d baştan sabittir. Geçerli mesafelerin listesi aşağıdaki tarihsel
n tablosuyla aynıdır; d<=n−3 koşulu değişmez.

## A–F: sürümlenmiş genişletilmiş sözleşme

| Kol | Karşılaştırma | Kapsam |
|---|---|---|
| A | EN canonical → EN equivalent | Her sorgu × isim × ilişki |
| B | TR canonical → TR equivalent | Her sorgu × isim × ilişki |
| C | EN canonical → TR canonical | Her sorgu × isim × ilişki |
| D | EN equivalent → TR equivalent | Her sorgu × isim × ilişki |
| E | TR canonical → TR P1..P5 | Her sorgu × isim × ilişki |
| F | Endpoint → internal/distance | Her isim × ilişki, EN/TR canonical/equivalent |

Kaynağa yakın yapısal temel kolun açık seçim kuralı: **A–D AND
relation_id=spatial_front AND query_type=endpoint**. Diğer hücreler Türkçe/
ilişki/sorgu uzantılarıdır; sonuçlar bu filtreyle ayrı raporlanmalıdır.

A–E aynı sorgudur ve gold aynıdır. F aynı latent sıralamada sorgu
karmaşıklığı karşılaştırmasıdır; eşdeğerlik çifti değildir. Genişletilmiş E,
önceki v0.4.0 yalnız endpoint/native E'den ayrı doğrulayıcıyla sürümlenir.
Bütün dil/isim/sunum/sorgu türevleri latent sıra bazında bağımlıdır.

| Kol | training | development | pilot | test | Toplam |
|---|---:|---:|---:|---:|---:|
| A | 95.850 | 95.850 | 51.300 | 383.400 | 626.400 |
| B | 95.850 | 95.850 | 51.300 | 383.400 | 626.400 |
| C | 95.850 | 95.850 | 51.300 | 383.400 | 626.400 |
| D | 95.850 | 95.850 | 51.300 | 383.400 | 626.400 |
| E | 479.250 | 479.250 | 256.500 | 1.917.000 | 3.132.000 |
| F | 334.800 | 334.800 | 172.800 | 1.339.200 | 2.181.600 |

## Gerçek tokenizer ve saklama

Her prompt için cl100k_base ve o200k_base gerçek BPE sayısı saklanır;
referans modeller gpt-4-turbo-2024-04-09 ve gpt-4o-2024-08-06'dır. Bunlar
API çağrısı veya model performans sonucu değildir. Sözlük/paket sürümü ve
hashler `tokenizers/encodings.json`, `requirements-tokenizers.lock` ve tam
matris manifestinde sabittir. Düz metin sayımı chat/rol/sistem ek yükünü içermez.

SQLite `measurements` tablosunda her 243 koordinatın gerçek hash ve sayıları
bulunur. `comparison_cells` görünümü bütün 7.819.200 karşılaştırmayı açıkça
sunar. Bu, üretilmemiş hücrelerin var sayılması değildir; bütün prompt metinleri
kaynak ve koordinattan birebir yeniden açılabilir. JSONL dışa aktarımı
README'deki komutla yapılır. Fiziksel tam-metin satırı sayısı ile ölçülmüş
prompt kaydı ve benzersiz metin sayısı birbirine karıştırılmaz.

P4/P5 çoklu morphology/syntax/lexical müdahaledir. Yalnız Türkçe ablation
sayısı yüksek olduğundan bütün EN/TR satırlarının ham ortalaması dil etkisi
sayılmaz; karşılaştırma C/D gibi eşleşmiş hücrelerde yapılmalıdır.

## Önceki v0.4.0 hedefli matris — değiştirilmeden korunan veri


Bu matris üretilen veriyi tanımlar; model deneyi çalıştırıldığı anlamına gelmez.
Sabit konfigürasyon [configs/matched_dataset.toml](configs/matched_dataset.toml),
makinece okunabilir bütün hücre sayıları
[experiment_dataset_manifest.json](experiment_dataset_manifest.json) içindedir.

### Kollar

| Kol | Sol → sağ | İsim koşulları | Sorgu | Track |
|---|---|---|---|---|
| A | EN canonical → EN equivalent | Üçü | Endpoint | reference_structural |
| B | TR canonical → TR equivalent | Üçü | Endpoint | reference_structural |
| C | EN canonical → TR canonical | Üçü | Endpoint | reference_structural |
| D | EN equivalent → TR equivalent | Üçü | Endpoint | reference_structural |
| E | TR canonical → TR P1..P5 | native_shared | Endpoint | turkish_extension |
| F | Endpoint → internal veya distance | native_shared | Ayrı query blokları | turkish_extension |

A–D çekirdeği front/behind ilişkisidir. Pilot endpoint/native_shared üzerinde
sekiz ek ilişki de A–E ile üretilir: spatial_right, spatial_above,
temporal_arrival, temporal_completion, comparison_height, comparison_speed,
comparison_weight, ranking. Bunlar `turkish_extension` etiketlidir.
Bütün kollar aynı kaynak sıra grubundan türetilir; bağımsız örnekler olarak
toplanamaz. F, EN/TR × canonical/equivalent olmak üzere dört tabakada bulunur.
F'de farklı sorgular eşdeğerlik çifti gibi puanlanamaz. Endpoint ve internal/
distance sonuçları ayrı raporlanmalıdır.

`canonical`: bütün komşu kenarlar pozitif ilişki, karışık öncül/isim listesi.
`equivalent`: aynı n−1 kenarın m tanesi argüman takası ve converse sözcükle
yazılır; m düzgün tamsayı örneklemesidir. Yeniden yazılan zincir konumları
düzgün, tekrarsız seçilir; her iki liste ayrıca karıştırılır. Bu plan dil,
isim, ilişki ve sorgu hedefinden bağımsızdır ve aynı sırada korunur.
Kaynağa yakın yapısal kontrol [makalenin protokol P açıklamasına](https://arxiv.org/html/2610.22894v1)
dayanır. Kaynak yayınla aynı isim havuzu, seed, tam prompt bağlamı veya model
çıktısı kullanılmadığından birebir doğrudan performans replikasyonu değildir.

E/P1 yalnız iki listeyi; P2 bütün kenarların yön ifadesini; P3 cümle sırasını;
P4 iyelikli özellik kuruluşunu; P5 P1+P2+P4'ü değiştirir. P4/P5 çoklu değişim
olarak kaydedilir, saf morphology etkisi değildir. `equivalent` ile P2 farklıdır.
P6 bu aşamanın eşleştirilmiş veri setine dahil değildir.

### Boyut ve mesafe blokları

| n | Geçerli d (internal uçlar) | Sorgu bloğu sayısı |
|---:|---|---:|
| 4 | 1 | 3 |
| 8 | 1,2,4 | 5 |
| 16 | 1,2,4,8 | 6 |
| 32 | 1,2,4,8,16 | 7 |
| 64 | 1,2,4,8,16,32 | 8 |
| 128 | 1,2,4,8,16,32,64 | 9 |
| 256 | 1,2,4,8,16,32,64,128 | 10 |
| 512 | 1,2,4,8,16,32,64,128,256 | 11 |
| 1024 | 1,2,4,8,16,32,64,128,256,512 | 12 |

Blok sayısı = endpoint + internal + her geçerli d. Pilot 38, diğer splitler
71 blok içerir. Internal kolu iki interior pozisyonu rastgele seçer; seçilen
mesafe ayrıca saklanır. Distance kontrollüde d sabittir, başlangıç pozisyonu
rastgeledir. n−3'ten büyük d kullanılmaz. Bu tasarımda uçta sorgulanan ilişki
zincirin tamamından, diğer sorgular ilgili altzincirden çıkarılabilir.

### Kesin hacimler

| Split | Latent sıra | Latent sorgu | Tam prompt | Karşılaştırma |
|---|---:|---:|---:|---:|
| training | 450 | 3.550 | 20.050 | 20.050 |
| development | 450 | 3.550 | 20.050 | 20.050 |
| pilot | 300 | 1.900 | 33.100 | 33.100 |
| test | 1.800 | 14.200 | 80.200 | 80.200 |
| Toplam | 3.000 | 23.200 | 153.400 | 153.400 |

Her sırada endpoint için 3 isim × 2 dil × 2 sunum = 12 çekirdek prompt,
E için 5 Türkçe prompt; her ek sorgu için 4 native_shared prompt vardır.
Pilotta her sıra için sekiz ek ilişki × (4 bilingual + 5 E) = 72 prompt eklenir.
Karşılaştırmalar aynı prompt bankasını kullanır; aynı toplamlar tasarımın
sayısal sonucudur, her promptun tek karşılaştırmaya girdiği anlamına gelmez.

| Kol | training | development | pilot | test |
|---|---:|---:|---:|---:|
| A | 1.350 | 1.350 | 3.300 | 5.400 |
| B | 1.350 | 1.350 | 3.300 | 5.400 |
| C | 1.350 | 1.350 | 3.300 | 5.400 |
| D | 1.350 | 1.350 | 3.300 | 5.400 |
| E | 2.250 | 2.250 | 13.500 | 9.000 |
| F | 12.400 | 12.400 | 6.400 | 49.600 |

A–D pilot sayılarının her birinde çekirdek 900, ek sözlük 2.400'dür. Her
200'lük source query bloğu direction×gold hücresinde tam 50; 50'lik blokta
12/13 bulunur. Ekstra iki hücre iki marjini dengeler. Aynı sayılar her isim,
dil ve sunum kopyasında korunur; E/F bağımlılığı gözetilir.

Bütün tekil hücrelerin `(arm,n,name_control,relation_id,languages,conditions,
query_type,distance,direction,gold)` sayıları her splitin `split_manifest.json`
ve kök manifestin `comparison_cell_counts` alanındadır. Prompt hücreleri
`prompt_cells`, source denge `query_cells` ile raporlanır. Üretilmeyen tam
isim×ilişki×query×varyant çapraz hücreleri sıfır varsayılarak analiz edilmez.

Performans metrikleri, istatistiksel testler, güç analizi ve model token
sayıları burada hesaplanmadı. Bunları belirleyen sonraki aşama başlatılmadı.

E yalnız Türkçe olduğundan bütün promptların dil bazında kaba ortalaması
dil etkisi tahmini değildir. Token uzunluğu/dil analizinde C/D gibi aynı
hücrede eşlenmiş kayıtları ve isim kontrolünü kullanın.

Liste permütasyonu identity kalırsa rotasyon uygulanır; listeler değişir fakat
permütasyon dağılımı tam uniform değildir. Converse kenar konumları için
tekrarsız uniform örnekleme aynen geçerlidir.
