# Turkish-CBE deney protokolü

Sürüm: 0.7.0 · Tarih: 2026-10-25 · Durum: taslak; önkayıt değildir.
İlk sürüm proje kodundan önce oluşturuldu. Prompt 2'de latent üretici/oracle
uygulandı ve seed-11 geliştirme verisi üretildi; ayrıntılar §11'de.
Prompt 3'te Türkçe yüzey üretimi ve kapalı gramer doğrulaması eklendi (§12).
LLM çağrısı, eğitim veya model sonuç analizi yapılmadı. §5'teki sayılar gelecek
model deneylerinin taslak bütçesidir; veri envanteri veya başarı sonucu değildir.
Prompt 5 inference ve Prompt 6 analiz kodu hazırdır; eski bölümler tarihsel
tasarım kaydıdır. Parser/sistem mesajında §15, analizde §16 güncel sözleşmedir;
özellikle EVET/HAYIR ve B=2000 bu bölümlerdeki kullanıcı talimatıyla geçerlidir.

## 1. Koddan önce çözümlenmesi gereken belirsizlikler

| Kimlik | Belirsizlik / risk | Bu aşamanın kararı ve çalıştırma önkoşulu |
|---|---|---|
| M01 | Kaynakta yöntem ile yayımlanmış üretici aynı mı? | v1 HTML, §3–4 ve Ek B/H incelendi. Protocol P ve L karıştırılmaz. Arşiv kodu/verisi henüz incelenmedi; birebir replikasyon iddiası yok. |
| M02 | Büyüklük, çıkarım derinliği mi? | Hayır. Ana görevde `n` varlık sayısıdır. Uç nokta sorguları kısa yola açıktır. RQ3 uzaklık kolu ayrı, keşifseldir. |
| M03 | Türkçe dönüşüm gerçekten anlamı koruyor mu? | Her dönüşüm kanonik ilişkiye geri eşlenmeli; ayrıca iki bağımsız Türkçe yetkin değerlendirici bütün şablonları ve her hücreden en az 20 örneği (daha azsa tümünü) incelemeli. Anlaşmazlıklar gerekçeli uzlaşmayla giderilmeden ilgili şablon kullanılmaz. |
| M04 | Saf biçimbilim etkisi ayrıştırılabilir mi? | Çoğu yeniden ifade sözcük/sözdizimini de değiştirir. Şimdilik `tr_morph_joint` ortak müdahaledir. Saf biçimbilimsel nedensellik iddia edilmez; minimal çift onayı olmadan bu adla analiz yapılmaz. |
| M05 | Sıklık neyle ölçülecek? | Derlem, lisans, tarih, sürüm ve lemma/yüzey biçimi/çok sözcüklü ifade sayım kuralı seçilmedi. RQ4 keşifsel ve kapalıdır; sıklık değeri uydurulmaz. |
| M06 | Hangi modeller? | Konfigürasyonlardaki model kimlikleri yer tutucudur. Tam snapshot/revision, tokenizer, sağlayıcı, bütçe ve desteklenen decoding alanları çalıştırma öncesi sabitlenir. |
| M07 | Çıktı dili ve parser dil etkisi yaratır mı? | Ana karşılaştırmada her iki dil için ortak `YES/NO` çıktı etiketi önerilir; bu Türkçe yönergede de açıkça söylenir. `EVET/HAYIR` yerelleştirme kolu ancak ayrı duyarlılık deneyi olur. |
| M08 | Örneklem yeterli mi? | Pilot yalnızca dil/veri/format kontrolüdür. Temel örneklem 200/hücre değil, 200/n/seed toplamdır (dört sorgu hücresine 50'şer). Temel değerlendirme öncesi eşleştirilmiş etki ve küme bağımlılığını kullanan güç/duyarlılık hesabı gerekir. |
| M09 | Tekrar üretim bit düzeyinde mümkün mü? | Yerel veri üretimi hedef olarak deterministiktir. Uzak API cevabı seed ile bile aynı olmayabilir. Ham yanıt arşivinden yeniden puanlama ile yeni çağrıyla tekrar deney ayrılır. |
| M10 | Eğitim değerlendirmeye sızar mı? | Bütün dil, sorgu ve yüzey türevleri `ordering_id` grubuyla aynı bölmede tutulur. İleri eğitim seçeneği kapalıdır; test verisiyle şablon/model seçilmez. |
| M11 | Reasoning ile token bütçesi karışır mı? | Reasoning yönergesi ile sağlayıcının reasoning effort ayarı ayrı müdahalelerdir. Bütçe ve kesilme kaydedilir; eşit token bütçesi/eşit maliyet duyarlılık kolları önceden belirlenir. Gizli muhakemeye erişim varsayılmaz. |
| M12 | Önkayıt ve lisans durumu? | Bu dosya zaman damgalı taslaktır. Model çıktıları görülmeden sürümlenmiş protokol, veri/model lisansları, bütçe ve analiz aileleri dondurulmalıdır. Dış önkayıt henüz yoktur. |
| M13 | Sistem mesajının dili RQ1'i değiştirir mi? | Ana estimand ortak, sabit ve nötr iki dilli sistem mesajı altında kullanıcı problemi dilinin etkisidir. Tam şablon ve iki dilli mesajın sırası dilsel incelemede dondurulur; farklı rol desteği olan modeller sessizce birleştirilmez. |

Bu kayıtlar belirsizlikleri çözülmüş gibi göstermez. Pilot kapısı: M03, M06,
veri şeması, oracle ve parser testleri. Temel değerlendirme kapısı: bunlara ek
M08/M12 ve sabitlenmiş şablon/konfigürasyon/veri manifesti. RQ3-uzaklık, RQ4
ve fine-tuning kendi ek kapılarını geçmeden çalıştırılamaz.

## 2. Kaynak ile önerilen katkının sınırı

Referans: Jiang, J. ve Zhai, C. (2026), *Competence-Based Evaluation of Logical
Reasoning in Large Language Models*, [arXiv:2610.22894v1](https://arxiv.org/abs/2610.22894v1),
[tam metin](https://arxiv.org/html/2610.22894v1). Erişim: 2026-10-23.

**Kaynaktan alınan yöntem özeti:** CBE latent örnek, parametrik üretici, kesin
oracle, cevabı koruyan sunumlar ve tüm sunumlarda doğruluk koşulunu birleştirir
(§3). Sıralama örneklemi tam sıra, komşu karşılaştırmalar ve dengeli uç nokta
sorguları kullanır (§4.1). `n`, giriş/sunum yüküdür; çok adımlı çıkarımı tek
başına ölçmez. P protokolü yeniden yazılan konumları sıra boyunca rastgele
seçer; L protokolündeki konum örüntüsüyle aynı değildir. 2048 koşulunda sunum
dağılımı da değişir (Ek B). Tekrar kontrolü davranışsal bir karşılaştırmadır
(Ek H). Kaynak bu proje için sonuç veya Türkçe geçerlilik kanıtı sağlamaz.

**Bu projenin yeni önerileri:** eşleştirilmiş TR/EN sunumları; Türkçe dilbilimsel
varyantlar; ayrı boyut/uzaklık ızgarası; kaynaklandırılmış sıklık ve tokenizer
özellikleri; Türkçeye yönelik reasoning ve isteğe bağlı eğitim karşılaştırması.
Bu öneriler kaynak yöntemin birebir replikasyonu olarak adlandırılmaz. İlk
çekirdek sıralama görevidir; Horn-clause gibi ek görevler bu aşamanın dışındadır.

## 3. Latent yapı, cevap ve eşdeğerlik sözleşmesi

Hedef yeterlik: verilen kesin ve geçişli katı tam sıra altında ikili sıralama
sorgusunu doğru cevaplama. Üretici parametreleri `n`, sorgu uçları/mesafesi,
isim ataması, önerme permütasyonu ve sunum planıdır; doğruluk kuralı değişmez.

Latent kayıt `ordering_id`, `instance_id`, `generator_version`, `seed`, `n`,
eşsiz varlıklar, `n-1` komşu kenar, sorgu uçları/yönü ve Boolean `gold_label`
içerir. `split` henüz atanmadı; bütün sorgu/adlandırma/çeviri türevleri ileride
`ordering_id` ile aynı bölmeye gitmelidir. Bu kimlik soyut çekiliş ve sıra
permütasyonunu kapsar, sorgu ve isim etiketinden bağımsızdır. Pozisyon oracle'ı
metne/LLM'ye bakmaz; bağımsız ikinci oracle tüm öncül grafiğini çözerek denetler.

Dört sorgu hücresi: uçların sırası (`forward/reverse`) × sorgudaki ilişki
(`before/after`). Bu tam çapraz tasarım iki evet ve iki hayır hücresi verir.
Her `n × seed` bloğu dört hücreye eşit bölünür. Bir hücreye ait bir latent
sorgunun bütün dilleri ve varyantları aynı gold değerini taşır.

Sunum kaydı `presentation_id`, `instance_id`, `language`, `template_id/version`,
`variant`, isim eşlemesi, önerme sırası, ters çevrilen kenar indeksleri,
kanonik ilişkiye geri eşleme ve prompt SHA-256 alanlarını içerir. Bir kenarı
`A before B` yerine `B after A` biçiminde yazmak kenar yönünü değiştirmez.
Önerme sırası ve isim listesi latent sıradan bağımsız çekilir. Yeniden yazılan
konumlar uniform altkümeden seçilir; suffix kuralı kullanılmaz. Bu projenin
tasarım kimliği `tr_cbe_v0_1` olup kaynak Protocol P ile birebir eşitlik iddiası
taşımaz. İlk sunum taslağında rewrite sayısı uniform `1..n-1` önerilmişti.
Prompt 3'ün somut P2 sözleşmesi bütün `n-1` öncülü converse yapar; kısmi
uniform rewrite bu sürümde uygulanmadı. Bu iki plan sessizce eşitlenmez.

Türkçe tasarımın müdahale sözleşmesi (kaynak sunumlarının birebir kopyası değil):

| Varyant | Anchor'a göre değişen | Sabit tutulan |
|---|---|---|
| `canonical` | İlk bağımsız isim/önerme permütasyonu; bütün önermeler forward | Latent sıra, sorgu, gold |
| `converse` | Uniform seçilmiş kenarlarda argüman takası + converse ifade | İsim listesi ve önerme sırası, sorgu |
| `premise_shuffle` | Yalnız önerme sırası | İsim listesi, ifade yönü ve sözcükler, sorgu |
| `syntax` | Onaylı minimal cümle içi sözcük sırası değişimi | Sözcük/ekler, önerme ve isim sırası, sorgu |
| `morph_joint` | Onaylanacak biçimbilim + zorunlu ortak yeniden ifade | Latent/gold, önerme ve isim sırası, sorgu |

TR ve EN canonical/converse karşılıklarında isimler, permütasyonlar ve rewrite
konumları paylaşılır. `syntax` için aynı sözcük/ek dizimini yalnız yeniden
sıralama ile koruyamayan aday saf sözdizimi koluna kabul edilmez. Burada
converse kolu reshuffle içermez; bu, kaynak P/L bileşik dönüşümünden bilinçli
bir ayrımdır. Dönüşüm ailesi büyüdükçe protokol kimliği sürümlenir.

Eşdeğerlik kontrolleri: varlık/kenar kaybı yok; döngü/çelişki yok; katı sıra
varsayımı promptta açık; sorgu ve gold aynı; isim değişiminde birebir eşleme;
NFC Unicode, apostrof ve Türkçe I/İ/ı/i denetimi. Türkçe boş özne, gönderim
belirsizliği ve doğal olmayan devrik yapılar elle incelenir. Otomatik geri
eşleme, insan dilsel geçerliliğinin yerine geçmez. Ties, olumsuzluk, kip,
kanıtsallık ve nicemleyici değişiklikleri çekirdeğe alınmaz.

Örnek şablon adayları (henüz onaylı uyaran değildir): `A, B'den önce gelir.` /
`B, A'dan sonra gelir.`; `A, B'den daha uzundur.` / `A'nın boyu B'ninkinden
fazladır.` İkinci çift biçimbilim, sözdizimi ve sözvarlığını birlikte değiştirir.
Özel isim ekleri ses uyumuna göre üretilmeden morfoloji kolu açılmaz. Pilotun
ön/arka çekirdeğinde ekleme belirsizliğini azaltan aynı sabit alfanümerik
etiketler iki dilde de kullanılabilir; bu seçim daha doğal isimlere genellenmez.

## 4. Araştırma soruları, hipotezler ve değişkenler

Yön belirtilmeyen testler iki yönlüdür. Aşağıdaki hipotezler öneridir; sonuç
görülmeden doğrulayıcı/keşifsel statü ve karşılaştırma ailesi dondurulur.

| RQ | Hipotez | Bağımsız değişken / karşılaştırma | Bağımlı değişken | Durum |
|---|---|---|---|---|
| RQ1 | H1: Aynı latent örneklerde TR ve EN doğruluk veya sağlam doğruluğu farklıdır. | Dil; ortak isimler/gold/model/seed/çeviri eşlemesi | `A_all`, `R_all`, TR–EN eşleştirilmiş fark | Temel, iki yönlü |
| RQ2 | H2: Onaylı TR yüzey varyantı anchor'a göre sunuma bağlı hatayı artırır. | Sözdizimi, önerme sırası, converse ve ortak morfolojik kol ayrı ayrı | `R_all`, `U_valid`, invalid oranı | Temel; morfoloji onaya bağlı |
| RQ3 | H3a: `n` artınca doğruluk/sağlam doğruluk azalır. H3b: sabit `n` içinde mesafe artışı azalmayla ilişkilidir. | `n`; ayrı kolda `d` | `A_all`, `R_all`, boyut/mesafe profilleri | H3a temel; H3b keşifsel |
| RQ4 | H4: Sıklık, token parçalanması ve ifade yönü performansla ilişkilidir. | Log sıklık, model tokenizer ölçüleri, forward/converse | Örnek doğruluğu, `R_all`, invalid | Keşifsel; nedensellik yok |
| RQ5 | H5a: Reasoning yönergesi sağlam doğruluğu artırır. H5b: Eğitim artışı görülmeyen dil/şablon/büyüklüğe aktarılır. | Direct/reasoning; base/LoRA | Eşleştirilmiş `R_all` farkı, maliyet/token/kesilme | İleri; eğitim isteğe bağlı |

Kontroller: aynı latent bloklar, aynı isim havuzu, aynı sorgu dengesi, sabit
çıktı etiketi, mesaj rolleri, decoding ve çağrı planı. Kayıt edilecek kovaryatlar:
prompt token/karakter sayısı, sorgu konumu, rewrite oranı, biçim geçerliliği,
isim/şablon kimliği, model ve tokenizer revision. Dil ile token uzunluğu
doğal olarak bağlıdır; uzunluğu kontrol eden analiz farklı bir estimanddır,
ana dil etkisinin yerine geçirilmez. Morfoloji–tokenizasyon–sıklık birbirine
bağlı olabilir; az sayıda ilişki sözcüğü üzerinde aşırı regresyon kurulmaz.

## 5. Aşamalar ve deney matrisi taslağı

Her satır ayrı bloktur; tüm faktörlerin pahalı tam Kartezyen çarpımı önerilmez.
`instances_per_size_per_seed` bir sunum çifti sayısı değil latent sorgu sayısıdır.
Bir anchor birden fazla karşılaştırmada tekrar kullanılır; aynı prompt yeniden
çağrılmaz (tanımlı tekrar kontrolü hariç).

| Blok | `n` | Seed | Latent/n/seed | Sunum / model | Model yuvası | Kapsam |
|---|---|---|---:|---:|---:|---|
| Pilot | 4,8,16 | 11 | 16 (4/hücre) | 4 | 1 | TR/EN canonical+converse, tek ön/arka ilişkisi |
| Temel | 4,8,16,32,64,128 | 101,102,103 | 200 (50/hücre) | 7 | 2 | TR/EN canonical+converse; TR shuffle, syntax, morph_joint |
| İleri RQ3 | 8,16,32,64 | 201,202,203 | 200 / geçerli `(n,d)` | 4 | 2 | `d∈{1,2,4,8}`, yalnız `d≤n-3`; iç uçlu sorgular |
| İleri RQ4 | Temel gridin onaylı altkümesi | ayrı sabit seed planı | Güç/bütçe sonrası | ayrı ilişki blokları | tokenizer başına | Sıklık/tokenizasyon/yön |
| İleri RQ5 | 4,8,16,32,64,128 | 201,202,203 | 200 | 4 × 2 prompting | 2 | Aynı latent örneklerde direct/reasoning |
| Opsiyonel eğitim | Eğitim ≤32; test 64,128 + ayrı şablon/ilişki | ayrı train/dev/test seedleri | Henüz belirlenmedi | aynı test promptları | bir açık ağırlık revision | Base ve LoRA; testten seçim yok |

Temel satırdaki 7 sunum bir ilişki bloğu içindir. Yeni ilişki blokları ek bütçe
gerektirir; 31 ilişkiyi otomatik çoğaltma yoktur. Varsayılan ön/arka çekirdeğinde
onaylanamayan `morph_joint` varsa 7-sunum tasarımının tamamı bloke edilir;
sessizce 6'ya düşürülmez. Onaylı yeni protokol sürümü ve yeni bütçe gerekir.

Planlanan çağrı bütçesi (sonuç değildir): pilot `3×16×4×1=192`; temel
`6×3×200×7×2=50.400`. Bütün sunumlar için iki toplam gözlemli tekrar kontrolü:
pilot latentlerin %25'i için +48, toplam **240**; temel %10 için +5.040,
toplam **55.440** çağrı. Tekrar altkümesi her `n×seed×sorgu hücresi` içinde
dengeli ve model çıktısından önce seçilir. Konfigürasyon komutu bu iki bütçeyi
hesaplar. Hatalı ulaşım tekrarları bu sayıya dahil değildir; ayrı loglanır.

İleri konfigürasyon yalnız RQ5 prompting taslağını kodlar; RQ3-uzaklık, RQ4 ve
eğitim için çalıştırılabilir plan henüz yoktur. Yeni model eklemek bir yuva
seçimidir; mimari sağlayıcıya bağlanmaz. Fiyatlar/model seçimi yapılmadığından
parasal maliyet veya çalışma süresi tahmini verilmez.
İleri RQ5 taslağı `6×3×200×4×2 prompting×2 model=57.600` ilk çağrı ve
%10 altküme için +5.760 tekrar, toplam **63.360** planlı çağrı içerir.
Temel aşamadaki 64 ve ileri prompting aşamasındaki 1024 çıktı-token sınırı
birbirine karşı etki testi olarak kullanılmaz; ileri direct/reasoning kolları
kendi içinde aynı 1024 sınırını paylaşır. Bunlar sağlayıcı seçimi öncesi öneri
ayarlarıdır; kaynak çalışmanın varsayılan sampling ayarlarının kopyası değildir.

## 6. RQ3 uzaklık kolunun operasyonel tanımı

Çekirdek uç nokta sorgusunda `d=n-1` cebirsel mesafedir; buna çıkarım derinliği
denmez. Yeni kolda `q_i,q_j` tam sıranın iki iç düğümüdür,
`d=abs(i-j)` ve `1≤d≤n-3`. Aynı `n` içinde birden fazla `d`, aynı `d` için
birden fazla `n` bulunur. Sorgu çiftleri iç konumlar arasında uniform
örneklenir; 200 örnekte bütün konumların tam sayısal dengesi garanti edilmez.
Tüm görev yine `n-1` kenarlı bir zincirdir; gerçek düğüm derecesi, yerel yön
örüntüsü ve token yükü kaydedilir. Gold hesabı bağımsız graph reachability
ile doğrulanır. İç sorgular endpoint kısa yolunu azaltır, tüm kısa yolları
ortadan kaldırdığını kanıtlamaz. Bu kol doğrulanmış oracle/üretici ve ayrı
güç hesabı olmadan temel sonuçlara karıştırılmaz.

## 7. Ölçütler, paydalar ve hata politikası

`N`: planlı tüm promptlar; `N_pair`: karşılaştırmanın tüm latent çiftleri;
`V`: iki sunumu da parse edilmiş çift sayısı. Ana güvenilirlik ölçütü
`R_all = her iki cevap doğru çift sayısı / N_pair`; invalid, cevapsız ve
kesilmiş çıktılar bu paydadan çıkarılmaz. `A_all = doğru cevap / N`.
`R_all` birincil ölçüt seçimi bu projenin yeni analiz kararıdır.
Ulaşım hatası bütçe sınırında çözülemezse eksiklik olarak ayrıca işaretlenir,
ana intention-to-evaluate analizinde yanlış sayılır; gözlenen cevap altkümesi
duyarlılık analizi ayrı raporlanır. Tüm koşullarda eksik/invalid/kesilme sayılır.

Kaynakla karşılaştırılabilir tanısal ayrışım yalnız ortak valid-pair desteğinde:
`R_valid=both_correct/V`, `D_valid=both_wrong/V`, `U_valid=exactly_one_correct/V`.
Boolean ve aynı gold varsayımıyla `R+D+U=1`, `A_mean_valid=R+U/2`,
`I_valid=R+D`. `V=0` ise bu oranlar `null` olur, sıfır yapılmaz.
Yüksek agreement ortak hataları da içerebilir; tek başına yeterlik değildir.
`R_all` ve `R_valid` açıkça farklı adlarla ve `V/N_pair` coverage ile verilir.
Birden fazla varyant için ayrıca `AllCorrect_K` hesaplanabilir; çiftli R/D/U
doğrudan K sunuma genellenmez. Dil eşleştirmesinde etiketler Boolean'a çevrilir.

İkincil bias düzeltmesi: `q` ortak valid desteğinde iki sunumun birleşik
YES oranı, `I_e=q²+(1-q)²`, `kappa=(I-I_e)/(1-I_e)`. `q=0/1` ise `null`.
Bu pooled-bias referansı Cohen kappa veya nedensel bağımsızlık kanıtı olarak
sunulmaz. Büyüklük ortalamalarında geçerli seviye sayısı ve ağırlık belirtilir.

Gelecekteki parser kuralı: Unicode NFC + dış boşluk temizleme + harf büyüklüğü
normalizasyonu sonrası yalnız tam `YES` veya `NO`. Noktalama, açıklama,
iki etiket ve başka dilde cevap invalid sayılır. Reasoning kolu aynı son-cevap
sözleşmesini kullanır; iç reasoning ayrı alan ise saklanması sağlayıcının
izin verdiği metaveriyle sınırlıdır. Parser sürümü sabitlenmeden çağrı yapılmaz;
cevap görüldükten sonra gevşetilmez.
Bu strict parser Turkish-CBE için yeni bir operasyonel seçimdir; kaynak
parserın birebir replikasyonu olduğu iddia edilmez.

Tekrar kontrolü: seçilen latentlerin bütün promptları aynı ayarlarla ikinci
kez, yeni istek olarak gönderilir. Cache bypass ve request id kaydedilir.
Kaynak Ek H özgün sunumu tekrarlar; bütün sunumları tekrar etme tasarımı
bu projenin yeni uzantısıdır.
Tekrar agreement/doğruluk ile yüzey dönüşümü agreement'ı ayrı tahminlerdir;
birini diğerinden çıkararak saf dil etkisi elde edildiği iddia edilmez.
Tekrarlar yeni bağımsız latent örnek sayılmaz; ana analiz ilk gözlemi kullanır.

## 8. İstatistik ve raporlama planı

Analiz birimi bağımsız latent grup (`ordering_id`); sunumlar, diller, sorgular,
modeller ve tekrarlar bu grubun bağımlı gözlemleridir. Gelecekte aynı sıradan
birden fazla sorgu üretilirse bootstrap grubu değişmez. Seedler blok olarak
tutulur; yalnız üç seed üzerinden güvenilir seed-varyansı iddiası yapılmaz.

Ana karşılaştırmalar: dil farkı için model başına `R_all(TR)-R_all(EN)`;
RQ2 için her onaylı dönüşümün canonical eşleşmesi; büyüklük için her `n`
profilinin farkları; RQ5 için aynı latentlerde reasoning-direct. Yüzde puan
etki ve %95 eşleştirilmiş küme bootstrap aralığı raporlanır (`B=5000`, sabit
analiz seedi, `n×üretim seed` tabakaları). Test/pilot karışmaz. Model başına
tek ana RQ1 karşılaştırması; RQ2'de model×dönüşüm; RQ5'te model×müdahale ayrı
Holm aileleridir. Keşifsel RQ4 ve uzaklık karşılaştırmaları açık etiketlenir.
İki modelin iki RQ1 testi birlikte tek Holm ailesini oluşturur; model başına
ayrı aile kurulmaz.
RQ1'de `R_all(TR)` yalnız `(tr canonical, tr converse)`, `R_all(EN)` yalnız
`(en canonical, en converse)` çiftini kapsar. Türkçenin ek üç varyantı bu
birincil dil karşılaştırmasına eklenmez; iki dilde K eşittir. RQ2'de canonical
ile her bir TR varyantı ayrı eşleştirilir.

McNemar, yalnız aynı latentlerin ikili doğruluğuna uygulanacak ikincil testtir;
invalid yanlış sayılır, discordant sayısı küçükse exact binomial tercih edilir.
Sunum başına gözlemleri bağımsız sayan p-değeri kullanılmaz. `n,d` etkileri
için yeterli bağımsız küme varsa karma etkili lojistik model düşünülebilir;
önkayıt yapılmadan doğrulayıcı model olarak eklenmez. Etkileşimler ve çoklu
ilişki analizleri örneklem yeterli değilse betimsel bırakılır.

Tek bir başarı eşiğiyle evrensel yeterlik ilan edilmez. Test edilen dağılım,
boyut, şablon ve model revision kapsamı yazılır. Türkçeye özgü nedensel
mekanizma, iç muhakeme arızası veya sıklık/tokenizasyon nedenselliği davranış
puanlarından tek başına çıkarılamaz. Sıfır hata evrensel yeterlik kanıtı değildir.

## 9. Veri doğrulama ve aşama kapıları

Aşama 1'de konfigürasyon/provenance testleri vardı. Aşama 2'de üretici, iki
oracle, kayıt/dataset doğrulaması ve veri CLI testleri eklendi. Aşağıdaki
1–2 latent veri için uygulandı. Prompt 3'te 3'ün otomatik kontrolleri eklendi,
insan onayı bekleniyor; 4–6 model değerlendirmesi öncesinde gereklidir:

1. Üretici: seed determinismi, dört hücre dengesi, benzersiz kimlik, tam sıra,
   `n-1` komşu kenar, bağlılık, cycle yokluğu; küçük `n` için exhaustive oracle.
2. Oracle: sıra indeksine karşı bağımsız graph yöntemi; known-answer fixtures;
   doğru/yanlış sorgular ve tüm yönler. LLM doğrulayıcı olarak kullanılmaz.
3. Renderer: geri eşleme, gold korunumu, isim eki/ses uyumu/Unicode, bilinen
   bozuk dönüşümleri reddetme; bütün şablonlarda çift değerlendirici onayı.
4. Bölmeler: `ordering_id`/prompt hash sızıntısı, tüm türevlerin aynı splitte
   olması. Bütün tam sıralar graf olarak izomorfiktir; izomorfizmle held-out
   mantık yapısı iddia edilmez. İsim/atama/şablon ve boyut aktarımı ayrı yazılır.
5. Runner/parser: ağsız fixture testleri, invalid/truncation/missing ve tekrar
   kayıtları; gerçek sonuçlarla sentetik fixture klasörlerinin ayrılması.
6. Metrics: elle hesaplanmış R/D/U örnekleri, boş destek, ortak yanlış,
   sabit çıktı kappa, eksik çift ve bootstrap küme bağımlılığı.

Pilot bittiğinde şablon, parser ve örneklem revizyonu yeni sürüm alır. Pilot
verisi doğrulayıcı test setine eklenmez. Dev/test ayrımı seed tek başına değil,
grup ve hash kontrolleriyle doğrulanır. Veri ve sonuç doğrulaması başarısızsa
pipeline durur; eksik alanı varsayılan değerle sessizce tamamlamaz.

## 10. İleri müdahalelerin sınırları

Reasoning: direct ve kısa bir akıl yürütme yönergesi ayrı prompt koşullarıdır;
modelin yerleşik effort özelliği ayrıca adlandırılır. Mesaj dili, son cevap
etiketi ve latent örnekler sabittir; input/output/reasoning tokenları ile
finish_reason kaydedilir. Uzun yanıt kesilmesi yetenek kaybıyla özdeş sayılmaz.

Fine-tuning: şimdilik `enabled=false`. Model lisansı/donanım/bütçe, LoRA
tarifi, optimizer/scheduler, veri sürümü, train/dev/test ayrımı, en az üç
eğitim seedi ve checkpoint seçimi önceden yazılır. Dev ile seçilen checkpoint
tek sefer dondurulmuş testte değerlendirilir. Base, eğitim görmüş ve daha uzun
boyut/held-out ilişki/şablon kolları ayrılır. Train ve test prompt benzerliği
denetlenir; bütün mantık aileleri farklıymış gibi sunulmaz. GRPO, mekanistik
probe ve yeni görev aileleri bu aşamada uygulanmaz.

## 11. Prompt 2: uygulanan latent veri protokolü

`src/data_generation/`, kullanıcı tarafından istenen modül sınırıdır; Aşama 1'de
önerilen `logic.py` yerine bu paket kullanıldı. Model deneyleri için üç TOML
değiştirilmedi. Veri üretim parametreleri dataset manifestinde ayrıca saklanır.

Seed 11, dokuz n=4..1024, her n'de 200 temel sıra; her sorgu bloğunda yön ×
gold hücresinde 50. Endpoint ve strictly-internal kollar ayrı klasörlerde;
mesafe kolunda d∈{1,2,4,8,16,32,64,128,256,512} için yalnız d≤n−3 kullanılır.
53 geçerli `(n,d)` bloğu, 9 endpoint ve 9 internal bloğu: 71 dosya, 14.200
kayıt, 1.800 ortak temel sıralama. Bunlar gelecekteki performans analizinde
bağımsız 14.200 örnek sayılmaz; kollar ve d'ler kontrolsüz birleştirilmez.

İsim grupları dört hücrelik bloklarla atanır; her n'de 68/68/64 dağılımının
küçük grubu döner. Böylece her grup toplam 600 temel sıra ve kendi içinde
tam cevap/yön dengesi alır. Adlandırmalar aynı sıranın üç eşlenik sunumu
değildir. Grup karşılaştırmasında token uzunluğu ve bilinirlik karıştırıcıdır.
Türkçe tam ad havuzu 48 ad×32 soyad; sentetik CVCV ve sembolik havuzlar
sürümlenmiştir. Dilsel geçerlilik veya derlem sıklığı onayı verilmedi.

RNG akışları master seed, generator_version ve açık namespace üzerinden
SHA-256 ile ayrılır. Sıra/isim/öncül permütasyonu/sorgu/dosya sırası birbirini
etkilemez; süreç hash() veya global random durumu kullanılmaz. Oracle'lar
bağımsızdır; üretici sıralı indeks oracle'ını, doğrulama ayrıca tam graph
oracle'ını kullanır. Record ve global blok kontrolleri başarısızsa üretim
durur; sadece o çağrının oluşturduğu kısmi veri geri alınır. Mevcut veri silinmez.

Kayıt sözleşmesi, isim kuralları ve gerçek dağılımlar [data/README.md](data/README.md),
hashler/parametreler [dataset_manifest.json](dataset_manifest.json) içindedir.
Bu veri geliştirme içindir; dış önkayıt/test kilidi veya deney sonucu değildir.
Türkçe renderer, insan şablon onayı, model runner ve analiz bu promptta yapılmadı.

## 12. Prompt 3: renderer öncesi yöntem kararları

Bu bölüm Türkçe renderer uygulamasından önce eklendi. Dokuz ilişki, aynı
latent before/after eksenine bağlanır: ön/arka, sağ/sol, üst/alt, geliş,
tamamlanma, boy, hız, ağırlık ve sıra. Eksen yönü sabittir; bütün değerler
farklıdır. Üst/alt yalnız düşey konumdur, fiziksel temas değildir. Tamamlanma
ilişkisinde kişi adları görev etiketi olarak açıklanır; kişilerin kendisinin
"tamamlanması" gibi seçimsel olarak bozuk cümle üretilmez.

- P0 temel; P1 yalnız öncül/isim listeleri; P2 bütün öncüllerde argüman takası
  ve converse sözcüğü; P3 cümle içi tamamlayıcı-yüklem-özne sırası. P3 devrik
  cümlesi doğruluk koşulunu korusa da odak/pragmatik açıdan nötr değildir;
  doğallığı insan incelemesinde ayrıca değerlendirilir.
- P4 iyelikli özellik adlarına geçer (konumu, geliş zamanı, boyu, hızı,
  ağırlığı, sırası); bu morphology+syntax+lexical ortak müdahaledir, saf
  biçimbilimsel nedensellik iddiası taşımaz. P5 P1+P2+P4 birleşimidir.
- P6 varsayılan kapalıdır. Farklı a,b ve strict total order altında
  `not after(a,b) iff before(a,b)`; eşitlik ve karşılaştırılamazlık yoktur.
  Bu ispat yalnız tanımlı atomik ilişkiler ve bütün olayların gerçekleştiği
  bağlamda geçerlidir. Genel serbest olumsuzlama üretilmez.
- Genitive/ablative ekleri son okunuşun ünlüsü ve son ünsüzüne göre seçilir.
  Özel adın yazımı korunur, ASCII kesme işareti standarttır. Sembolik Edddd
  etiketlerinde harf ve rakamlar tek tek okunur; bu sözleşme promptta görünür.
  Genel Türkçe morfolojik çözümleme iddiası yoktur; kapalı havuz/şablonlar
  dışındaki belirsiz okunuşlar reddedilir veya açık okunuş gerektirir.
- Gerçek yüzey cümleleri kapalı gramerle normal forma geri çözülecek; sadece
  metadata veya renderer'ın kendi işaretleri doğruluk kanıtı sayılmayacak.
  Sorgu bütün P varyantlarında sabit kalacak ve bağımsız graph oracle çalışacak.
- En az 100 tam promptluk insan kontrol tablosu hazırlanacak; otomatik
  eşdeğerlik ve morfoloji sonucu ile insan dilsel kararı ayrı alanlardır.
  İnsan onayı uydurulmayacak; değerlendirme alanları boş/pending kalacak.

Yazım dayanakları: [TDK kesme işareti](https://tdk.gov.tr/icerik/yazim-kurallari/noktalama-isaretleri-aciklamalar/),
[büyük ünlü uyumu](https://tdk.gov.tr/icerik/yazim-kurallari/buyuk-unlu-uyumu/),
[küçük ünlü uyumu](https://tdk.gov.tr/icerik/yazim-kurallari/kucuk-unlu-uyumu/).
Erişim: 2026-10-23. Bu kaynaklar proje şablonlarının insan onayı değildir.

Uygulama sonucu: dokuz ilişki ve P0–P6 desteklenir. 36 farklı kaynak sıradan
252 tam prompt çifti [kontrol raporuna](data/validation/README.md) yazıldı.
Her ilişki için dört query-direction × gold hücresi ve yedi varyant vardır.
Kaynak seçimleri insan kontrolü içindir; ilişkiler arası model başarısı veri
kümesi değildir. İki insan değerlendiricinin alanları boş ve pending'dir.
P6 sadece bu raporda açıkça etkinleştirilmiştir; API varsayılanı kapalıdır.
Kod/CSV/kaynak hashleri rapor manifestinde yer alır. İlk üç deney TOML'ü
değiştirilmedi: P0–P6 için henüz inference eşleme planı yoktur; eski
premise_shuffle ile P1'in iki listeyi değiştirmesi aynı koşul değildir.

Bu dosyada veya repoda hiçbir tablo model performansı tablosu değildir.
Deneylerin yürütülebilirliği için REPRODUCIBILITY.md ve PROGRESS.md içindeki
tamamlanmamış kapılar esas alınır.

## 13. Tarihsel v0.4.0: eşleştirilmiş veri setinden önce alınan kararlar

Bu bölüm Prompt 4 kodundan önce yazılmıştır. Referansın [Appendix B](https://arxiv.org/html/2610.22894v1#A2)
protokol P'sine yakın kol, endpoint sorgusu, dengeli dört hücre, rastgele
öncül/isim sırası ve konumu düzgün örneklenen kısmi converse dönüşümü kullanır.
Yeniden yazılan kenar sayısı m, max(1,floor(0.2(n-1)))..n-1 aralığından
seçilir; iki liste eşdeğer sunumda yeniden karıştırılır. Bu, kaynak çalışmanın
veri/model sonuçlarının birebir replikasyonu değildir. Açık Türkçe bağlam,
isim kontrolleri ve farklı seedler yeni tasarım kararlarıdır. Kaynağın L kolu
(son m kenarı dönüştürme) uygulanmayacaktır. Eski P2 bütün kenarları çevirir;
yeni `equivalent` koşuluyla aynı dönüşüm sayılmaz.

- Pilot: n=4..128 güçleri, n başına 50 latent sıra; test: n=4..1024,
  n başına 200. Pilot 13/12/12/13 veya 12/13/13/12 hücreleriyle iki marjinde
  de tam dengelidir. Development ve training 50/n, n=4..1024 olarak
  seçilir; bu iki kümenin büyüklüğü kullanıcı tarafından belirlenmemiştir.
- Seedler training=401, development=402, pilot=403, test=404. Önceden
  yayımlanmış seed=11 verisi yalnız geliştirme kaynağıdır; frozen test
  olarak yeniden etiketlenmez. Bütün sorgu, dil, isim ve sunum türevlerinin
  split grubu `ordering_id`dir (`latent_instance_id` olarak yayımlanır).
- Ana kontrollü kaynak ad havuzu Türkçe tam adlardır. `native_shared` iki
  dilde aynı Unicode adı; `en_transliterated` yalnız İngilizcede ASCII ad;
  `symbolic_shared` iki dilde aynı Edddd kimliğini kullanır. Bu son kimlikler
  2048 elemanlık havuzdan sıradan bağımsız örneklenir; her örnekte sıfırdan
  başlayan sıra etiketi kullanılmaz. Transliterasyon çakışmaları reddedilir.
  Ad eşlemesi görünür metinden ayrı kimlik haritasıyla korunur.
- A–D ana kolu front/behind, endpoint, üç isim koşuludur. F aynı sıra,
  isim/öncül sırası ve converse maskesi üzerinde internal ve geçerli 2'nin
  kuvveti uzaklıklarını yalnız native_shared koşulunda karşılaştırır.
  E endpoint/native_shared üzerinde P1..P5'i ayrı etiketler. P4/P5 çoklu
  değişimdir; saf morphology etkisi değildir. Dokuz ilişki sözlüğünün tamamı
  pilot endpoint/native_shared uzantısında bulunur. Tam Kartezyen çaprazlama
  yapılmaması açıkça hücre matrisine yazılır; üretilmeyen hücreler veri değildir.
- Ayrı JSONL prompt bankası ve karşılaştırma hücresi dosyaları, aynı uzun
  promptun her karşılaştırmada kopyalanmasını önler. Kaynak latent kayıtları
  da split içinde bir kez saklanır; bütün başvurular ve iki bağımsız oracle
  doğrulanır. İsim/dil bazında aynı sıralama+hedef parmak izleri ayrıca kontrol
  edilir. Bütün zincirler izomorfiktir; yeni mantık kuralına genelleme iddiası
  kurulmaz. Kimlik ayrımı, modelin eğitim öncesi kontaminasyonunu kanıtlamaz.
- Model/tokenizer revision henüz yoktur. Unicode referans tokenizer sayısı,
  boşluk tokenları, karakter ve UTF-8 uzunluğu kaydedilir. Bunlar modelin BPE
  token sayısı değildir; `model_token_count=null` ve durum açıkça tutulur.
  Gerçek model tokenizer ölçümü model seçimiyle ayrıca dondurulmalıdır.
- Frozen test dosyaları exclusive oluşturulur ve SHA-256 ile dondurulur;
  değiştirme yerine yeni veri sürümü gerekir. Bu yapısal donma insan dilsel
  onayı değildir. İnsan denetimi beklenir; model çalıştırılmaz ve sonuç yoktur.

Liste karıştırma identity üretirse bir konum rotasyon uygulanır; bu, değişimin
gerçekleşmesini garanti eder fakat permütasyonlar tam uniform değildir.
Özellikle küçük n için kaynak shuffle protokolünden bu ayrım korunur.

## 14. Prompt 4 ek tamamlama: gerçek tokenizasyon ve tam faktör matrisi

2026-10-24: Kullanıcı, önceki iki kapsam sınırının tamamlanmasını istedi.
Bu kararlar genişletme kodundan ve yeni veri üretiminden önce kayıt altına
alınmıştır. Önceki v0.4.0 veri dosyaları ve Python kaynakları korunacaktır.
Yeni `complete_grid` paketi ve v0.4.1 faktör matrisi eklenir.

- Üç isim kontrolü × dokuz ilişki × her geçerli endpoint/internal/distance
  sorgusu çaprazlanır. Dil/sunum koşulları EN canonical/equivalent ve TR
  canonical/equivalent/P1..P5 olmak üzere dokuzdur: sorgu başına 243 prompt.
  P3/P4/P5 Türkçeye özgü müdahaleler olduğu için İngilizce karşılıkları icat
  edilmez. İsteğe bağlı P6 bu ana matrisin dışında kalır.
- Kaynağa yakın yapısal temel kol seçim kuralı: A–D ve
  `relation_id=spatial_front`, `query_type=endpoint`. Diğer ilişkiler,
  sorgu kolları ve Türkçe ablation hücreleri uzantıdır. Bu ayrım sonuç
  analizinde açık filtreyle uygulanır; toplu tam matris ortalaması değildir.
- Dört split ve aynı latent kaynaklar korunur; testte n başına 200 sıra,
  diğerlerinde önceki 50/n dengesi değişmez. Yeni ilişkiler bütün test n
  değerlerinde, üç isim kontrolünde ve bütün sorgu kollarında bulunur.
- Genişletilmiş E, bütün sorgu/isim/ilişki hücrelerinde TR canonical ile
  P1..P5'i karşılaştırır. Bu v0.4.1 sözleşmesi, eski yalnız endpoint/native
  E sözleşmesinden ayrı sürümlenir. F dört canonical/equivalent dil tabakasında
  sorgu karmaşıklığı karşılaştırması olarak kalır; eşdeğerlik çifti değildir.
- 5.637.600 prompt ve 7.819.200 karşılaştırmada uzun öncül metinlerini
  tekrar saklamak yerine kaynak+tasarım+ölçüm tabloları kullanılır. Her prompt
  kaydı tam metne deterministik açılabilir ve JSONL olarak dışa aktarılabilir.
  SQLite bir saklama biçimidir; üretilmeyen hücreyi var sayma yöntemi değildir.
  Her sorgu için bütün 243 hücrenin hash ve gerçek tokenizer sayıları üretilir.
- Gerçek düz metin BPE ölçümü tiktoken 0.12.0 ile cl100k_base ve o200k_base
  için yapılır; tokenizer eşleme referansları gpt-4-turbo-2024-04-09 ve
  gpt-4o-2024-08-06'dır. Bu modeller inference için çağrılmaz veya seçilmiş
  deney modelleri olarak raporlanmaz. Vocabularies, regex kuralları, özel
  tokenlar, paket sürümü ve dosya hashleri yerelde sabitlenir. Üretim ve
  doğrulama ilk indirmeden sonra ağsız çalışır.
- Ölçülenler tam görünür promptun gerçek BPE tokenlarıdır. Sağlayıcının rol,
  sistem mesajı, araç veya chat şablonu ek yükü dahil değildir. Yerel düz metin
  sayımının bu sınırı [resmî token sayımı belgesine](https://developers.openai.com/api/docs/guides/token-counting)
  uygundur; [encoding eşlemesi](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken)
  resmî kaynak ve sabit paket koduyla kontrol edilir.
- Tekrarlanan öncül bloğu için prefix/suffix sayım optimizasyonu yalnız regex
  ön-token sınırının birleşimde değişmediği ispatlanan ayırmada kullanılır;
  güvenli olmayan birleşimler tam metinle encode edilir. Gerçek tam encode
  fixture'ları ve en büyük n için karşılaştırmalı testler zorunludur.
- Bütün yüzey öncülleri bağımsız kapalı gramerle okunur, kaynak normal forma
  eşitlenir. Aynı komponenti paylaşan sorular için bu ispat yeniden kullanılır;
  her soru hedefi ve oracle gold'u ayrıca doğrulanır. Grup/sorgu/komponent
  yabancı anahtarları farklı latent sıraların sessizce eşleşmesini engeller.
- İnsan onayı, model inference, eğitim ve sonraki prompt bu tamamlama işinin
  dışında kalır. Yeni ölçümler, önceki null alanlarını gizlice değiştirmek
  yerine ek sürümlü ölçüm bankasında ve JSONL dışa aktarımında sunulur.


## 15. Prompt 5 — inference öncesi yöntem ve yürütme sözleşmesi

2026-10-24: Kullanıcı bu aşamada yalnız altyapı ve dry-run tamamlanmasını,
gerçek pilotun model seçimine bırakılmasını istedi. Ücretli API çağrısı,
model ağırlığı indirmesi, gerçek pilot/full değerlendirme veya eğitim yapılmaz.

- Frozen kaynak 0.4.1 veri ve önceki 31 Python dosyası korunur. İnference
  ayrı `inference` paketidir. Eski token sayıları yeni sistem/kullanıcı
  mesajlarının uzunluğu diye sunulmaz.
- Tam bağlam, varlık listesi, bütün öncüller ve soru hedefi korunur. Yalnız
  eski cevap talimatı kullanıcının istediği EVET/HAYIR ve YES/NO talimatına
  çevrilir; iki dilde eşdeğer sistem mesajı eklenir. Bu yeni prompting
  sürümüdür; kaynak prompt ve yeni mesaj hashleri ayrı dondurulur.
- Kesin parser yalnız dış boşlukları atılmış büyük harfli tek cevabı kabul
  eder. Noktalama, açıklama, JSON, çelişkili cevap, yanlış dil, refusal ve
  kesilmiş çıktı invalid olarak korunur. Ana paydadan çıkarılmaz; invalid
  yanıtı düzeltmek veya iyi cevap seçmek için tekrar çağrı yapılmaz.
- Aynı latent gruplar bütün seçilen modellere atanır; dil/canonical/equivalent
  grubu yarıda kesilmez. Endpoint/internal/distance kolları ayrı kalır.
  Küçük pilot varsayılanı n=4/8, boyut başına dört dengeli grup ve dört
  EN/TR canonical/equivalent sunumudur. Asıl pilot 300 ve full 1800 kaynak
  sıra bankası korunur; seçim ve toplam çağrı sayısı dry-run'da açıkça yazılır.
- Model, tokenizer ve altyapı adı/revision, sampling, seed desteği, reasoning,
  context/output sınırları ve bağımlılık sürümleri kaydedilir. Geçersiz veya
  kullanılamayan model sessizce değiştirilmez; alias yerine açık snapshot
  veya immutable commit gerekir. Desteklenmeyen ayar sessizce atılmaz.
- Yapılandırmadaki modeller örnektir ve kapalıdır. Closed API parametre sayısı
  bilinmiyorsa null; fiyat bilinmiyorsa null yazılır. Model kartındaki bağlam
  sınırı dağıtımın gerçekten o kapasiteyi sunduğuna kanıt değildir.
- Dry-run dış servis veya ağırlık yüklemez. Yerel mevcut tokenizer ile gerçek
  düz metin sayısı ya da açıkça etiketlenmiş tahmin kullanılır. Chat şablonu
  bilinmiyorsa chat dahil kesin sayı iddia edilmez. Maliyet ancak tarihli
  yapılandırılmış fiyat ve bilinen kullanım/tahminle hesaplanır.
- Gerçek yürütmede context taşması keserek düzeltilmez. Model başına sessiz
  örnek atma eşleştirmeyi bozar; hata/blokaj görünür kaydedilir. Gerçek
  sağlayıcı kullanım değerleri tahmini sayılardan ayrı tutulur.
- Her ağ denemesi benzersiz kimlik ve exclusive ham request/response/error
  kaydı alır. Ham kayıtlar değişmez; checkpoint ve normalizasyon türetilmiş
  katmandır. Resume tamamlanmış çağrıyı yeniden göndermez.
- Ağ zaman aşımı veya çağrı sonrası crash durumunda uzaktaki işlemin
  gerçekleşip gerçekleşmediği bilinemeyebilir. Böyle durumlar ambiguous
  olarak durur; tam exactly-once sağlayıcı garantisi iddia edilmez ve
  otomatik tekrar yapılmaz. Yalnız kesin reddedilmiş geçici hatalarda
  sınırlı exponential backoff uygulanır; bütün denemeler korunur.
- Cache isteğe bağlı ve varsayılan kapalıdır. Cache cevabı yeni bağımsız
  model yanıtı sayılmaz; kaynak çağrı kimliği, model/prompt/ayar hashleri
  kaydedilir. Tekrar deneyi cache ile çoğaltılmaz.
- Full evaluation, tamamlanmış gerçek pilotun bütünlüğüne ve kullanıcı
  onay kaydına bağlıdır. Dry-run veya fixture testleri pilot onayı değildir.
  Önceki insan dilsel review=pending ve confirmatory eligibility=false
  durumları devam eder; bu aşama insan onayı üretmez.


### Uygulamada netleştirilen Prompt 5 kararları

- Llama gibi tarih kullanan chat template'lerde açık ISO tarih seçilir ve
  tokenizer/count/generation yollarında aynı date_string kullanılır. Tarih
  parametresi sunmayan örtük saat çağrısı reddedilir; gerçek rendered HF
  input metni ve token ID'leri hashleriyle ham runtime kaydına alınır.
- Default dry-run bütün altı örneği yalnız planlar: pilot 192, temel full 43.200
  çağrı. Önceki 5.637.600 promptluk tüm matrisi varsayılan olarak inference'a
  açmaz. Daha geniş seçim CLI ile açıkça yapılır ve bütçe yeniden hesaplanır.
- Çevrimdışı bütçe: frozen kaynak BPE toplamı +64 token/çağrı heuristiği.
  Bu chat dahil kesin ölçüm/üst sınır değildir. Gerçek yürütme input_count
  ölçümünü ve context/output rezervini zorunlu tutar; hiçbir metin kesilmez.
- GPT-4o tarihsel örneğinin fiyatı resmî karttan 2026-10-24 tarihiyle seçildi.
  Deprecated örnek hesapta mevcut kabul edilmez; erişim hatası başka modelle
  düzeltilmez. Diğer modellerin kullanım/donanım fiyatı null kalır.
- Kod, cohort, budget veya environment elle re-hash edilse bile yürütmede
  plan sıfırdan hesaplanıp karşılaştırılır. Kurulu inference Python kaynakları
  proje kaynaklarından farklıysa yürütme/planlama reddedilir.
- Fixture koşuları açıkça test_fixture etiketlidir ve gerçek sonuç veya pilot
  onayı üretemez; cache namespace'leri live/fixture arasında ayrıdır.
- POSIX flock/checkpoint Linux/macOS içindir. Windows kilit altyapısı ve
  gerçek GPU/API servisleri bu teslimde test edilmemiştir. Exactly-once uzak
  servis garantisi yoktur; uncertainty açıkça durdurulur.

## 16. Prompt 6 — uygulama öncesi analiz kararları (2026-10-24)

Bu bölüm yeni analiz kodundan önce kaydedildi. Kaynağın §3.2–3.3, Appendix
A.1 ve B.2 metni yeniden kontrol edildi. Kaynaktan alınanlar ortak valid-pair
R/D/U, pooled-bias kappa, all-prompt accuracy, McNemar/Holm ve matched-instance
bootstrap/Wilson yöntemleridir. Aşağıdaki eksiklik politikası, etkileşimler,
Türkçe–İngilizce kontrastlar ve dosya sözleşmesi Turkish-CBE uzantısıdır.

- Önceki §8'deki taslak B=5000, kullanıcının Prompt 6 talimatıyla **B=2000**
  olarak güncellenir; seed ve B çıktı kökenine yazılır. Aralıklar %95,
  percentile bootstrap ve R için ayrıca Wilson'dır. Analiz birimi
  latent_instance_id kümesidir; bir sıranın bütün sunum/dil/sorguları birlikte
  yeniden örneklenir. n tabakası korunur; küme azlığı/tek küme aralıkları
  açıkça tanımsız veya betimsel işaretlenir.
- R/D/U, Mean Accuracy, invariance ve q aynı ortak ayrıştırılabilir çiftler
  üzerinde hesaplanır. q aynı n ve açık analiz hücresindeki iki cevabın YES
  oranıdır. Karmaşıklık seviyeleri kappa için birleştirilmez. V=0 veya kappa
  paydası=0 ise null ve nedeni raporlanır. Original/Equivalent Accuracy,
  invalid rate ve strict R bütün **planlı** çiftleri payda olarak kullanır.
  Eksik/taşıma hatalı/çözülememiş yanıtlar bu paydadan çıkarılmaz.
- Plan ve kayıtlar task_id ile eşleştirilir; eksik task plan üzerinden yeniden
  kurulur. Plan olmadan yalnız gözlenen yanıtlarla başarı yüzdesi üretilmez.
  Cache, yinelenen task, sürüm/cohort uyuşmazlığı reddedilir. Ana analiz
  replicate=0'dır; tekrarlar bağımsız örnek yapılmaz. Fixture verisi yalnız
  testte, açık sentetik etiketle kullanılabilir; gerçek analiz diye sunulamaz.
- Canonical her equivalent/P1–P5 ile ayrı eşleştirilir. Model/sürüm, split,
  isim kontrolü, ilişki, query türü, n ve distance tabakaları korunur.
  İlişki ailesi özetleri bağımlı kümeleri korur; endpoint/internal/distance
  sonuçları kontrolsüz birleştirilmez. Model ailesi sonuçları model kimliğini
  koruyarak etiketlenir; farklı modeller bağımsız latent örnek sayılmaz.
- McNemar invalid=yanlış doğruluk üzerinde iki yönlüdür. Discordant≤25 ise
  exact binomial, aksi halde continuity-corrected chi-square kullanılır.
  Bir hücrede aynı latentten çok gözlem varsa bağımsızlık ihlalinden dolayı
  sıradan McNemar p-değeri verilmez; küme bootstrap kontrastı raporlanır.
  Wilson da böyle hücrelerde bağımsızlık varsayımıyla betimsel etiketlenir.
- Holm ailesi testin bilimsel kontrastı ve korunan tasarım eksenleriyle
  tanımlanır, modeller o ailede birlikte düzeltilir. Henüz bulunmayan model
  ailesi ve eksik kontrastlar listelenir. Bu keşifsel uygulama dış önkayıt
  yerine geçmez; veri sonrası seçilmiş aile doğrulayıcı sayılmaz.
- Dil farkı TR−EN'dir; presentation etkileşimi (TR_eq−TR_org)−(EN_eq−EN_org),
  complexity etkileşimi bir n'deki dil farkı eksi en küçük n'deki farktır.
  Dört hücrenin planlı eşleşmesi zorunludur; invalid yanlış kalır. Token–invalid
  ilişkisi kaynağı/coverage'ı belirtilen token sayısı üzerinde betimsel
  korelasyondur; dil/token uzunluğu üzerinden nedensel etki iddia edilmez.
- Altı şekil ve CSV/LaTeX tabloları yalnız analiz edilebilir gerçek kayıt varsa
  araştırma çıktısı olur. Sentetik şekiller test geçici klasörlerinde kalır.
  Boş gerçek sonuç klasörü için yalnız no_model_results durum kaydı yazılır;
  sahte performans tablosu, p-değeri veya grafik oluşturulmaz. Girdi/kod/config
  hashleri, eksik/invalid sayıları ve kullanılan paydalar rapora dahil edilir.

## 17. Prompt 7 — lexical analiz öncesi kararlar (2026-10-25)

Prompt 6 final 597 test/iki Python ve kurulu paket doğrulaması tamamlandıktan
sonra bu aşamaya geçildi. Önceki 50 Python kaynağı korunur. Bu ek analiz
korelasyoneldir; model ağırlığı, eğitim korpusu veya nedensel mekanizma ölçümü
değildir. Yeni lexical koddan önce şu kararlar kaydedildi:

- Bağımsız referans korpus olarak UD Turkish-BOUN'un yayımlanmış CC BY-SA 4.0
  veri lisansı ve metin içerdiği metadata doğrulandı. Sabit release/commit,
  dosya SHA'ları ve tüm train/dev/test parçaları kaydedilecek. Bunlar Turkish-CBE
  train/test verisi değildir; frekans sayımıdır. Haber/nonfiction seçkisi genel
  Türkçe veya LLM eğitim dağılımının temsilcisi sayılmaz. Kaynak alınamazsa
  sıklık null kalır; sıfır yalnız gerçekten sayılıp bulunmayan biçim içindir.
- Surface sayımı CoNLL-U multiword token yüzeylerini tek kere kullanır;
  lemma sayımı ayrı syntactic-word LEMMA katmanındadır. Noktalama/satır ve
  cümle sınırı gözetilir; lemma çözümleyici tahmini yapılmaz. Türkçe I/İ
  dönüşümü ve NFC sabittir. Exact surface, canonical lemma ve çok sözcüklü
  ifade kapsamları ayrı raporlanır. Log ve yön oranında sıfır/tanımsızlık
  politikası açık kaydedilir, sessiz smoothing yapılmaz.
- Query lexical direction (before/after), latent query_direction ve öncüllerin
  sunumdaki converse karışımı farklı eksenlerdir. P2 öncül yönünü değiştirir;
  gold korumak için özne/nesne yer değiştirmesiyle birlikte değerlendirilir.
  Canonical/equivalent list-order+direction birleşimidir; P2 ile karıştırılmaz.
  Soru ilişki yönü altgrupları gold dengesi ve invalid coverage ile raporlanır.
- Token sayısı modelin bildirilmiş tokenizer kimliği/revision'ı ile hesaplanır.
  OpenAI'nin donmuş BPE dosyası düz metin sayar; chat framing sayısı değildir.
  HF/vLLM tokenizer varlıkları yerelde yoksa başka tokenizer kullanılmaz,
  sayılar null/missing_local_tokenizer olur. Bu aşamada model ağırlığı indirme
  ve ücretli/count API çağrısı yoktur. Gerçek inference usage varsa ayrı
  input-token alanında tutulur. TR/EN oranı yalnız aynı latent/soru/sunum/
  isim kontrolü ve aynı tokenizer/model eşleşmelerinde hesaplanır.
- Predicted YES/NO oranı geçerli yanıtlar üzerinde, valid coverage ve planlı
  gold dağılımıyla birlikte verilir. Aşırı eğilim tanı eşiği 0.90/0.10 olarak
  önceden belirlenir; bu bir anlamlılık testi değildir. Ortak valid desteğinin
  gold dağılımı da raporlanır; seçici invalidler gizlenmez.
- Spearman rank bağları ortalama rank ile ele alınır. Aynı frekans değerinin
  yüzlerce latent sunumda tekrarını bağımsız sözcük kanıtı sayan p-değeri yoktur.
  Analiz hücreleri model/n/ilişki ailesi/kol/isim/sunum kontrollüdür. Çok
  değişkenli regresyon model, log2(n), family, token uzunluğu ve lexical yönü
  kontrol eden keşifsel tanıdır. Rank eksikliği, az bağımsız sözcük ve sabit
  değişkenlerde sonuç null/uyarı olur. Model×vocabulary ayrı modelde incelenir;
  vocabulary fixed effects ile sabit lexical frequency aynı etkide ayrıştırılmaz.
- Gerçek LLM sonucu olmadığı durumda sıklık/tokenizer betimlemeleri üretilebilir;
  bunlar model performansı değildir. Bias, accuracy/R/invariance ilişkileri,
  p-değerleri ve performans grafikleri üretilmez. Eksik korpus/tokenizer/model
  kapsamı çıktı manifestinde ve LEXICAL_ANALYSIS.md'de açık kalır.

## 18. Prompt 8 — reasoning kodundan önce belirlenen protokol (2026-10-25)

- A, eski direct mesajlarını aynen kullanır. B (zero-shot reasoning) ve C
  (normalize et / yönleri belirle / geçişleri değerlendir / sonuç) aynı problem
  metnini koruyan, iki dilde sürümlenmiş talimatlardır. B/C'de tek bir son
  satır `FINAL: EVET|HAYIR` veya `FINAL: YES|NO` zorunludur. Açıklama içinden
  cevap tahmin edilmez; yinelenen işaret, yanlış dil, işaretten sonraki metin,
  refusal ve kesilmiş yanıt invalid'dir. Açıklamanın doğruluğu puanlanmaz.
- D, direct mesajıyla yalnız sağlayıcının yerleşik effort ayarını değiştirir;
  B/C ile aynı müdahale sayılmaz. Destek, tam model/sürüm ve taşıma parametresi
  için kaynak/tarih içeren açık capability kaydına dayanır. Bilinmeyen veya
  desteklenmeyen seviyeler sıfır çağrı ve gerekçeyle raporlanır; varsayılan
  seviye `none` kabul edilmez. Eski GPT-4o/Qwen2.5/Llama3.2 kayıtları için native
  effort desteği varsayılmaz. Modele sessiz geçiş yapılmaz.
- Aynı planın bütün etkin kolları aynı latent/soru/sunum/dil koordinatlarını
  paylaşır. Gold, query türü, yönü ve mesafesi değişmez. Model/sürüm, sampling,
  seed ve bütçe kaydedilir. Birincil karşılaştırma ortak output token üst
  sınırı kullanır; doğrudan cevap kolu dahil daha geniş sınır yeniden çalışılan
  bir baseline'dır, eski 32-token sonuçlarıyla kontrolsüz birleştirilmez.
  Reasoning tokenları sağlayıcı saydığı ölçüde bu sınırı tüketir. Bütçe yeterli
  kabul edilmez: truncation/invalid/coverage tanıları model ve n bazında ayrıca
  incelenir. Bağlam taşmasında metin kesilmez, çağrı durdurulur.
- Sampling desteği effort ile değişiyorsa desteklenmeyen temperature/top_p
  gönderilmez; bu ayar bütün karşılaştırılabilir kollar için ortak ve açık
  yapılandırılır. Ayar otomatik silinmez veya örnekler sessizce düşürülmez.
- R/D/U, invariance ve kappa yalnız ortak geçerli presentation çiftleriyle,
  marginal accuracy ve strict R bütün planlı çiftlerle hesaplanır. Eksik veya
  kesilmiş cevaplar yanlış/invalid kalır. n, query türü/mesafesi, dil, ilişki,
  isim kontrolü ve model korunur. Strateji farkları aynı latent kümeleriyle
  eşleştirilir; bootstrap B=2000, seed sabit, %95 aralıklar, uygun McNemar ve
  Holm uygulanır. Koşullu R/invariance farklarında değişen valid desteği ayrıca
  raporlanır; ortak dört-cevap desteği duyarlılık analizi sağlanır.
- Reasoning kayıtları eski strict inference şemasını değiştirmeyen ayrı,
  sürümlü dizinde tutulur. Ham kanıt değiştirilemez, resume belirsiz çağrıyı
  yinelemez; cache bağımsız gözlem kabul edilmez. Dry-run çağrı, token ve
  yapılandırılmış fiyat tahminini raporlar; gerçek pilot onayı olmadan full
  çalışmaz. Kullanıcının model seçimini erteleme tercihi sürer: bu aşamada
  gerçek inference/model indirme yoktur. Sonuç yoksa yalnız durum manifesti
  üretilir; sentetik test çıktıları araştırma grafiği olarak yayımlanmaz.
- Gözlenen artış bütün problem ailesinde yetkinliği kanıtlamaz. Açık reasoning
  metni modelin iç hesaplamasının güvenilir açıklaması kabul edilmez; native
  hidden reasoning istenmez veya geri çıkarılmaz. Bu deney, prompting ve
  sağlayıcı ayarlarının son cevap doğruluğu ve sunum dayanıklılığına etkisini
  sınırlı, donmuş benchmark üzerinde ölçer.

Yerleşim düzeltmesi: ilk entegrasyon testi, `experiments/` altına yeni Python
dosyası eklemenin Prompt 4'ün özyinelemeli kaynak hash'ini bozduğunu gösterdi.
Önceki 57 kaynak ve veri manifestlerini korumak için uygulama `src/reasoning/`
paketine taşındı. İstenen `src/experiments/reasoning/` dizini bu sınırı açıklayan
README içerir. Veri/hash kontrolü gevşetilmedi; donmuş bankanın yeniden üretimi
ve önceki inference planlarının değiştirilmesi gerekmedi.

## 19. Prompt 9 — eğitim kodundan önce belirlenen protokol (2026-10-25)

- Yerel aygıt Apple M1/16 GiB birleşik bellektir; CUDA, torch/PEFT eğitim
  ortamı ve model ağırlıkları doğrulanmamıştır. Bu aşamada model indirme veya
  eğitim başlatılmaz. Donanım/bellek, bağımlılık, yerel ağırlık ve veri koşulları
  kaydedilir; başarısız koşulda yalnız plan/config/veri hazırlığı yapılır.
- Altı kol base, SFT-LoRA, paired SFT, SFT+consistency, GRPO correctness ve
  GRPO correctness+agreement olarak ayrıdır. Model kimliği mevcut registry'den
  alınır; örnek aday çokdilli Qwen2.5-1.5B-Instruct'tır. Ailenin daha büyük
  üyeleri için TurkishMMLU raporlanmıştır; bu 1.5B sürümün Türkçe sıralama
  yeterliliği bu ortamda ölçülmüş sayılmaz ve gerçek pilotta doğrulanmalıdır.
  Registry/dependency/model sürümü ve yerel asset hash'leri sabitlenmeden
  yürütme yapılamaz. Kapalı kayıt etkin kabul edilmez, model değiştirilmez.
- Makale B.3'teki rank16/alpha32/dropout.05, attention+MLP hedefleri,
  AdamW/cosine/warmup.03; SFT lr2e-4, epoch1/batch16/bf16/max_length1536;
  GRPO lr1e-5/beta0/G4/600step/seed42,43 başlangıç referansıdır. Bizim model,
  veri/vocabulary, dil ve donanım farklıdır; aynen replikasyon iddiası yoktur.
  Paper'da açık olmayan JS katsayısı/guard ve diğer ayrıntılar uydurulmaz.
- Yalnız frozen training latent kimlikleri ve n<=128 eğitime girebilir.
  Development model seçimi içindir; test kimlikleri hiçbir supervised target,
  reward baseline, avantaj grubu, checkpoint seçimi veya eğitim batch'ine
  giremez. Held-out relation family/vocabulary ve n>128 transfer cohort'u ayrı
  saklanır. Yüzey/metin hash, lexeme/lemma/ifade ve morfolojik örtüşme denetimi
  yapılır; stopword benzerliği ile relation-content çakışması ayrılır. Türkçe
  training ve iki dilli değerlendirme koşulları açıkça kaydedilir.
- SFT yalnız assistant cevabında CE kullanır; prefix etiketleri maskelenir.
  Paired kolda aynı latent/soru/gold için canonical/equivalent birlikte tutulur.
  JS, bütün YES/NO veya EVET/HAYIR dizilerinin teacher-forced olasılıklarından
  hesaplanır; tek token varsayılmaz. Bu ikili koşullu dağılım invalid çıktı
  kütlesini ölçmez; serbest üretim valid coverage değerlendirmede kalır.
  Bizim JS guard'ımız her iki sunumun gold olasılığı .5 üstündeyse etkin olan
  detached doğruluk guard'ıdır; paper'ın belirtilmemiş guard formülü diye
  sunulmaz. CE birincil hedef, consistency katsayısı config'de açık kalır.
- GRPO'da her instance/query için aynı gold'lu farklı sunumlar ve G=4 örnek
  birlikte örneklenir. q yalnız o grubun valid cevaplarından hesaplanır;
  chance=q²+(1-q)², bonus=max(0, diğer sunumlardaki anlaşma−chance).
  Invalid/truncated cevapların correctness/bonus/reward'u sıfırdır; eksik
  valid karşılık bonus alamaz. Reward=correctness+0.5*bonus. Kendi sunumunun
  tekrarları cross-presentation bonusuna sayılmaz; latent/soru karıştırılmaz.
- Paper prompt-normalized advantage referansı ve instance-centered unscaled
  alternatif ayrı etiketlenir. İlki binary çıktı halinde bonusu etkisiz
  bırakabilir (Ek G); correct-only ile advantage farkı/cosine ve sıfır-update
  adımları raporlanır. İkinci gruplama yeni kontrol müdahalesidir; paper'ın
  aynı pozitif-clipped bonus formülüyle uyarlanır, signed grpo-ig replikasyonu
  sayılmaz. Correctness ayrı ve öncelikli izlenir; salt aynı yanlış cevabı
  tekrarlamakta q=0/1 ve bonus=0'dır.
- Karşılaştırmalar aynı LoRA parametre setini ve disjoint train/eval koşulunu
  korur. Referans esinli epoch/step planı ile optimizer-step eşitlenmiş kontrol
  ayrı kaydedilir. Örnek/prompt/pair/rollout sayısı, gerçek token exposure,
  step, trainable parametre×update, GPU zamanı ve maliyet ayrı alanlardır.
  Eşit step eşit token/GPU bütçesi sayılmaz. Ölçülmemiş GPU süre/maliyet null
  kalır; bellek hesapları yalnız açıklanmış alt sınır/sezgisel tahmindir.
- Adapter değerlendirmesinde aynı held-out latent/query/presentation setleri
  kullanılır; eski CBE accuracy/R/D/U/invariance/kappa/coverage paydaları
  aynen korunur. Büyük n ve clean held-out family transfer ayrı raporlanır.
  Kaydedilmiş adapter/base pin ve veri bağları doğrulanmadan sonuç kabul
  edilmez. Eğitilmemiş kollar planned/blocked kalır; sonuç veya başarı grafiği
  uydurulmaz. Eğitim kodu ile sonuç raporlama modülü ayrıdır.

Uygulama denetimiyle kesinleşen GRPO sampling kararı: davranış dağılımı ile
old/current likelihood aynı `softmax(logits/temperature)` olmalıdır. Bu küçük
backend tam destekli temperature sampling kullanır; top_p=1/top_k=0 zorunludur.
Fresh GenerationConfig ve use_model_defaults=false ek decoder işlemcisinin
snapshot'tan miras kalmasını engeller. JS answer scoring bu sampling policy'si
değildir ve sıcaklık1'de kalır. Bu yerel karar referans sampling ayarı olarak
sunulmaz. Loop wall bütçesi adım öncesi/sonrası ve rollout aralarında kontrol
edilir; final adımda aşım completed durumuna dönüşmez.

## 20. Prompt 10 — publication reporting protocol (2026-10-25)

This stage prepares an English scientific draft, not an empirical finding.
Before reporting code is added, the following unresolved conditions apply:

- No recorded model responses or completed training runs exist. Training
  hardware validation (seven optional Torch tests and CUDA smoke), a selected
  real inference pilot, and native-speaker linguistic review remain pending.
  Those external conditions cannot be replaced with simulated results.
- Dataset statistics are measured properties of frozen generated data. They
  are not sample sizes of independent model trials. Latent orders, queries,
  rendered prompts, matched pairs and repeated calls must be distinguished.
- Every empirical paper table must come from native validated run readers and
  their existing statistical modules. Arbitrary CSVs, test fixtures, dry-run
  plans and gold labels alone cannot supply a model-performance estimate.
  Missing results require explicit status and null estimates, sample sizes,
  intervals and p-values; missing evidence is not zero performance.
- RQ1 maps to Table 3, RQ2 to Table 4, RQ3 to Table 6, RQ4 to Table 5 and RQ5
  to optional Table 7. Table 1 describes the dataset, Table 2 describes model
  profiles. Endpoint/internal/distance, n, relation, name controls, language,
  presentation, model revision, prompting and training arms remain stratified.
- Invalid responses remain wrong in marginal accuracy. R/D/U use common valid
  pairs; coverage and strict R retain planned denominators. Kappa with zero
  denominator is undefined. Matched-latent bootstrap uses 2000 repeats with
  fixed seed and n strata. Holm and appropriate paired tests come from the
  existing analyses; dependent Wilson intervals are descriptive only.
- Architecture/pipeline figures may be produced without model observations.
  Accuracy, differences, R/D/U and vocabulary-bias figures must remain explicit
  missing-result placeholders until genuine corresponding measurements exist.
- Human review is pending; P4/P5 are multiple transformations and do not isolate
  a purely morphological effect. Corpus frequency is not model training
  exposure and is correlational. Held-out English contexts share higher/lower.
  A finite-family score cannot establish universal logical competence.
- Existing Turkish benchmarks and morphology work preclude an unsupported
  first-ever claim. The contribution is a controlled ordering adaptation and
  an auditable evaluation protocol. CBE inheritance and Turkish extensions are
  distinguished, with primary sources and explicit citation scope.
- Reporting source resides in reports/publication, separate from the frozen
  src tree. This avoids invalidating Prompt 9's full-src plan fingerprints.
  Package discovery will include both source roots with explicit package
  discovery. All prior 75 Python sources and historical plans stay unchanged.
- Publication writes only new paper/report directories. Inputs are never
  edited. Output/input/code/config/environment hashes and unexecuted conditions
  are recorded; a new report directory is required for each new analysis.
  A draft with missing data is not submission-ready experimental evidence.


## 21. Prompt11 denetim kararları (2026-10-25)

- Yazılım dağıtımı0.11.0; dört Major yürütme/raporlama açığı RESEARCH_AUDIT.md içinde önce/sonra kanıtıyla kapatılır. Veri üretimi, frozen bankanın kaynak bağları ve eski ham/işlenmiş çıktılar değişmez.
- Dilsel review eşik tanımı aşama adından bağımsız **toplam planned calls >2.000**'dir. Full inference onayı ve büyük pilot/reasoning/training evaluation aynı dataset-bound validator kullanır. Önce gerçek pilot, sonra onaylı full; max-calls artışı review kapısını kaldırmaz.
- İki bağımsız Türkçe yeterliği belirtilmiş insan incelemeci gerekir. Her gerçek örnekte dilbilgisi, doğallık, anlamsal eşdeğerlik ve morfoloji pass; uzlaşma approved olmalı. En az100 benzersiz TR grid prompt, her seçilmiş relation×presentation×name template hücresinde20 örnek, her n ve query_type için kapsam gerekir. n×query×template tam çapraz hücre başına20 olarak yorumlanmaz. Bu örneklem onayı bütün yüzeylerin doğruluğunun ispatı değildir. Gerçek inceleme yapılmadan approved alanı doldurulmaz; AI ve fixture-only kayıtlar onay değildir.
- Training evaluation yeni, neutral greedy GenerationConfig kullanır; modelin generation defaults ayarları devralınmaz. Config/seed/hash hem manifestte hem response satırlarında izlenir.
- Base–trained karşılaştırması exact aynı plan koordinatları üzerinde trained-minus-base olarak hesaplanır. Accuracy, strictR ve coverage tüm planlı örnekler üzerindedir; R/invariance/kappa common-four-valid destek üzerinde duyarlılık analizidir. İki kolun ve ortak desteğin sayıları, cluster-bootstrap CI ve uygun paired test birlikte raporlanır. Nonbase değerlendirme, tamamlanmış eğitim kaydıyla exact adapter identity eşleştirmesi gerektirir.
- Holm aileleri model/design/metric bazındadır; training karşılaştırmalarında5 önceden tanımlı kol eksik kollar dahil düzeltme yuvası tutar. Bu uygulama çalışma çapında FWER sağlamaz. Sonuçlar görülmeden confirmatory primary aileler ayrıca sabitlenmedikçe analizler exploratory kalır; mevcut hipotez taslağı dış preregistration sayılmaz.
- Bağımlı R profillerinde cluster-bootstrap CI ana yorumdur; Wilson betimseldir. Endpoint derece kısa yolu bu bankadaki3.000 örnekte doğrulanır; n veya d modelin gerçek hesaplama derinliği diye raporlanmaz.
- İnsan onayı, gerçek LLM sonucu ve uygun CUDA backend doğrulaması beklenmektedir. Saf Python/sentetik testler bu eksikleri kapatmaz. Eski source-bound planlar yeni kaynakla yürütülmez; tarihsel kanıtlar korunup yeni hedefte plan/rapor üretilir.
