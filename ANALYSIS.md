# CBE analizi — v0.6.0

Bu teslim analiz **altyapısıdır**. Model inference sonucu yoktur. Sentetik test
çıktıları model başarısı olarak raporlanmaz; şekil testleri geçici dizinlerde
çalışır. İnsan dilsel onayı ve gerçek pilot bekliyor. Protokol §16 koddan önce
kaydedildi; bu belge dış önkayıt yerine geçmez.

## Metrikler ve paydalar

Her analiz hücresinde model/sürüm, split, isim kontrolü, ilişki, dil, n,
query türü ve mesafesi, canonical ile karşılaştırılan varyant sabittir.
Canonical/equivalent ve canonical/P1–P5 çiftleri ayrı tutulur. Kappa farklı n
seviyelerinden havuzlanmaz. Değerler 0–1 oranıdır; farklar TR−EN yönündedir.

| Alan | Payda ve anlam |
|---|---|
| original_accuracy | Bütün planlı çiftler; invalid/eksik original yanlış |
| equivalent_accuracy | Bütün planlı çiftler; invalid/eksik equivalent yanlış |
| mean_accuracy | Ortak geçerli çiftlerde iki doğruluğun ortalaması |
| R | Ortak geçerli çiftlerde ikisi doğru |
| D | Ortak geçerli çiftlerde ikisi yanlış |
| U | Ortak geçerli çiftlerde yalnız biri doğru |
| invariance | Ortak geçerli çiftlerde aynı cevap; R+D |
| kappa | (invariance−I_expected)/(1−I_expected) |
| invalid_response_rate | Invalid ve eksik yanıtlar / (2×planlı çift) |
| valid_pair_coverage | Ortak geçerli çift / bütün planlı çift |
| strict_robust_correctness | İkisi doğru / bütün planlı çift |
| difference | Eşleşmiş TR accuracy − EN accuracy; invalid yanlış |

R+D+U=1 ve mean_accuracy=R+U/2 aynı ortak geçerli destek üzerinde geçerlidir.
q iki sunumun **aynı geçerli çiftlerindeki** pooled YES/EVET oranıdır;
I_expected=q²+(1−q)². q=0/1 ise kappa `null`; geçerli çift yoksa koşullu
metrikler `null` olur. Boş payda sıfır başarıya çevrilmez. undefined_reasons
CSV içinde JSON metni olarak nedenleri korur. `_accuracy_valid` alanları
koşullu tanısal ölçülerdir; ana marjinal doğrulukla karıştırılmaz.

## Kayıt ve eşleştirme sözleşmesi

`python -m analysis` ve kurulu `turkish-cbe-analyse` aynı arayüzdür:

```sh
python -m pip install --require-hashes -r requirements-inference.lock
python -m pip install --require-hashes -r requirements-tokenizers.lock
python -m pip install --require-hashes -r requirements-analysis.lock
PYTHONPATH=src python -m analysis --root . --output results/analysis/empty_check
# Yalnız gerçek run oluştuğunda:
PYTHONPATH=src python -m analysis --root . --run RUN_ID --output results/analysis/RUN_ID --seed 20261024 --bootstrap-repeats 2000 --figures
```

`--run` verilmezse mevcut raw run manifestleri bulunur. Model kayıtları yoksa
sadece `analysis_manifest.json` yazılır: `status=no_model_results`,
`contains_model_results=false`, boş tablo/dosya envanteri. Bu bir performans
sonucu değildir. Yalnız taşıma hataları olan ve hiç model response'u bulunmayan
run da metrik tablosu üretmez; planlı/eksik/invalid sayıları durum kaydında kalır.

Her run için frozen cohort tekrar kurulur; model/mesaj/task kimlikleri dondurulmuş
kaynaklardan hesaplanır. Sadece normalize JSONL okumak yeterli değildir.
Run manifesti, execution kaydı, checkpoint outbox, ham request/response/error ve
resolution kanıtları birlikte doğrulanır. JSON yinelenen anahtarları, bool/int
şema kaymaları, hash/kimlik/terminal durum sapmaları reddedilir. Mevcut parser
ham cevaba tekrar uygulanır; etiket ve correctness alanına kör güvenilmez.
Aktif run'ın ortak okuma kilidi alınamazsa analiz durur. Okuyucu checkpoint'i
onarmaz ve ham/normalize baytları değiştirmez.

Hiç başlanmamış task dahil bütün planlı kayıtlar paydada kalır. Ana analiz
replicate=0'dır; diğer tekrar sayısı kökende raporlanır. Cache ve sentetik
fixture kayıtları araştırma CLI'sinde reddedilir. Aynı task'ın iki run'da
bulunması veya farklı cohort'ların birlikte verilmesi reddedilir. Testler için
kütüphane düzeyindeki `allow_synthetic=True` dışında fixture kapısı yoktur.
Analiz model çağrısı yapmaz. Uzak sağlayıcı geçmişi bağımsız kriptografik tasdik
sayılmaz; burada doğrulanan, arşivlenmiş runner kanıtlarının iç tutarlılığıdır.

## İstatistik ve ek analizler

- Varsayılan seed 20261024, bootstrap B=2000, güven düzeyi %95. Her hücre için
  kararlı SHA-türetilmiş alt seed, sıralamadan bağımsız yeniden örnekleme vardır.
  Latent sıra kümesi bütün bağlı satırlarıyla taşınır; n tabakası korunur.
  İki bağımsız kümeden az toplam veya herhangi bir tabakada tek küme varsa CI
  tanımsızdır. Bir bootstrap tekrarında metrik tanımsızsa o metrik için CI null
  kalır; defined_replicates raporlanır. Tanımsız tekrarlar sessizce atılmaz.
- Accuracy, R/D/U, invariance/kappa, strict R ve coverage için percentile CI;
  R için ayrıca ortak geçerli paydada Wilson CI. Bağımlı çiftli Wilson aralığı
  betimsel etiketlenir. Şekil 3 Wilson'ı tercih eder; bootstrap sonuçları ayrıca
  CSV'de korunur. Aralıklar pointwise'dır; eşzamanlı güven bandı değildir.
- McNemar all-planned doğruluk üzerinde iki yönlüdür. Discordant≤25 exact
  binomial, daha büyükse continuity-corrected chi-square. b=sol doğru/sağ yanlış,
  c=tersi. Birden çok satır aynı latent sırayı paylaşıyorsa normal McNemar
  p-değeri verilmez; cluster bootstrap kontrastı kullanılır.
- Holm, aynı bilimsel kontrast ve tasarım hücresinin bütün modelleri üzerinde
  uygulanır; model başına ayrı düzeltme yapılmaz. Null p-değerleri aile
  büyüklüğünde korunur. Aile ID/büyüklükleri çıktıda yer alır. Çok sayıda
  n/ilişki hücresinin bu keşifsel aile tanımı, tek bir doğrulayıcı RQ1 omnibus
  testi veya dış önkayıt değildir; bu karar gerçek çıktı görülmeden dondurulmalıdır.
- Language×Presentation: (TR_eq−TR_org)−(EN_eq−EN_org). Language×Complexity:
  n'deki dil farkı eksi aynı profilin en küçük n'sindeki fark. İki farklı n'nin
  aynı latent sıra olduğu varsayılmaz; n-tabakalı kümeler birlikte örneklenir.
- Endpoint vs internal/distance doğruluk farkı aynı latent sıra içinde
  internal−endpoint yönündedir. Soru/gold değişebildiğinden bu bir anlamsal
  eşdeğerlik veya saf çıkarım derinliği testi değildir. Query kolları ayrı kalır.
- Model ailesi her satırda vardır; farklı model/sürümler havuzlanmaz. İlişki
  ailesi özeti satır ağırlıklıdır ve latent kümeleri/bağımlılığı korur.
- Token–invalid ilişkisi Pearson/point-biserial betimsel korelasyondur; p-değeri
  ve nedensellik iddiası yoktur. `provider_reported_input_including_chat` ile
  `frozen_old_prompt_bpe_proxy` ayrı tutulur. İkincisi yeni chat uzunluğu değildir.
  Outcome invalid veya provider-failed kayıt; eksik response ayrı sayılır.
  Token eksikliği/coverage ve sabit değişken/tanımsızlık açık raporlanır.
  TR-only ablation gibi bulunmayan dil kontrastları ayrı tabloya kaydedilir.

## Tablolar, şekiller ve köken

Çıktı klasörü yeni olmalıdır; mevcut analiz ezilmez. CSV alanları kayıpsızdır;
LaTeX özel karakterleri kaçırır. Undefined/missing hücre `NA`, manifestte
`null` olur. `profiles`, `relation_family_profiles`, `presentation_tests`,
`language_differences`, `language_robustness`, iki etkileşim, `endpoint_internal`,
`invalid_token_relationship` ve `unavailable_contrasts` CSV/LaTeX üretilir.
Ayrıntılı CI metadata JSON metnidir; makale tablosu için ilgili kolonlar seçilebilir.

1. Accuracy vs n: original/equivalent, dil serileri.
2. Invariance vs n.
3. Robust Correctness vs n, Wilson aralıkları.
4. R/D/U stacked decomposition.
5. TR−EN paired accuracy difference ve CI.
6. Query distance vs accuracy; n sabit tutulur.

Bütün şekiller PDF/SVG/PNG'dir. Dosya/facet kimliği model/sürüm, split, isim,
ilişki, query türü, varyant gibi diğer eksenleri sabit tutar; sessiz ortalama
alınmaz. Endpoint n eğrisinde d=n−1 yapısal bağı açık etiketlenir; diğer n
profillerinde distance sabittir. SVG metni düzenlenebilir. Sentetik grafik
testleri açık watermark/metadata kullanır ve teslim araştırma sonuçlarına taşınmaz.

Manifest input dosya hashleri, plan/veri hashleri, analiz kod hashleri, config,
seed/B, Python/platform/kütüphane sürümleri, paydalar ve üretilen her dosyanın
SHA-256/byte uzunluğunu içerir. `requirements-analysis.lock` 12 tam sürümü ve
485 resmî PyPI artifact hash'ini sabitler. Matplotlib 3.10.7 bilinçli sürüm
seçimidir; en güncel sürüm olduğu iddia edilmez. Kilit macOS arm64 Python
3.12/3.13 üzerinde doğrulandı; başka platform/yorumlayıcı kombinasyonları denenmedi.
Temel metrik/istatistik modülleri standart kütüphaneyle bağımsız kullanılabilir.

## Kaynak ve uzantı ayrımı

CBE ortak-valid ayrışım, pooled-bias kappa ve kaynak istatistikleri için
[Jiang ve Zhai, §3 ve Appendix A/B](https://arxiv.org/html/2610.22894v1)
kontrol edildi. Strict R, Türkçe–İngilizce kontrastlar, etkileşimler, eksik task
politikası ve sağlam kayıt doğrulaması Turkish-CBE kararlarıdır. Kappa Cohen
kappa veya bağımsızlık mekanizması kanıtı diye yorumlanmaz.
Yöntem kontrolleri: [McNemar resmî API açıklaması](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
[NIST binomial proportion aralıkları](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[Matplotlib savefig](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html).
Bu uygulama statsmodels/SciPy bağımlılığı kullanmaz; küçük yöntemler
standart kütüphane ile uygulanmış ve bilinen sayısal örneklerle test edilmiştir.
