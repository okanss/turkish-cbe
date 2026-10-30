# Turkish-CBE — Reasoning deneyleri (Prompt 8, 0.8.0)

Bu aşama deney altyapısını, deterministik planları ve analiz/test yolunu ekler.
**Gerçek model sonucu yoktur.** Kullanıcının önce altyapı/dry-run, sonra model
seçimi tercihi korunur. API çağrısı, model ağırlığı indirmesi, gerçek pilot veya
full evaluation başlatılmadı. Başarı artışı veya yetkinlik sonucu raporlanmaz.

## Soru ve yöntem sınırı

Reasoning prompting ortalama doğruluğu artırsa bile, aynı latent problemin
eşdeğer sunumlarında iki kez doğru cevap verme kapasitesi R ve invariance aynı
ölçüde artmayabilir. Özellikle invariance'ın Shared Wrong (D) yoluyla artması
başarı değildir. Bu yüzden accuracy, R/D/U, kappa, coverage ve strict R birlikte
raporlanır. Sonuçlar sınırlı benchmark desteğinde yorumlanır; bütün problem
ailesinde yetkinliğin kanıtı sayılmaz.

CBE'nin ortak latent yapı, oracle ve anlam koruyan sunum ilkeleri korunur.
Bu aşamadaki A–D prompting müdahaleleri, standart final işaretleyicisi,
provider capability kaydı ve strateji kontrastları **Turkish-CBE uzantısıdır**;
referans makalenin doğrulanmış doğrudan replikasyonu olarak adlandırılmaz.
Koddan önce kararlar EXPERIMENT_PROTOCOL.md §18'e kaydedildi.

## Deney kolları

| Kol | `strategy` | Müdahale | Parser |
| --- | --- | --- | --- |
| A | `direct` | Önceki sistem/kullanıcı mesajları aynen; yalnız EVET/HAYIR veya YES/NO | `strict-answer-v1` |
| B | `zero_shot_reasoning` | Sonuçtan önce adım adım değerlendirme talimatı | `reasoning-final-answer-v1` |
| C | `structured_reasoning` | Normalize et, yönleri belirle, geçişleri değerlendir, sonuç üret | `reasoning-final-answer-v1` |
| D | `provider_effort` | A mesajı aynen; yalnız doğrulanmış native effort değiştirilir | `strict-answer-v1` |

B/C son satırları Türkçede `FINAL: EVET` / `FINAL: HAYIR`, İngilizcede
`FINAL: YES` / `FINAL: NO` biçimindedir. Tam bir işaretleyici beklenir; eksik,
yinelenen (aynı cevap olsa bile), bozuk veya son satırda olmayan işaret invalid
olur. Yanlış cevap dili ayrıca kaydedilir. Açıklamadaki YES/NO sözcükleri cevap
adayı değildir; açıklamanın içeriği veya ikna ediciliği puanlanmaz. Refusal ve
truncation, son token görünse de invalid kalır. D, görünür açıklama istemez;
provider'ın iç reasoning metni istenmez veya yeniden oluşturulmaz.

## Yerleşik effort ve destek doğrulaması

`configs/models.yaml` değişmedi; model adı/sürümü/tokenizer/sampling/serving
tanımları oradan seçilir. `configs/reasoning.toml` ortak output üst sınırını,
stratejileri ve model/sürüm bazında doğrulanmış capability beyanlarını saklar.
Varsayılan ortak sınır **2048 token**; yeterli olduğuna dair bir sonuç değildir.
A dahil bütün etkin kollara uygulanır. Eski 32-token direct sonuçları bunun
yerine geçirilemez. Karşılaştırılan kollar sıcaklık/top_p/seed açısından aynı
kalır; effective spec ve değişmemiş base spec ayrı hashlenir.

Native effort desteği evrensel değildir; seviyeler ve varsayılanlar modele
bağlıdır. Reasoning tokenları context/output bütçesinde yer tüketebilir.
Kaynak: [OpenAI reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)
(2026-10-25 erişimi). Bazı effort/model kombinasyonlarında temperature/top_p
desteklenmez; bu kısıtlar capability kaydında açıkça belirtilir ve uyumsuz plan
durdurulur. Kaynak: [OpenAI deployment checklist](https://developers.openai.com/api/docs/guides/deployment-checklist).
Talimatların varlığı doğal dil açıklamasının modelin iç sürecine sadık olduğunu
göstermez.

Her beyan provider, **tam** `model_id`, revision, HTTPS kanıt URL'si, doğrulama
tarihi, desteklenen seviyeler ve her seviyenin sampling kısıtlarını içerir.
Bu yerel kayıt operatörün kaynak kontrolünü belgeler; planlayıcı internetten
model desteği tahmin etmez. Gerçek yürütme ayrıca model kimliği ve exact token
sayımı için preflight yapar. Sağlayıcı hatası durumunda model/ayar değiştirilmez.

Mevcut altı kapalı model kaydında doğrulanmış native effort beyanı yoktur.
Bu yüzden D'nin none/low/medium/high seviyeleri **24 unavailable hücre**, her
biri sıfır çağrı olarak planlanır. Yerel HF/vLLM adaptörleri boolean thinking
anahtarını bu effort seviyelerine eşitlemez. Bu adaptörlerde A/B/C çalışabilir;
D, desteklenen taşıma ve doğrulanmış model kaydı gerektirir. OpenAI adaptörünün
dört seviyeyi aynen gönderdiği sentetik taşıma testi, herhangi bir gerçek modelin
bu seviyeleri desteklediğinin kanıtı değildir. `provider_default`, `none` demek
değildir.

## Eşleştirme, bütçe ve günlükler

Her planın bütün etkin kolları aynı latent/soru/prompt koordinatlarını paylaşır.
n, query türü/mesafesi/yönü, hedef kişiler, gold ve sunumlar değişmez. Model,
seed veya bağlam sınırı yüzünden sessiz örnek düşürülmez. Planlı bir hücrede
cevap yoksa analiz o cevabı missing/invalid olarak yeniden kurar.

Plan generation çağrılarını, retry üst sınırını, auxiliary HTTP/token sayımı
işlemlerini, model başına token proxy'sini, ortak output üst sınırını ve mevcut
fiyatlarla tahmini maliyeti saklar. Offline token tahmini, eski donmuş BPE
uzunlukları + 64 chat tokenı sezgiseli + ek talimatın tam UTF-8 byte farkını
token proxy'si olarak kullanır. **Exact yeni chat/tokenizer sayımı değildir**;
HF/vLLM için de yalnız proxy'dir. Bilinmeyen fiyat null'dır, sıfır maliyet
anlamına gelmez. Gerçek input/output/reasoning/cached usage sağlayıcı yanıtından
ayrıca alınır. Auxiliary/hardware maliyetleri bilinmiyorsa uydurulmaz.

Yürütmede exact token sayımı yapılır; input + output üst sınırı context'i aşarsa
metin kesilmeden çağrı durdurulur. Rate limit, exponential backoff, append-only
ham kayıtlar, benzersiz task/attempt ID'leri ve checkpoint/resume eski inference
primitiflerini kullanır. Timeout sonrası belirsiz çağrı otomatik yinelenmez;
`resolve` açık karar ve retry için olası çift ücretlendirme kabulü ister.
Cache varsayılan kapalıdır; cache reuse bağımsız araştırma gözlemi değildir ve
performans analizi tarafından reddedilir.

Reasoning namespace'i:

```text
src/reasoning/                    # bağımsız uygulama paketi (8 modül)
src/experiments/reasoning/README.md # donmuş kaynak sınırı ve yönlendirme
configs/reasoning.toml
results/reasoning/plans/          # değiştirilemez dry-run JSON
results/reasoning/raw/<run_id>/   # plan, execution, attempts, HTTP, preflight
results/reasoning/normalized/    # append-only JSONL
results/reasoning/checkpoints/   # SQLite, lock, rate limit
results/reasoning/cache/         # yalnız açık istenirse
results/reasoning/<analysis>/    # manifest, gerçek veri varsa CSV/LaTeX/figures
figures/reasoning/               # durum bilgisi; veri olmadan grafik yok
```

Uygulamayı `src/experiments/reasoning/*.py` içine ekleyen ilk entegrasyon testi,
donmuş Prompt 4 kaynak hash kontrolünü bozdu: eski builder `experiments/`
ağacını özyinelemeli bağlar. Bu nedenle yalnız yeni paket `src/reasoning/`
olarak ayrıldı; önceki 57 Python kaynağı ve veri manifestleri değiştirilmedi.
Hash kontrolüne istisna veya runtime monkeypatch eklenmedi. İstenen dizindeki
README bu bilinçli yerleşim farkını belgeler.

## Çalıştırma

Proje kökünden, kurulu paket yoksa `PYTHONPATH=src` kullanılır. Yeni dependency
eklenmedi; inference ve analiz için mevcut hashli lock dosyaları geçerlidir.

```bash
# Örnek, gerçek model seçimi veya API çağrısı değildir; kapalı kayıtla plan üretir.
PYTHONPATH=src python3 -m reasoning pilot --model qwen25-05b-hf --output results/reasoning/plans/my_pilot.json

# Kaydedilmiş planı çağrı yapmadan yeniden denetle.
PYTHONPATH=src python3 -m reasoning pilot --plan results/reasoning/plans/my_pilot.json --dry-run

# Full kapsamını yalnız planla; varsayılan bütçe/pilot-onay engelleri raporlanır.
PYTHONPATH=src python3 -m reasoning full --model qwen25-05b-hf --output results/reasoning/plans/my_full.json

# Gerçek kayıt yoksa yalnız no_model_results manifesti oluşturur.
PYTHONPATH=src python3 -m reasoning analyse --output results/reasoning/my_analysis --figures

# Unit testler gerçek inference değildir.
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_reasoning_*.py' -v
```

Kurulumla aynı komutlar `turkish-cbe-reasoning` üzerinden kullanılabilir.
Gerçek pilot için model seçimi, erişim ve uygun sabit sürümler belirlendikten
sonra model kaydı açıkça etkinleştirilir, yeni plan üretilip incelenir; ancak
ardından `pilot --plan ... --execute` kullanılır. Eski dry-run planı elle
değiştirilmez. Full için gerçek, tam, cache kullanılmamış, taşıma hatasız pilot
ve aynı veri bankasına bağlı insan dilsel onayıyla `approve-pilot` makbuzu
gerekir. Reasoning ayarları/model/kaynaklar değişirse eski onay geçmez.

```bash
turkish-cbe-reasoning approve-pilot --run-id PILOT_ID --approved-by REVIEWER --linguistic-review APPROVED_REVIEW.json --output APPROVAL.json
turkish-cbe-reasoning full --plan FULL_PLAN.json --execute --pilot-approval APPROVAL.json
```

Bu komutlar kullanım sözleşmesidir; teslimde approval veya gerçek pilot kaydı
üretilmemiştir. Yeni kaynak değişikliği/env farkı, eski reasoning planından
yürütmeyi durdurur. Eski P5 planları ve strict doğrudan parser korunur.

## Analiz sözleşmesi

- Her arm önce ayrı CBE analizine girer; sonra strateji etiketiyle birleştirilir.
  Model/sürüm, split, isim kontrolü, relation/family, n, query türü/mesafesi,
  dil ve sunum koşulu korunur. Endpoint/internal/distance kontrolsüz havuzlanmaz.
- R/D/U, ortak geçerli canonical/equivalent cevaplarla; marjinal accuracy,
  strict R ve invalid oranı bütün planlı örneklerle hesaplanır. kappa paydası
  sıfırsa null ve neden verilir; geçerli çift kapsamı daima raporlanır.
- Strateji−direct farkları original/equivalent ve all-planned mean accuracy,
  R, invariance, kappa, coverage ve strict R için aynı latent kümeleri üzerinde
  hesaplanır. Koşullu metrikler her kolun valid desteğini kullanır; ayrıca
  **aynı dört cevabın da geçerli olduğu** destekle duyarlılık analizi verilir.
  Azalan coverage yoluyla görünen iyileşme gizlenmez.
- Bootstrap 2000 tekrar, seed 20261025 ve %95 aralıklar kullanır. Aynı latent
  bütün sunumlarıyla birlikte yeniden örneklenir. Bağımsızlık uygunsa exact /
  asimptotik paired McNemar ve kayıtlı ailelerde Holm uygulanır. R'nin Wilson
  aralığı eski modülden gelir; yinelenen latentlerde betimsel sınırlama korunur.
- TR−EN profilleri ve `(TR_strateji−TR_direct)−(EN_strateji−EN_direct)` ayrı
  raporlanır. Türkçeye özgü karşılıksız P1–P5 için dil etkileşimi uydurulmaz.
  Token tüketimi, token raporlama coverage'ı, truncation ve invalid tanıları
  arm/model/n/query-distance/dil/sunum düzeyindedir.
- Tablolar CSV ve LaTeX; altı CBE şekil ailesi arm etiketli ayrı PDF/SVG/PNG
  olarak dışa aktarılır. Çıktılar yazılırken eskiler ezilmez. Gerçek gözlem
  yoksa performans tablosu/p-değeri/grafik üretilmez. Sentetik grafikler yalnız
  test geçici dizinlerinde, görünür sentetik etiketiyle doğrulanır.

## Kalan sınırlar

Gerçek model seçimi ve pilot; native effort destekli, tam sürümü sabitlenmiş
model kaydı; gerçek truncation oranına göre bütçe değerlendirmesi; pending
Türkçe insan dilsel incelemesi bekliyor. Yanıt dili/final işaretleyicisi de
prompt müdahalesinin bir parçasıdır. Ortak bütçe eşit hesaplama veya eşit
üretilen token anlamına gelmez; native effort ile görünür CoT ayrı yorumlanır.
