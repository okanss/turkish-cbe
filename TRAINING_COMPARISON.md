# Turkish-CBE — isteğe bağlı SFT/GRPO aşaması (0.9.0)

**Bu teslim eğitim sonucu içermez.** Altı kolun konfigürasyonları, verisi,
ödülleri, kaynak kontrolleri ve koşullu çalıştırma kodu hazırlandı. Hiçbir
adapter eğitilmedi, model ağırlığı indirilmedi ve model inference yapılmadı.
Makaledeki eğitim başarısı bu projede elde edilmiş gibi aktarılmaz.

## Ön kontrol ve aday model

Yerel bilgisayar Apple M1, 7 Metal GPU çekirdeği ve **16 GiB birleşik belleğe**
sahiptir. Bu bellek CUDA VRAM değildir. NVIDIA/CUDA eğitim backend'i yoktur;
PyTorch, PEFT ve Accelerate kurulu değildir. Tokenizer amaçlı Transformers
4.57.1 mevcuttur. Yerel model ağırlığı ve ağırlık manifesti bulunmaz. Gerçek
envanter `results/training/hardware.json` içinde, engeller her planda kayıtlıdır.
MPS/CPU'ya otomatik geçiş veya başka model indirmesi yapılmaz.

Örnek aday, mevcut registry'deki `qwen25-15b-hf` kaydıdır:
`Qwen/Qwen2.5-1.5B-Instruct`, model/tokenizer revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Model kimliği koddan alınmaz.
[Sabit model kartı](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/README.md)
ve Apache-2.0 lisansı aday seçimini belgeler. Çokdilli ailede Türkçe
değerlendirmeler yayımlanmıştır ([Qwen2.5 raporu, §5.2.2](https://arxiv.org/html/2412.15115v1));
bunlar bu 1.5B snapshot'ın Turkish-CBE yeterliliğinin ölçümü değildir. Gerçek
Türkçe pilot hâlâ gerekir. Kapalı model registry kaydı aynen korunmuştur.

Konfigürasyon CUDA:0, native BF16 ve en az 16 GiB doğrulanmış **boş CUDA
belleği** ister. Bu yönetim eşiği bir fit tahmini veya 16 GB kart önerisi
değildir. Base BF16 ağırlık alt sınırı yaklaşık 3.08 GB'dır; aktivasyon,
LoRA/optimizer ve rollout cache belleği ölçülmediğinden null kalır. GPU saat
ücreti ve throughput bilinmiyor; süre ve toplam maliyet uydurulmaz.

## Kollar ve üç profil

| Kol | Amaç | Eğitim girdisi |
| --- | --- | --- |
| `base` | Değişmemiş ağırlık kontrolü | Eğitim yok |
| `sft_lora` | Assistant cevabına CE | Canonical |
| `paired_sft` | Eşdeğer sunumlarla SFT | Canonical + equivalent |
| `sft_consistency` | CE + kontrollü JS | Aynı sorgunun iki sunumu |
| `grpo_correctness` | Correctness reward | Dört sunum × G=4 |
| `grpo_consistency` | Correctness + agreement bonus | Aynı dört sunum × G=4 |

`pilot.toml` küçük veri ve iki optimizer adımlık bir smoke konfigürasyonudur.
`reference_inspired.toml` SFT'de bir epoch, GRPO'da 600 adım planlar.
`matched_steps.toml` eğitilen beş kolda **600 optimizer adımı** tutar.
Bu son profil eşit token, GPU süresi veya benzersiz örnek bütçesi değildir.
Kaynak havuz tekrarları yeni bağımsız örnek sayılmaz. Base'in eğitim bütçesi
sıfırdır. Bütün profiller varsayılan kapalıdır.

| Profil | SFT-LoRA adımı | Paired/JS adımı | GRPO adımı |
| --- | ---: | ---: | ---: |
| Pilot | 2 | 2 | 2 |
| Referans esinli | 357 | 713 | 600 |
| Adım eşitlenmiş | 600 | 600 | 600 |

Referans esinli profilin yönetim amaçlı üst sınırı 1000 adımdır; 600 olsaydı
paired bir epoch 713 adımda tamamlanamayacaktı. Bu üst sınır makaleye ait bir
hiperparametre olarak sunulmaz. Pilot yalnız kısıtlı maruziyet sağlar; tüm
havuzun görüldüğü veya başarılı öğrenme gerçekleştiği anlamına gelmez.

## Kaynak yöntem ve bizim uyarlamamız

Başlangıç referansı [CBE makalesi B.3](https://arxiv.org/html/2610.22894v1#A2.SS3)
ve [Ek G](https://arxiv.org/html/2610.22894v1#A7)'dir:

| Referans bileşeni | Başlangıç ayarı |
| --- | --- |
| LoRA | rank 16, alpha 32, dropout .05; attention ve MLP |
| SFT | AdamW, lr 2e-4, cosine, warmup .03, epoch 1, batch 16, BF16, length 1536 |
| GRPO | lr 1e-5, beta 0, G=4, 600 adım; seed 42/43 |

Bizim modelimiz, üç training ilişkimiz, Türkçe veri havuzumuz, örnek sayıları
ve backend farklıdır. JS katsayısı .1, detached doğruluk guard'ı, per-sequence
loss ortalaması ve tek on-policy güncelleme yerel tasarım seçimleridir.
Orijinal çalışmanın aynen tekrarlandığı iddia edilmez. İlk plan seed 42'dir;
43 için ayrı dosya/run üretilir; test performansıyla iyi seed seçilmez.

## Veri ayrımı ve aktarım

Yalnız frozen **training** partition'ı eğitime girer; n≤128. İlişkiler
`comparison_height`, `comparison_speed`, `comparison_weight`; dil Türkçe,
isim kontrolü `symbolic_shared`'dir. Sorgular endpoint, internal ve
distance-controlled olarak ayrı metadata taşır. Her unit tek latent/sorgu/
relation/gold ve canonical/equivalent/P1/P2 sunumlarından oluşur.

| Havuz | Latent sıra | Sorgu | Query×relation unit | Paired satır | GRPO seçilen unit |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot | 8 | 32 | 96 | 192 | 8 |
| İleri profiller | 300 | 1900 | 5700 | 11400 | 600 |

İleri GRPO yalnız n=16,32,64,128 kullanır; uygun 4500 unit arasından seed ile
600 seçilir. Dört query-direction×gold hücresinde 150'şer unit vardır.
Bu seçimin tamamı bir kez kullanılırsa 600×4×4=9600 rollout cevabı planlanır;
gerçek üretim sayısı sıfırdır. Pilotun iki adımı seçilen sekiz unitin yalnız
ikisinde güncelleme dener; bu bir istatistiksel sonuç deneyi değildir.

Değerlendirme test partition'ında aynı EN/TR ve sunum koordinatlarını tutar:

1. n≤128, training relation vocabulary;
2. n≤128, held-out spatial vocabulary;
3. n=256,512,1024, training vocabulary;
4. büyük n ve held-out spatial vocabulary birlikte.

Training/test latent ID ve görünür prompt hash kesişimleri sıfır olarak
doğrulanmıştır. Aynı isim listesi yeni latent örnek sayılmaz. Held-out spatial
ailesinin **yüklem/soru sözlüğü** training sözlüğünden ayrıdır; ancak İngilizce
bağlam cümleleri `higher/lower` sözcüklerini paylaşır. Dolayısıyla
`clean_full_context_holdout=false` ve insan incelemesi gereği açık kalır.
Tam sözcüksel izolasyon veya bütün ailesiyle transfer başarısı iddia edilemez.
Renderer ve donmuş promptlar bu aşamada değiştirilmedi.

## Loss ve reward sözleşmesi

SFT yalnız assistant tokenlarını ve terminal chat/EOS tokenlarını supervise
eder; bütün prompt prefix etiketleri -100'dür. Cevap tek token kabul edilmez.
Paired sunumlar optimizer adımı içinde birlikte tutulur. JS, tam HAYIR/EVET
cevap dizilerinin teacher-forced log olasılıkları üzerinden ikili koşullu
dağılımda hesaplanır. İki sunumda da gold olasılığı .5 üstündeyse detached
guard açıktır. Bu guard makalenin yayımlanmamış guard ayrıntısının tahmini
değildir. İkili koşullama invalid çıktı kütlesini ölçmez; valid coverage serbest
üretimde ayrıca değerlendirilir.

GRPO grubunda farklı sunumlar aynı latent ve **aynı sorgu/gold**'a aittir.
Valid örneklerin instance içi YES oranı q'dur:

`chance = q² + (1−q)²`

`bonus_i = max(0, agreement_with_other_presentations_i − chance)`

`reward_i = correctness_i + 0.5 × bonus_i`

Kendi sunumunun tekrarları agreement paydasına girmez. Invalid/truncated
cevaplar sıfır reward alır, q'ya girmez; valid karşılık yoksa bonus yoktur.
Sürekli aynı yanlış cevapta q=0/1, chance=1 ve bonus=0'dır. Correctness, bonus,
invalid rate, gradient norm ve zero-gradient adımları ayrı kaydedilir.

İki advantage modu ayrı konfigürasyon müdahalesidir:

- `prompt_normalized`: presentation başına ortalama/population std, epsilon
  1e-4. Binary grupta bonus etkisiz veya yalnız epsilon kaynaklı ölçekleme
  yapabilir; bu durum correctness-only advantage ile cosine/fark tanısıyla
  ölçülür. Ham reward artışı etkili öğrenme katkısı olarak yorumlanmaz.
- `instance_centered`: dört sunum birlikte merkezlenir, std bölmesi yoktur.
  Bu bir kontrol uzantısıdır; burada bonus pozitif-clipped kalır ve paper'ın
  signed grpo-ig deneyinin aynısı değildir.

Sampling ve old/current logp aynı tam-destekli `softmax(logits/temperature)`
dağılımını kullanır. `temperature=.6`, `top_p=1`, `top_k=0` açıktır; nucleus
filtreleme bu küçük backend'de desteklenmez. Yeni `GenerationConfig` ve
`use_model_defaults=false`, snapshot'tan ceza/forced-token/warper mirasını
engeller. JS answer scoring sıcaklığı 1 olarak ayrı kalır. Bu sampling seçimi
makalenin aynen tekrarlandığı iddiası değildir.

Backend beta=0 ile her on-policy rollout grubunda tek clipped policy-gradient
güncellemesi uygular. KL/reference policy özelliği varmış gibi gösterilmez.
Dropout, GRPO sampling/old/current likelihood karşılaştırmasında kapalıdır;
SFT'de konfigürasyon dropout'u aktiftir. Gradient clipping 1; AdamW weight decay
0 açık uygulama seçimidir. Seed LoRA başlatılmadan önce sabitlenir. Aynı donanım
ve sürümler olmadan bit düzeyinde eşitlik sözü verilmez; deterministic kernel
yoksa işlem hata verir, sessizce nondeterministic moda geçmez.

## Bütçe ve kanıt

Planlar exact optimizer adımlarını, unique unit/latent/surface havuzunu,
sequence exposure ve token üst sınırlarını kaydeder. Exact eğitim tokenları
başta null'dır. Bütün seçilen diziler tokenizer ile **optimizer oluşturulmadan
önce** ölçülür; 1536 veya rollout bütçesini aşan tek satır varsa kesme/düşürme
yapılmadan işlem durur. Türkçe girdiler bu sınırı aşabilir; daha geniş sınır
gerekiyorsa yeni config/plan ve ortak karşılaştırma koşulları gerekir.

Gerçek çalıştırma CE/policy tokenları, JS answer-scoring, old-policy scoring,
rollout input/output tokenlarını ayrı ve toplam muhasebe alanında saklar.
Parametre sayısı yüklenen LoRA'dan ölçülür; parametre×optimizer-step bütçesi
ayrıca verilir. Zero-gradient step sayısı, çağrılan optimizer adımıyla aynı
kavram değildir. GPU event süresi ve training-loop wall time ölçülür; bunlar
aktif kernel kullanım oranı değildir. Saat ücreti girilirse loop süresinden
etiketli maliyet tahmini çıkar; loading/preflight/evaluation ve fatura dahil
değildir. Wall sınırı adım öncesi/sonrası ve rollout aralarında kontrol edilen yumuşak
loop sınırıdır; son adım sınırı aşarsa run tamamlanmış sayılmaz.

Ham step/rollout logları, plan, execution, adapter ve hash envanteri yeni run
dizinine yazılır. Eski run ezilmez. Bu küçük backend training-resume uygulamaz:
yarım run `aborted_not_completed` kalır; yeni run gerekir. Tamamlanan adapter
değerlendirmeden önce plan/base-model ve payload hashleriyle doğrulanır.

## Kurulum ve komutlar

Mevcut veri/tokenizer bağımlılıkları önceki hash kilitlerinden kurulur.
Yeni aday paket pinleri pyproject `[training]` içindedir: torch 2.9.0,
transformers 4.57.1, peft 0.17.1, accelerate 1.11.0. Resmi release metadata
doğrulandı; tam CUDA stack entegrasyonu **henüz doğrulanmadı**. Transitive
eğitim kilidi üretildiği iddia edilmez; plan kurulu dağıtım sürümlerini saklar.
Kaynak API kontrolü: [PEFT v0.17.1](https://github.com/huggingface/peft/tree/v0.17.1),
[PyTorch 2.9.0](https://pypi.org/project/torch/2.9.0/).

```bash
# Kaynak ağacından; kurulu eşdeğeri turkish-cbe-train.
PYTHONPATH=src python3 -m training inspect --output results/training/NEW_hardware.json
PYTHONPATH=src python3 -m training plan --config configs/training/pilot.toml --output results/training/plans/NEW_pilot.json
PYTHONPATH=src python3 -m training export --plan results/training/plans/NEW_pilot.json --output results/training/data/NEW_pilot
PYTHONPATH=src python3 -m training run --plan results/training/plans/NEW_pilot.json --arm sft_lora --output results/training/runs/NEW_sft
PYTHONPATH=src python3 -m training evaluate --plan results/training/plans/NEW_pilot.json --arm base --output results/training/evaluations/NEW_base
PYTHONPATH=src python3 -m training report --plan results/training/plans/NEW_pilot.json --output results/training/reports/NEW_report
```

Son iki çalışma komutu da varsayılan dry-run'dır. Gerçek CUDA makinesinde
yalnız gereken ortam kurulduktan ve exact snapshot yerelde hazırlandıktan
sonra model registry ve config açıkça etkinleştirilir. `weights_path` ve
`weights_manifest` mutlak yerel yollardır; `local_files_only=true` zorunludur.
Manifest bütün dosyaları SHA/boyut ve model/tokenizer revision ile bağlar:

```json
{"schema_version":1,"model_id":"FULL_HF_MODEL_ID","revision":"40_HEX_COMMIT",
 "tokenizer_id":"FULL_HF_TOKENIZER_ID","tokenizer_revision":"40_HEX_COMMIT",
 "files":[{"path":"config.json","bytes":123,"sha256":"64_HEX"},
          {"path":"model.safetensors","bytes":123,"sha256":"64_HEX"}]}
```

Bu yalnız şemadır; gerçek manifest değildir. Bütün tokenizer/config/weight
dosyaları listelenmelidir; dosya hash'i kimlik beyanını bağlar, tek başına
upstream ağırlığın bağımsız attestation'ı değildir. Model/tokenizer ve chat
template ayrıca yüklemede doğrulanır. Yeni makinede `plan --probe-cuda`
ile yeni plan üretilir; ancak sonra `run --plan ... --arm ... --execute`
çağrılır. Evaluate da açık `--execute`, tamamlanmış adapter için `--adapter-run`
ve yeterli `--max-calls` ister. Default eval limiti 2000'dir; mevcut tam
cohort'u sessizce küçültmez. `report --training-run ... --evaluation ...`
doğrulanmış gerçek usage ve CBE/held-out tablolarını birleştirir.

## Test kapsamı ve kalan işler

Ödül/advantage matematiği, prompt masking, multi-token cevaplar, batch/step
hesabı, frozen veri ve vocabulary audit, donanım/weights hash kapıları,
CLI/no-results yolları birim testlerle doğrulandı. CPU PyTorch kurulumu ağ
iznine rağmen shell ve Node erişiminde DNS/bağlantı hatası aldı. Bu nedenle
**yedi opsiyonel Torch sayısal/gradient testi ve gerçek GPU training/evaluation
smoke'u çalıştırılmadı**. Saf Python testlerini bunların yerine geçirmiş gibi
göstermiyoruz. Ayrıntı `results/training/backend_validation.json` içindedir.

Gerçek adapter yokken rapor `no_training_or_model_results` durumundadır;
accuracy/R/D/U/invariance/kappa/coverage veya transfer başarısı uydurulmaz.
Altyapı, GPU ortamında kontrollü doğrulama ve gerçek pilot için hazırdır;
eğitimin veya özgün tarifin başarılı replikasyonunun tamamlandığı söylenmez.


## Prompt11 denetim düzeltmeleri

Dağıtım0.11.0 nonbase evaluation raporlamak için exact completed adapter identity ister. `training_arm_contrasts` tabloları her declared trained arm eksi base etkisini ekler: all-planned accuracy/strictR/coverage, common-four-valid R/invariance/kappa duyarlılığı, destek sayıları,2.000 tekrarlı matched latent bootstrap ve uygun McNemar. Holm beş tanımlı kol üzerinde aynı model/design/metric ailesindedir; study-wide düzeltme değildir. Eksik kollar unavailable/null'dır.

`evaluate --execute --max-calls ... --linguistic-review <real_review.json>` toplam2.000 çağrıyı aşarsa dataset/scope bağlı iki-insan incelemesi ister; doğrulama GPU/model yüklemesinden öncedir. Bu teslim gerçek review üretmez. Evaluation fresh neutral greedy GenerationConfig kullanır, repo decoding defaults devralınmaz; sampling/seed/hash kaydedilir. Büyük değerlendirmelerde receipt ve incelenmiş CSV sonuç kökenine değişmez olarak eklenir.

Önceki planlar tarihsel source-bound kayıtlardır; Prompt11 kaynakla yeni plan gerekir. Hiçbir eğitim kolu çalıştırılmadı; gerçek CUDA stack, yedi Torch testi, measured budgets ve held-out model sonuçları hâlâ bekler.
