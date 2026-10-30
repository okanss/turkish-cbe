# Turkish-CBE

A reproducible framework for evaluating logical-ordering competence of language
models across Turkish and English and across answer-preserving Turkish
presentations. Companion repository of the manuscript in `paper/`
(*Beyond Translation: Evaluating Logical Reasoning Competence in Large Language
Models under Turkish Morphosyntactic Variations*, Oluçoğlu & Bursa).

Frozen bank: 3,000 latent orders, 23,200 source queries, 5,637,600 measured
prompt identities (not independent problem counts). Executed evidence, all
exploratory by design: two registered pilot runs (Qwen2.5-0.5B, gpt-4o), a
supplemental quantized local sweep of four Qwen2.5 scales over 16,618 prompts
each (manuscript §7.7), and a completed two-reviewer Turkish linguistic review
of 360 sampled stimuli (§7.8; artifacts in `validation/review_results/`).
The frozen-test confirmatory run and the training-adapter evaluation are
gated and unexecuted; no confirmatory claim is made. Methods and
source/extension separation: [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md);
data coverage: [EXPERIMENT_MATRIX.md](EXPERIMENT_MATRIX.md); schema:
[DATASET_CARD.md](DATASET_CARD.md); change history and evidence:
[PROGRESS.md](PROGRESS.md).

## What is NOT in this repository

- `data/{test,training,development,pilot,latent}/` and
  `data/complete_grid/grid.sqlite` — multi-GB materializations of the frozen
  grid, rebuilt deterministically from the tracked sources, plans, and binding
  manifests (which hash-verify the rebuild); see
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- `tokenizers/hf/` — license-gated `meta-llama` tokenizer assets that may not
  be redistributed; fetch them with your own Hugging Face account as described
  in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- `results/training/runs/*/adapter/` — trained LoRA adapter weights (71 MB
  each); plans, summaries, and step logs are tracked.

## Quality audit

[RESEARCH_AUDIT.md](RESEARCH_AUDIT.md) reports the 18-scope audit, the four
corrected Major software issues, and the open scientific prerequisites. The
execution gates it defines (two-reviewer receipts for >2,000-call runs, pilot
approval for frozen-test execution) remain in force and unissued.

```sh
PYTHONPATH=src:reports python3 -m research_audit --project-root . --output reports/audit/replay_001
```

Kurulu komut `turkish-cbe-audit`'tır. Yeni çıktı adı zorunludur; model çağrısı yapmaz. Denetim kanıtı `reports/audit/current/`, güncel kaynak planları `reports/audit/plans/` içindedir. 2.000 çağrıyı aşan pilot veya training evaluation da gerçek, kapsamı doğrulanmış dilsel inceleme gerektirir. Eski Prompt10 makale/plan kayıtları tarihsel olarak saklanır. Dağıtım0.11.0 içinde toplam87 Python kaynak ve12 CLI vardır; katmanların veri/şema sürümleri kaynak hashleriyle birlikte kullanılır.

## İngilizce makale ve raporlama

[paper/manuscript.md](paper/manuscript.md) sekiz makale bölümünü birleştirir.
Yedi tablo CSV/LaTeX/Markdown biçimindedir. Veri istatistikleri gerçek donmuş
bankadan alınmıştır; model performansı, güven aralığı ve testler sonuç bekleyen
NA/null yer tutuculardır. Mimari ve veri üretim şemaları PDF/SVG/PNG olarak
vardır; sonuç gerektiren dört grafik üretilmemiştir.

[reports/README.md](reports/README.md) komutları ve kanıt sözleşmesini,
[reports/current/REPORT_STATUS.md](reports/current/REPORT_STATUS.md) beş RQ'nun
durumunu açıklar. Kaynak, girdi ve çıktı hashleri aynı rapor dizinindedir.
Yeni rapor için mevcut dizinleri ezmeden yeni hedef seçilir:

```sh
PYTHONPATH=src:reports python3 -m publication --paper-dir paper_next --report-dir reports/next
```

Kurulu eşdeğer `turkish-cbe-paper`'dır. Raporlama model çağrısı/eğitim yapmaz;
frozen veri ve eski çıktı/plan bytes korunur. Prompt11 gerekli yürütme/raporlama düzeltmeleri nedeniyle eski planlar güncel kaynakla yeniden üretilmelidir. Taslak,
gerçek model sonuçları ve insan dilsel incelemesi olmadan yayına hazır bir
ampirik çalışma sayılmaz.

## İsteğe bağlı eğitim

[TRAINING_COMPARISON.md](TRAINING_COMPARISON.md) altı kolu, üç bütçe profilini,
ödül/consistency ayrımını ve çalıştırma koşullarını tanımlar. Yerel M1/16 GiB
envanteri CUDA eğitim ortamını karşılamıyor; eğitim başlatılmadı. Ağırlık
indirmesi ve adapter yoktur. Planlar exact eğitim adımı/havuz sayılarını tutar;
ölçülmemiş GPU süresi, token tüketimi ve maliyet null kalır.

```sh
PYTHONPATH=src python3 -m training plan --config configs/training/pilot.toml --output results/training/plans/my_pilot.json
PYTHONPATH=src python3 -m training run --plan results/training/plans/my_pilot.json --arm sft_lora --output results/training/runs/my_sft
```

İki komut da varsayılan olarak eğitim yapmaz. Kurulu CLI `turkish-cbe-train`'dir.
65 eski kaynak ve donmuş banka korunur. Predicate holdout temiz olsa da
İngilizce bağlamda higher/lower örtüşmesi ve eksik Torch/GPU doğrulaması açıktır.

## Reasoning deneyleri

[REASONING_EXPERIMENTS.md](REASONING_EXPERIMENTS.md) A direct, B zero-shot
reasoning, C structured reasoning ve D modelin desteklediği native effort
kollarını tanımlar. Ortak latent/sorgu/sunum/dil koordinatları ve ortak output
bütçesi korunur; yalnız son cevap puanlanır. Doğruluk, R/invariance/kappa,
coverage, dört-geçerli-cevap duyarlılığı, token ve truncation tanıları hazırdır.
Mevcut modellerde native effort desteği doğrulanmadığından D hücreleri gerekçeli
unavailable kalır. Gerçek model çağrısı yapılmadı.

```sh
PYTHONPATH=src .venv/bin/python -m reasoning pilot --model qwen25-05b-hf --output results/reasoning/plans/my_pilot.json
PYTHONPATH=src .venv/bin/python -m reasoning analyse --output results/reasoning/my_analysis --figures
```

İlk komut yalnız kapalı örnek kayıtla dry-run yapar; model seçimi/etkinleştirmesi
değildir. Kurulu entry point `turkish-cbe-reasoning`'dir. Uygulama, donmuş
`experiments` kaynak hash'ini değiştirmemek için `src/reasoning/` içindedir;
[yerleşim kaydı](src/experiments/reasoning/README.md) istenen dizinde korunur.
Önceki 57 Python kaynağı ve donmuş veri bankası değişmedi.

## Sözcük ve tokenizer analizi

[LEXICAL_ANALYSIS.md](LEXICAL_ANALYSIS.md) kapsam ve yöntem sözleşmesidir.
CC BY-SA 4.0 lisanslı, sabit UD Turkish-BOUN r2.18 korpusunda 9.761 cümle
sayıldı. Katalog dokuz ilişki/18 yön ve 14 benzersiz lemma içerir. 216 sözlük
ve 1.458 pilot-önizleme tokenizer satırı üretildi. OpenAI ve üç Qwen
yapılandırmasında sayımlar mevcuttur; iki Llama tokenizer'ı erişim nedeniyle
null kalır. Düz metin, yerel chat template ve sağlayıcı usage kapsamları ayrıdır.

```sh
.venv/bin/python -m pip install --require-hashes -r requirements-lexical.lock
PYTHONPATH=src USE_TORCH=0 USE_TF=0 USE_FLAX=0 HF_HUB_OFFLINE=1 .venv/bin/python -m analysis.lexical_bias --root . --output results/lexical/my_measurement
```

Bu komut için aşağıdaki inference/tokenizer ve analysis kurulumları da gerekir.
Teslimdeki [lexical manifest](results/lexical/baseline/lexical_manifest.json)
yalnız gerçek korpus/tokenizer betimlemelerini içerir; model sonucu içermez.
Spearman, kontrollü regresyon, lexical/latent yön bias ve canonical/P2
karşılaştırmaları gerçek model kayıtlarına hazırdır. Frekans nedenselliği
veya model parametreleri hakkında çıkarım yapılmaz.

## CBE analizi

[ANALYSIS.md](ANALYSIS.md), metriklerin paydalarını, kayıt doğrulamasını,
McNemar/Holm, 2.000 tekrarlı latent-küme bootstrap ve Wilson aralıklarını
tanımlar. On iki istenen ölçüt, TR−EN farkları, dil×sunum/dil×karmaşıklık,
query kolları ve token–invalid ilişkisi uygulanmıştır. Modeller/sürümler,
isim kontrolleri, n, ilişkiler ve query kolları açık tabakalarda tutulur.
CSV/LaTeX tabloları ve altı grafik ailesinin PDF/SVG/PNG dışa aktarımı hazırdır.
Test çıktıları sentetiktir ve geçici klasörlerde kalır.

```sh
.venv/bin/python -m pip install --require-hashes -r requirements-analysis.lock
PYTHONPATH=src .venv/bin/python -m analysis --root . --output results/analysis/my_analysis
# Gerçek bir run oluştuğunda:
PYTHONPATH=src .venv/bin/python -m analysis --root . --run RUN_ID --output results/analysis/RUN_ID --figures
```

İlk komut dizisinden önce aşağıdaki temel inference/tokenizer kurulumu gerekir.
Gerçek kayıt yoksa analiz yalnız `no_model_results` durum manifesti yazar;
performans tablosu, p-değeri ve grafik oluşturmaz. Teslimdeki
[durum kaydı](results/analysis/no_model_results/analysis_manifest.json) bu durumu
açıkça belgeliyor. Gerçek sonuçlar henüz olmadığı için araştırma bulgusu yoktur.

## Inference katmanı

- OpenAI Responses REST API, yerel Hugging Face Transformers ve vLLM HTTP
  adaptörleri aynı çalışma sözleşmesini kullanır. Kodda model adı sabitlenmez.
- [configs/models.yaml](configs/models.yaml): altı kapalı örnek; OpenAI,
  Qwen2.5 0.49/1.54B, Llama3.2 1.23/3.21B ve vLLM Qwen örneği. Ağırlık/tokenizer
  commit'leri, chat template hash'i, tarih, altyapı, sampling/reasoning/seed,
  context/output sınırları ve fiyat metadatası burada tutulur. Llama template
  erişimi gated; null hash ile açılamaz. vLLM sunucu sürümü ayrıca seçilmelidir.
- Türkçe tam EVET/HAYIR, İngilizce tam YES/NO ayrıştırılır. Dış boşluklar
  dışında biçim esnetilmez. Açıklama, noktalama, JSON, çelişki, yanlış cevap
  sözcüğü dili, refusal ve kesilmiş çıktı invalid olarak korunur.
- Bütün modeller aynı latent cohort ve canonical/equivalent çiftlerini alır.
  Endpoint/internal/distance kolları ayrı metadata ve tabakalarda tutulur.
- Ham kayıtlar atomik ve exclusive yazılır. Checkpoint/resume tamamlanmış
  yanıtı yeniden göndermez. Sonucu belirsiz timeout/crash çağrıları otomatik
  tekrarlanmaz; açık, denetlenebilir retry/abandon kararı gerekir.
- Hız sınırı endpoint/credential adı düzeyinde SQLite ile paylaşılır. Yalnız
  kesin reddedilmiş geçici hatalarda sınırlı exponential backoff uygulanır.
  Invalid cevabı iyileştirmek için yeniden çağrı yapılmaz. Cache varsayılan
  kapalıdır; cache kaynağı gösterilir ve bağımsız yeni gözlem sayılmaz.

## Kurulum

Python >=3.11,<3.14; test ortamları 3.12.14 ve 3.13.1'dir. CPython 3.12/macOS
arm64 için aşağıdaki çevrimdışı bağımlılık kurulumu uygundur; PyYAML için
CPython 3.13 wheel'i de teslimdedir. Diğer platformlarda aynı kilit dosyalarını
--no-index/--find-links olmadan kullanın. Varsayılan yapılandırmada model
ağırlıkları indirilmez; API anahtarı ortam değişkeninden okunur.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --no-index --find-links vendor/inference --require-hashes -r requirements-inference.lock
.venv/bin/python -m pip install --no-index --find-links vendor/tokenizers --require-hashes -r requirements-tokenizers.lock
.venv/bin/python -m pip install --require-hashes -r requirements-build.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

OpenAI/vLLM HTTP istemcileri standart kütüphaneyi kullanır; gizli SDK retry'ı
bulunmaz. Yerel Transformers için ayrı `.[transformers]` kurulumu gerekir;
bu geniş uyumluluk aralığı GPU/CPU ortamının tam kilidi değildir. Gerçek
çalışmadan önce seçilen platformun torch/transformers/CUDA ortamını ayrıca
sabitleyin. Bu teslimde ağır kütüphaneler veya model ağırlıkları kurulmadı;
üç adaptörün sözleşmeleri çevrimdışı fixture testleriyle denetlendi.

## Dry-run

Proje kökünde açık model ID'leriyle yeni bir plan oluşturun:

```sh
.venv/bin/python scripts/run_pilot.py --model qwen25-05b-hf llama32-1b-hf --output logs/my_pilot_plan.json
.venv/bin/python scripts/run_full.py --model qwen25-05b-hf llama32-1b-hf --output logs/my_full_plan.json
```

Varsayılan küçük pilot n=4/8, boyut başına dört dengeli latent sıra, bir ilişki
ve EN/TR canonical/equivalent koşullarıdır: **32 çağrı/model**. Varsayılan
full, testte dokuz n ve 200 sıra/n kullanır: **7.200 çağrı/model**. Kollar,
ilişkiler, isim kontrolleri, boyutlar, tekrar ve hız/bütçe parametreleri CLI
ile açık seçilir; bütçe aşılınca cohort sessizce küçültülmez.

Teslimdeki [pilot planı](logs/pilot_dry_run.json) altı örnek için 192 çağrı,
[full planı](logs/full_dry_run.json) 43.200 çağrı planlar; gerçek çağrı sayısı
sıfırdır. Bütün örnek modeller kapalıdır. Full plan ayrıca varsayılan 2.000
çağrı bütçesini aşar ve gerçek pilot/onay olmadan çalışmaz.

Token planı önceki kaynak prompt BPE sayısı üzerine açıkça etiketlenmiş
64 token/çağrı heuristiği ekler. Bu, yeni sistem/kullanıcı mesajlarının gerçek
chat sayısı veya kanıtlanmış üst sınır değildir; HF/vLLM için model tokenizer'ı
ölçümü de değildir. Gerçek yürütmede mesajlar sağlayıcının gerçek chat
tokenizer/count endpoint'iyle ölçülür; context taşması kırpılmaz.

Tarihli fiyat varsa tahmin hesaplanır. OpenAI tarihsel GPT-4o örneğinin
standart tarifesi [resmî model kartından](https://developers.openai.com/api/docs/models/gpt-4o)
2026-10-24'te kontrol edildi: milyon input/output/cached input token başına
USD 2.50/10.00/1.25. Bu örnek deprecated'dir; hesap erişimi veya çalışabilirlik
iddiası değildir. Diğer fiyatlar ve bütün toplam donanım maliyeti bilinmiyor;
null sıfır maliyet demek değildir. Tarife/tahmin gerçek fatura değildir.

## Gerçek pilot ve full yürütme

Model seçimi, erişim ve çalıştırma kararı verildikten sonra seçilen örneği
uygun revision/template/fiyatla yapılandırıp enabled=true yapın; **yeni bir
dry-run planı** oluşturun. Anahtarın kendisini dosyaya koymayın;
api_key_env yalnız ortam değişkeni adıdır. Önceden kaydedilmiş planla:

```sh
.venv/bin/python scripts/run_pilot.py --plan logs/my_enabled_pilot_plan.json --execute --run-id pilot-selected-models
```

Aynı komut/ID resume eder; bitmiş çağrılar tekrar gönderilmez. Kod, ortam,
model, cohort veya bütçe değişmişse yeni plan gerekir. Belirsiz çağrı için
CLI `resolve` bir gerekçe ve retry seçilirse duplicate-risk acknowledgement
ister; ham kayıt değişmez. Rastgele başka run ID açmak resume değildir.

Full için tamamlanmış gerçek, cache kullanılmamış pilot ve veri manifestine
bağlı insan dilsel onay artefaktı gerekir. `approve-pilot` kullanıcı tarafından
sağlanan onayı kayda geçirir; insan değerlendirmesi yapmaz. Reviewer dosyası
status=approved, reviewers ve dataset_manifest_sha256 alanlarını taşımalı ve
frozen veri kapsamını gerçekten onaylamalıdır. Fixture veya dry-run onaylanamaz.

```sh
PYTHONPATH=src .venv/bin/python -m inference approve-pilot --run-id pilot-selected-models --approved-by RESEARCHER --linguistic-review review.json --output logs/pilot_approval.json
.venv/bin/python scripts/run_full.py --plan logs/my_enabled_full_plan.json --execute --pilot-approval logs/pilot_approval.json
```

Bunlar gelecekteki yürütme komutlarıdır; bu teslimde çalıştırılmadı. Full
planın bütçesi ayrıca açıkça gözden geçirilmelidir. Onay; dataset/model/ayar,
prompt/parser sürümü ve deney eksenlerine bağlanır, başka tasarıma taşınmaz.

## Dosya düzeni ve önceki katmanlar

```text
src/inference/      config, dataset, planning, prompts, parser, providers,
                    storage, runtime, runner, cli, __main__, __init__
src/analysis/       metrics, statistics, io, pipeline, reporting, __main__, __init__
src/analysis/lexical_bias/ corpus, catalog, tokenization, bias, statistics, CLI
configs/models.yaml
scripts/run_pilot.py, scripts/run_full.py
results/raw/        değişmez run/attempt/HTTP kayıtları; bu teslimde model kaydı yok
results/normalized/ yalnız eklenen JSONL; bu teslimde model sonucu yok
logs/               dry-run planları ve gelecekte türetilmiş checkpoint/rate DB
results/analysis/   yalnız analiz kökeni/durum kaydı; bu teslimde model sonucu yok
analysis/lexical_bias/ paylaşılabilir relation kataloğu ve kapsam açıklaması
results/lexical/    bağımsız corpus/tokenizer ölçümleri; model sonucu yok
figures/lexical/    gerçek performans verisi olmadığı için durum açıklaması
```

Dağıtım/lexical 0.7.0; CBE analiz 0.6.0, inference 0.5.0, frozen grid 0.4.1,
matched core 0.4.0, TR renderer 0.3.0, generator 0.2.0'dır. Önceki 50 Python
kaynağı, seedler ve veri dosyaları korunur.
Kaynak makale ve Türkçe uzantılar protokolde ayrıdır; yayımlanmış model
performansının birebir replikasyonu iddiası yoktur.

```sh
PYTHONPATH=src python3 -m complete_grid.dataset_builder verify --manifest data/complete_grid/manifest.json --hashes-only
PYTHONPATH=src python3 -m data_generation verify dataset_manifest.json --replay
PYTHONPATH=src python3 -m experiments.dataset_builder verify --manifest experiment_dataset_manifest.json --hashes-only
```

Tam faktör veri şeması ve JSONL dışa aktarım komutları DATASET_CARD.md ve
REPRODUCIBILITY.md içindedir. Türkçe 252 örneklik insan raporunun incelemeci
alanları boş kalır. Model seçimi/gerçek pilot ve insan onayı bekliyor.
Analiz altyapısı hazırdır; gerçek model verisinin istatistiksel değerlendirmesi
henüz yapılmadı. Bir sonraki prompta kendiliğinden geçilmedi.
