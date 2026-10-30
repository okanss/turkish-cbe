# Reasoning deneyleri: kaynak yerleşimi

Uygulama [src/reasoning](../../reasoning/) paketindedir; komut
`python -m reasoning` / `turkish-cbe-reasoning` üzerinden çalışır.

Prompt 4 kaynak kimliği `experiments/` altındaki bütün Python dosyalarını
özyinelemeli olarak kapsar. Bu dizine yeni Python kodu eklemek, önceki kaynak
baytları değişmese bile donmuş veri bankasının doğrulamasını bozar. Prompt 8
entegrasyon testi bunu yakaladı. Eski 57 Python kaynağını, veri manifestlerini,
SQLite bankasını ve önceki dry-run planlarını aynen korumak için yeni kod ayrı
pakette tutulur. İstenen bu dizin yerleşim kararını kaydeder. Veri üreticiye
istisna, runtime monkeypatch, hash gevşetmesi veya veri yeniden üretimi yoktur.

Protokol ve kullanım: [REASONING_EXPERIMENTS.md](../../../REASONING_EXPERIMENTS.md).
