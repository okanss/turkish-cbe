# Planlar ve checkpoint

pilot_dry_run.json ve full_dry_run.json dış çağrı yapmadan oluşturulmuş
planlardır; gerçek pilot veya model sonuçları değildir. Seçilen bütün
modeller aynı latent cohort'u kullanır. Çağrı bütçeleri tam seçimden
hesaplanır; token tahmini ve bilinmeyen chat framing alanları açıkça
etiketlidir. Gelecekte run SQLite dosyaları türetilmiş checkpoint'i, lock
dosyaları süreç kilidini ve rate SQLite dosyaları ortak hız pencerelerini
tutar. Bunlar results/raw değişmez kayıtlarının yerine geçmez.
