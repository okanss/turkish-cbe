# Normalleştirilmiş sonuçlar

Gerçek inference henüz çalıştırılmadı; model sonucu yoktur. Her run için
JSONL dosyası yalnız ekleme yapılarak büyür. Kesin parser sonucu, invalid
nedeni, cevap dili, kaynak/model/message hashleri, usage ve maliyet burada
saklanır. Checkpoint outbox yarım kalmış satırı yalnız eksik byte'ları
ekleyerek tamamlayabilir; ham çıktıyı değiştirmez. Cache cevabı kaynak
çağrıya bağlıdır ve bağımsız model gözlemi sayılmaz.
