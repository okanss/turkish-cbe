# Değiştirilemez ham kayıtlar

Gerçek inference henüz çalıştırılmadı; bu klasörde model cevabı yoktur.
Yürütmede her run ayrı manifest, execution.json ve benzersiz attempt
klasörlerinde request/response/error JSON kayıtları alır. Dosyalar exclusive
ve atomik oluşturulur; mevcut ham kayıtlar değiştirilmez. HTTP metadata ve
token sayımı istekleri de benzersiz kimlikle saklanır; API anahtarı
değerleri kaydedilmez. Fixture testleri geçici dizinlerdedir.
