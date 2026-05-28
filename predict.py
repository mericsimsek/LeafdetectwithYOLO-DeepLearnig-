from ultralytics import YOLO

# 1. Eğitilmiş Modeli Yükle
model = YOLO('weights/best.pt')

# 2. Test Edilecek Resmin Yolunu Belirle 
# (Kendin internetten indirdiğin herhangi bir hasta yaprak fotoğrafı olabilir)
test_resmi = 'test_images/ornek_hastali_yaprak.jpg' 

# 3. Tahmin Yap ve Sonucu Göster
print("Tahmin yapılıyor...")
results = model.predict(source=test_resmi, save=True, show=True)

print("İşlem tamam! Sonuçlar 'runs/detect/predict' klasörüne kaydedildi.")