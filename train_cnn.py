import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. AYARLAR VE MODEL YÜKLEME
# ==========================================
# Cihazda GPU varsa kullan, yoksa CPU'da çalıştır
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_PATH = 'weights/resnet18_fast_plant.pth'
TEST_DIR = 'test_images'

print(f"Kullanılan Donanım: {DEVICE}")
print("Model yükleniyor, lütfen bekleyin...")

# Ağırlıkları ve Sınıf İsimlerini Yükle
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint['classes']

# ResNet18 Mimarisini Kur
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# ==========================================
# 2. ÖN İŞLEME VE TAHMİN (INFERENCE)
# ==========================================
preprocess = transforms.Compose([
    transforms.Resize(176),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("\n" + "="*50)
print("🔍 KLASÖRDEKİ RESİMLER ÜZERİNDE TAHMİN")
print("="*50)

# test_images klasöründeki tüm resimleri tara
if not os.path.exists(TEST_DIR):
    print(f"Hata: '{TEST_DIR}' klasörü bulunamadı.")
else:
    for filename in os.listdir(TEST_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(TEST_DIR, filename)
            
            # Resmi aç ve tensöre çevir
            image = Image.open(img_path).convert('RGB')
            input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)
            
            # Tahmin Yap
            with torch.no_grad():
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                confidence, predicted_idx = torch.max(probabilities, 0)
                
            predicted_class = class_names[predicted_idx.item()]
            confidence_score = confidence.item() * 100
            
            # Sınıf adını okunaklı hale getir
            try:
                parts = predicted_class.split('___')
                plant = parts[0].replace('_', ' ')
                disease = parts[1].replace('_', ' ')
            except:
                plant = predicted_class
                disease = "Bilinmiyor"

            print(f"\n📁 Dosya: {filename}")
            print(f"🌱 Bitki Türü: {plant}")
            print(f"🦠 Hastalık  : {disease}")
            print(f"🎯 Eminlik   : %{confidence_score:.2f}")

print("\n🏆 Tahmin işlemleri tamamlandı!")