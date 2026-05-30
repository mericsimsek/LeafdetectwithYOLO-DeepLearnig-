import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sys

# Ayarlar
MODEL_PATH = 'weights/resnet18_plant_disease.pth'
IMAGE_PATH = 'train_images/1.jpg' # Test etmek istediğin resim

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Modeli Yükle
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint['classes']

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(DEVICE)
model.eval()

# Resmi İşle
preprocess = transforms.Compose([
    transforms.Resize(176),
    transforms.CenterCrop(160),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

# Tahmin
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, predicted_idx = torch.max(probabilities, 0)

predicted_class = class_names[predicted_idx.item()]
confidence_score = confidence.item() * 100

print(f"\n🌱 Tespit Edilen Sınıf: {predicted_class}")
print(f"🎯 Modelin Eminlik Skoru: %{confidence_score:.2f}\n")