import os
import yaml
import random
import shutil
import kagglehub
from ultralytics import YOLO

print("Kagglehub önbelleğinden veri aranıyo")
download_path = kagglehub.dataset_download("sebastianpalaciob/plantvillage-for-object-detection-yolo")

ORIGINAL_DATASET = ''
for root, dirs, files in os.walk(download_path):
    if 'images' in dirs and 'labels' in dirs:
        ORIGINAL_DATASET = root
        break

if not ORIGINAL_DATASET:
    raise ValueError("Orijinal veri seti bulunamadı")


print("\nÇalışma ortamı kuruluyor ve veriler bölünüyor")
WORKING_DIR = './yolo_plantvillage'

if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR) 
os.makedirs(WORKING_DIR, exist_ok=True)

for split in ['train', 'val']:
    os.makedirs(os.path.join(WORKING_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(WORKING_DIR, 'labels', split), exist_ok=True)

orig_images_dir = os.path.join(ORIGINAL_DATASET, 'images')
orig_labels_dir = os.path.join(ORIGINAL_DATASET, 'labels')

all_images = [f for f in os.listdir(orig_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.seed(42)
random.shuffle(all_images)

split_index = int(len(all_images) * 0.8)
train_images = all_images[:split_index]
val_images = all_images[split_index:]

def fast_copy(image_list, split_type):
    for img_name in image_list:
        txt_name = os.path.splitext(img_name)[0] + '.txt'
        orig_img_path = os.path.join(orig_images_dir, img_name)
        orig_txt_path = os.path.join(orig_labels_dir, txt_name)
        if os.path.exists(orig_txt_path):
            shutil.copy(orig_img_path, os.path.join(WORKING_DIR, 'images', split_type, img_name))
            shutil.copy(orig_txt_path, os.path.join(WORKING_DIR, 'labels', split_type, txt_name))

fast_copy(train_images, 'train')
fast_copy(val_images, 'val')

classes_yaml_path = [os.path.join(root, f) for root, dirs, files in os.walk(download_path) for f in files if f.endswith('yaml')][0]
with open(classes_yaml_path, 'r') as f:
    class_data = yaml.safe_load(f)

yaml_path = os.path.join(WORKING_DIR, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump({
        'train': os.path.abspath(os.path.join(WORKING_DIR, 'images', 'train')),
        'val': os.path.abspath(os.path.join(WORKING_DIR, 'images', 'val')),
        'nc': len(class_data['names']), 
        'names': class_data['names']
    }, f, default_flow_style=False)

print("✔ Veri hazırlığı tamamlandı")


if __name__ == '__main__':
    print("\n🚀 Gerçek Dünya (Robust) Eğitimi Başlıyo\n")
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data=yaml_path,         
        epochs=50,              
        imgsz=416,              
        batch=16,               # Yerel bilgisayar optimizasyonu
        patience=10,            
        optimizer='AdamW',
        weight_decay=0.01,
        lr0=0.002,              
        cos_lr=True,
        dropout=0.15,           
        hsv_h=0.015,            
        hsv_s=0.7,              
        hsv_v=0.4,              
        degrees=15.0,           
        shear=2.0,              
        scale=0.5,              
        fliplr=0.5,             
        mosaic=0.5,             
        mixup=0.1,              
        project='runs',
        name='plantvillage_robust'
    )