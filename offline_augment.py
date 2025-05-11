import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1️⃣ Ayarlar
SRC_ROOT = 'dataset'           # Orijinal el crop’larının olduğu klasör
DST_ROOT = 'augmented_dataset' # Oluşacak augment’lı klasör
TARGET_PER_CLASS = 800         # Her harf için istediğin örnek sayısı
IMG_SIZE = (128, 128)          # Yeniden boyutlandırma

# 2️⃣ ImageDataGenerator parametreleri
datagen = ImageDataGenerator(
    rotation_range=20,           # ±20° döndürme
    width_shift_range=0.1,       # %10 yatay kaydırma
    height_shift_range=0.1,      # %10 dikey kaydırma
    zoom_range=0.1,              # %10 yakınlaştırma/uzaklaştırma
    brightness_range=(0.7,1.3),  # parlaklık varyasyonu
    shear_range=5,               # 5° kaydırmalı kesim
    fill_mode='nearest'          # yeni pikselleri en yakın komşudan al
)

# 3️⃣ Hedef klasör yapısını oluştur
os.makedirs(DST_ROOT, exist_ok=True)
for letter in os.listdir(SRC_ROOT):
    os.makedirs(os.path.join(DST_ROOT, letter), exist_ok=True)

# 4️⃣ Her harf için kopyala + augment
for letter in sorted(os.listdir(SRC_ROOT)):
    src_dir = os.path.join(SRC_ROOT, letter)
    dst_dir = os.path.join(DST_ROOT, letter)

    # 4a. Mevcut imajları kopyala
    images = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]
    for f in images:
        img = cv2.imread(os.path.join(src_dir, f))
        cv2.imwrite(os.path.join(dst_dir, f), img)

    count = len(images)
    print(f"[{letter}] Başlangıç örnek sayısı: {count}")

    # 4b. Augment ile tamamla
    idx = 0
    for f in images:
        if count >= TARGET_PER_CLASS:
            break
        img = cv2.imread(os.path.join(src_dir, f))
        x = cv2.resize(img, IMG_SIZE)
        x = x.reshape((1,) + x.shape)  # datagen için batch boyutu ekle

        for batch in datagen.flow(x, batch_size=1):
            out_name = f"{letter}_aug_{idx}.jpg"
            cv2.imwrite(os.path.join(dst_dir, out_name), batch[0].astype(np.uint8))
            idx += 1
            count += 1
            if count >= TARGET_PER_CLASS:
                break

    print(f"[{letter}] Nihai örnek sayısı: {count}\n")

print("🎉 Tüm sınıflar offline olarak augment edildi! 🎉")
