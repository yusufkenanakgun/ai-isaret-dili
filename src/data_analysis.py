"""
ASL Alphabet Dataset Analizi
- Sınıf sayısı ve isimleri
- Her sınıftaki görüntü sayısı
- Görüntü boyutları
- Örnek görselleştirme
"""

import os
from pathlib import Path
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Dataset yolu (iç içe klasör yapısı)
TRAIN_PATH = Path("data/asl-alphabet/asl_alphabet_train/asl_alphabet_train")
TEST_PATH = Path("data/asl-alphabet/asl_alphabet_test/asl_alphabet_test")

def analyze_dataset():
    print("=" * 50)
    print("ASL ALPHABET DATASET ANALİZİ")
    print("=" * 50)

    # Sınıfları bul
    classes = sorted([d.name for d in TRAIN_PATH.iterdir() if d.is_dir()])
    print(f"\nSınıf sayısı: {len(classes)}")
    print(f"Sınıflar: {classes}")

    # Her sınıftaki görüntü sayısı
    print("\n" + "-" * 50)
    print("SINIF DAĞILIMI")
    print("-" * 50)

    total_images = 0
    class_counts = {}

    for cls in classes:
        cls_path = TRAIN_PATH / cls
        count = len(list(cls_path.glob("*.jpg"))) + len(list(cls_path.glob("*.png")))
        class_counts[cls] = count
        total_images += count
        print(f"{cls}: {count} görüntü")

    print(f"\nToplam: {total_images} görüntü")

    # Görüntü boyutlarını kontrol et
    print("\n" + "-" * 50)
    print("GÖRÜNTÜ BOYUTLARI")
    print("-" * 50)

    sample_class = classes[0]
    sample_path = TRAIN_PATH / sample_class
    sample_images = list(sample_path.glob("*"))[:5]

    sizes = set()
    for img_path in sample_images:
        try:
            with Image.open(img_path) as img:
                sizes.add(img.size)
        except:
            pass

    print(f"Örnek boyutlar: {sizes}")

    # Test seti kontrolü
    print("\n" + "-" * 50)
    print("TEST SETİ")
    print("-" * 50)

    if TEST_PATH.exists():
        test_images = list(TEST_PATH.glob("*"))
        print(f"Test görüntü sayısı: {len(test_images)}")
    else:
        print("Test klasörü bulunamadı")

    return classes, class_counts

if __name__ == "__main__":
    classes, counts = analyze_dataset()
