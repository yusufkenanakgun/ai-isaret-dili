"""
ASL Alphabet Dataset - Train/Val/Test Split
Dosyaları fiziksel olarak ayırır
%70 Train / %15 Validation / %15 Test
"""

import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Kaynak ve hedef yolları
SOURCE_PATH = Path("data/asl-alphabet/asl_alphabet_train/asl_alphabet_train")
OUTPUT_PATH = Path("data/asl-split")

# Split oranları
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def create_split():
    print("Dataset bölünüyor...")
    print(f"Oranlar: Train {TRAIN_RATIO*100:.0f}% / Val {VAL_RATIO*100:.0f}% / Test {TEST_RATIO*100:.0f}%")
    print(f"Hedef klasör: {OUTPUT_PATH}\n")

    # Her sınıf için
    classes = sorted([d.name for d in SOURCE_PATH.iterdir() if d.is_dir()])
    total_train, total_val, total_test = 0, 0, 0

    for cls in classes:
        cls_path = SOURCE_PATH / cls
        images = list(cls_path.glob("*.jpg"))

        # Hedef klasörleri oluştur
        for split in ['train', 'val', 'test']:
            (OUTPUT_PATH / split / cls).mkdir(parents=True, exist_ok=True)

        # Split yap
        train_imgs, temp_imgs = train_test_split(
            images, train_size=TRAIN_RATIO, random_state=42, shuffle=True
        )
        val_imgs, test_imgs = train_test_split(
            temp_imgs, train_size=0.5, random_state=42, shuffle=True
        )

        # Kopyala
        for img in train_imgs:
            shutil.copy2(img, OUTPUT_PATH / 'train' / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, OUTPUT_PATH / 'val' / cls / img.name)
        for img in test_imgs:
            shutil.copy2(img, OUTPUT_PATH / 'test' / cls / img.name)

        total_train += len(train_imgs)
        total_val += len(val_imgs)
        total_test += len(test_imgs)
        print(f"{cls}: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    print("\n" + "=" * 50)
    print("ÖZET")
    print("=" * 50)
    print(f"Train: {total_train} görüntü ({total_train/870:.1f}%)")
    print(f"Val:   {total_val} görüntü ({total_val/870:.1f}%)")
    print(f"Test:  {total_test} görüntü ({total_test/870:.1f}%)")
    print(f"\nKlasör: {OUTPUT_PATH.absolute()}")

if __name__ == "__main__":
    create_split()
