"""
ASL Alphabet Dataset - PyTorch DataLoader
Görüntüleri yükler, augmentation uygular, modele besler
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

# Sınıf isimleri (alfabetik sıra)
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Sınıf -> index mapping
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


class ASLDataset(Dataset):
    """ASL Alphabet Dataset"""

    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: train, val veya test klasörü yolu
            transform: uygulanacak dönüşümler
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        # Tüm görüntüleri topla
        for cls in CLASSES:
            cls_path = self.data_dir / cls
            if cls_path.exists():
                for img_path in cls_path.glob("*.jpg"):
                    self.samples.append((img_path, CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(mode='train', img_size=224):
    """
    Augmentation ve normalizasyon dönüşümleri

    Args:
        mode: 'train' (augmentation var) veya 'val'/'test' (augmentation yok)
        img_size: çıktı görüntü boyutu (EfficientNet için 224)
    """
    # ImageNet normalizasyon değerleri (pretrained model için)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.3),           # Yatay çevirme
            transforms.RandomRotation(15),                     # ±15° döndürme
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Parlaklık/kontrast
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])


def get_dataloaders(data_root='data/asl-split', batch_size=32, img_size=224, num_workers=4):
    """
    Train, val ve test DataLoader'larını döndürür
    """
    train_transform = get_transforms('train', img_size)
    val_transform = get_transforms('val', img_size)

    train_dataset = ASLDataset(f"{data_root}/train", transform=train_transform)
    val_dataset = ASLDataset(f"{data_root}/val", transform=val_transform)
    test_dataset = ASLDataset(f"{data_root}/test", transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# Test
if __name__ == "__main__":
    print("DataLoader test ediliyor...\n")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32, num_workers=0)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Bir batch al
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")  # [32, 3, 224, 224]
    print(f"Labels: {labels[:5]}")
    print(f"Label names: {[IDX_TO_CLASS[l.item()] for l in labels[:5]]}")
