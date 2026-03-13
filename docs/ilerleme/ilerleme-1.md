# İlerleme Raporu #1
**Tarih:** 13 Mart 2026
**Proje:** AI Destekli İşaret Dili Tanıma Sistemi
**Öğrenci:** Yusuf Kenan Akgün

---

## Tamamlanan Fazlar

### FAZ 1: Ortam Kurulumu ✓

| Bileşen | Versiyon | Açıklama |
|---------|----------|----------|
| Python | 3.x | Sanal ortam (venv) |
| PyTorch | 2.6.0+cu124 | CUDA destekli, GPU eğitimi |
| torchvision | 0.21.0+cu124 | Pretrained modeller (EfficientNet) |
| MediaPipe | 0.10.32 | El landmark tespiti |
| OpenCV | 4.13.0 | Görüntü işleme, kamera |
| NumPy | 2.4.3 | Array işlemleri |
| scikit-learn | - | Metrikler, split |
| matplotlib, seaborn | - | Görselleştirme |

**GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM)
**CUDA:** 12.4 (Driver 13.0 uyumlu)

**Klasör Yapısı:**
```
ai-isaret/
├── data/           → Veri setleri
├── models/         → Eğitilmiş modeller
├── notebooks/      → Jupyter notebook'lar
├── src/            → Kaynak kodları
├── docs/           → Dokümantasyon
├── demo/           → Demo uygulaması
├── requirements.txt
└── .gitignore
```

---

### FAZ 2: Veri Seti Hazırlığı ✓

**Dataset:** ASL Alphabet (Kaggle)

| Özellik | Değer |
|---------|-------|
| Toplam görüntü | 87,000 |
| Sınıf sayısı | 29 (A-Z + del, nothing, space) |
| Sınıf başına | 3,000 görüntü (dengeli) |
| Görüntü boyutu | 200×200 piksel |

**Train/Val/Test Split:**

| Set | Görüntü | Oran |
|-----|---------|------|
| Train | 60,900 | %70 |
| Validation | 13,050 | %15 |
| Test | 13,050 | %15 |

**Augmentation (Train için):**
- RandomHorizontalFlip (p=0.3)
- RandomRotation (±15°)
- ColorJitter (brightness, contrast)
- Resize → 224×224 (EfficientNet için)
- ImageNet normalization

**Oluşturulan Dosyalar:**
- `src/data_analysis.py` — Dataset analizi
- `src/data_split.py` — Train/Val/Test split
- `src/dataset.py` — PyTorch DataLoader ve augmentation

---

## Kalan Fazlar

### FAZ 3: EfficientNet-B0 Transfer Learning (Sıradaki)
- [ ] Pretrained EfficientNet-B0 yükleme
- [ ] Son katmanı 29 sınıfa göre değiştirme
- [ ] Önce son katman eğitimi (feature extraction)
- [ ] Sonra tüm model fine-tuning
- [ ] Hyperparameter denemeleri (lr, batch_size, optimizer)
- [ ] Training loop (loss tracking, early stopping)
- [ ] Sonuçlar: accuracy, F1-score, confusion matrix

### FAZ 4: MLP Landmark-Based Model
- [ ] MediaPipe ile tüm görüntülerden landmark çıkarma
- [ ] 21 nokta × 3 koordinat = 63 boyutlu vektör
- [ ] Landmark normalizasyonu
- [ ] MLP mimarisi tasarımı
- [ ] Eğitim ve değerlendirme
- [ ] Detection failure rate ölçümü

### FAZ 5: Karşılaştırmalı Analiz
- [ ] İki modelin metriklerde karşılaştırması
- [ ] Top-1/Top-5 accuracy
- [ ] Per-class F1-score
- [ ] Confusion matrix analizi
- [ ] Model boyutu ve inference hızı (FPS)

### FAZ 6: LSTM Kelime Tanıma
- [ ] WLASL dataset indirme
- [ ] Video → frame extraction
- [ ] Frame başına landmark → sequence
- [ ] LSTM/GRU model eğitimi

### FAZ 7: Demo Uygulama
- [ ] Python + OpenCV webcam demo
- [ ] Gerçek zamanlı harf tanıma
- [ ] (Opsiyonel) React Native mobil demo

### FAZ 8: Tez Yazımı
- [ ] Introduction, Related Work
- [ ] Methodology, Implementation
- [ ] Experimental Results
- [ ] Discussion, Conclusion

### FAZ 9: Sunum
- [ ] 15-20 slide sunum
- [ ] Canlı demo hazırlığı
- [ ] Prova

---

## Notlar

### Data Leakage Hakkında
Dataset tek kaynaktan geldiği için person-wise split yapılamadı. Random split kullanıldı. Gerçek generalization performansı demo'da (yeni el görüntüleri ile) test edilecek. Bu durum tezde "Limitations" bölümünde belirtilecek.

### GitHub
Repo: https://github.com/yusufkenanakgun/ai-isaret-dili

---

*Son güncelleme: 13 Mart 2026*
