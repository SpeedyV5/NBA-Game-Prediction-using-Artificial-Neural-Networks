# NBA Maç Sonucu Tahmini - Sunum Taslağı

**CENG 481 - Yapay Sinir Ağları Dersi Projesi**

**Ekip:**
- İbrahim Ersan Özdemir (202211054)
- Ekin Efe Kızıloğlu (202211043)

---

## Slide 1: Başlık + Ekip

**Başlık:**
# NBA Game Prediction using Artificial Neural Networks

**Alt Başlık:**
Win/Loss Classification and Score Difference Regression

**Ekip:**
- İbrahim Ersan Özdemir (202211054)
- Ekin Efe Kızıloğlu (202211043)

**Kurum:**
CENG 481 - Yapay Sinir Ağları Dersi

**Tarih:**
Aralık 2025

---

## Slide 2: Problem

### Problem Tanımı
- **Görev 1**: Ev sahibi takımın maçı kazanıp kazanmayacağını tahmin etme (Binary Classification)
- **Görev 2**: Maç skor farkını tahmin etme (Regression)

### Motivasyon
- Spor analitiği alanında önemli bir problem
- Spor bahis endüstrisi için değerli bilgiler
- Takım performans analizlerinde kullanım
- Fan engagement uygulamaları

### Zorluklar
- Çok sayıda değişken (takım istatistikleri, rest days, schedule, vb.)
- Zaman serisi özellikleri (takım formu, geçmiş performans)
- Non-linear ilişkiler

---

## Slide 3: Veri & Özellikler

### Veri Kaynakları
- **NocturneBear**: 2010-2024 box scores (~17,000+ maç)
- **NBAstuffer**: 2025-2026 takım/oyuncu istatistikleri
- **Toplam**: 17,832 maç, 203 feature

### Feature Engineering
- **Takım İstatistikleri**: PPG, oPPG, PACE, Efficiency, Win%, vb.
- **Rest Days Features**: 1-3 gün dinlenme, B2B, 3IN4, 4IN5 senaryoları
- **Rolling Windows**: 5 ve 10 maçlık win rate, score diff ortalamaları
- **ELO Ratings**: Takım güç değerlendirmesi (base: 1500, K: 20)
- **Diff Features**: Home-away takım feature farkları
- **Tarih Features**: Ay, haftanın günü, hafta sonu, playoff

### Dataset Split
- **Random Shuffle**: Train (70%) / Val (15%) / Test (15%)
- **Random State**: 30 (reproducibility)

---

## Slide 4: Modeller

### 1. MLP (Multi-Layer Perceptron)

**Classifier (MLP_C3 - Best):**
- 4 katman: [512, 256, 128, 64]
- BatchNorm + Dropout (0.3)
- Adam optimizer (lr=0.0005)
- Early stopping (patience=10)

**Regressor (MLP_R2 - Best):**
- 3 katman: [256, 128, 64]
- Dropout (0.1)
- Huber loss (delta=5.0)

### 2. Sequence LSTM

- LSTM tabanlı sequence model
- Sequence length: N=10 (N=5 ve N=10 denendi)
- Real sequence: Takım bazında tarih sırasına göre
- Leakage önleme: Date bucket yaklaşımı

### 3. Baseline: Gradient Boosting Machine

- Sklearn GradientBoostingClassifier/Regressor
- Default hyperparameters
- Referans model olarak kullanıldı

---

## Slide 5: Sonuçlar

### Classification Metrikleri (Test Set)

| Model | Accuracy | F1 | ROC-AUC | LogLoss | Brier |
|-------|----------|----|---------|---------|-------|
| **MLP** | 0.6665 | 0.7257 | **0.7108** | 0.6143 | 0.2124 |
| **LSTM** | 0.6618 | 0.7208 | 0.7008 | 0.6202 | 0.2155 |
| **Baseline GBM** | 0.6680 | 0.7303 | **0.7176** | 0.6099 | 0.2106 |

**En İyi Model**: Baseline GBM (ROC-AUC: 0.7176)

### Regression Metrikleri (Test Set)

| Model | MAE | RMSE |
|-------|-----|------|
| **MLP** | 10.06 | 12.91 |
| **Baseline GBM** | 10.04 | 12.87 |

**Performans**: Çok yakın (MAE ~10 puan)

### Figürler

1. **ROC-AUC Karşılaştırması**: Tüm modeller > 0.70
2. **Calibration Plot**: Brier score karşılaştırması
3. **Confusion Matrix**: MLP C3 (en iyi MLP varyantı)
4. **Regression Comparison**: MAE ve RMSE karşılaştırması
5. **Loss Curves**: MLP C3 training history

---

## Slide 6: Tartışma & Future Work

### Ana Bulgular

1. **Baseline GBM En İyi Performans**: Tree-based modeller tabular data için hala güçlü
2. **MLP Yakın Performans**: Neural network'ler de uygun (ROC-AUC: 0.7108)
3. **LSTM Overfit Eğilimi**: Sequence model overfitting gösterdi
4. **Feature Engineering Etkili**: 203 feature ile kapsamlı feature seti

### Sınırlılıklar

- **Injury Features Eksik**: Veri yetersizliği nedeniyle kaldırıldı
- **Real-time Pipeline Bug**: `predict_today.py` feature mismatch sorunu
- **Veri Kapsama**: 2010-2025 (daha eski veriler eklenebilir)

### Gelecek Çalışmalar

1. **Injury Feature Entegrasyonu**: Daha kapsamlı injury verisi + ablation çalışması
2. **Model Mimari İyileştirmeleri**: Daha derin MLP, attention mekanizmalı LSTM, ensemble
3. **Feature Engineering**: Feature selection, importance analizi
4. **Real-time Pipeline Düzeltmesi**: Feature mismatch sorununun çözülmesi
5. **Zaman Bazlı Split**: Distribution drift analizi

### Sonuç

- Tüm modeller makul performans gösterdi (ROC-AUC > 0.70)
- Baseline GBM en iyi, MLP yakın performans
- Feature engineering önemli katkı sağladı
- Gelecek çalışmalarda injury features ve model iyileştirmeleri öncelikli

---

## Ek Notlar (Sunum İçin)

### Slide 1
- Büyük, okunabilir font
- Ekip üyelerinin isimleri ve numaraları
- Proje logosu/ikon (opsiyonel)

### Slide 2
- Problem görselleştirmesi (opsiyonel)
- Basit, anlaşılır dil

### Slide 3
- Veri kaynakları logosu/ikonları (opsiyonel)
- Feature sayısı vurgulanmalı (203)
- Dataset split görselleştirmesi (opsiyonel)

### Slide 4
- Model mimarileri diyagramları (opsiyonel)
- Kısa, öz açıklamalar

### Slide 5
- Tablo net ve okunabilir olmalı
- Figürler yüksek kalitede
- En önemli metrikler vurgulanmalı (ROC-AUC)

### Slide 6
- Bulgular maddeler halinde
- Sınırlılıklar dürüstçe belirtilmeli
- Gelecek çalışmalar somut ve uygulanabilir olmalı

---

**Sunum Süresi Önerisi**: 10-15 dakika (6 slide için)

**Sorular İçin**: Her slide sonunda kısa bir duraklama yapılabilir







