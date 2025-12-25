# NBA Maç Sonucu Tahmini: Yapay Sinir Ağları ile Sınıflandırma ve Regresyon

**CENG 481 - Yapay Sinir Ağları Dersi Projesi**

**Ekip:**
- İbrahim Ersan Özdemir (202211054)
- Ekin Efe Kızıloğlu (202211043)

---

## 1. Giriş

### 1.1 Problem Tanımı

Bu proje, NBA (National Basketball Association) maçlarının sonuçlarını yapay sinir ağları kullanarak tahmin etmeyi amaçlamaktadır. İki ana görev üzerinde çalışılmıştır:

1. **Sınıflandırma (Classification)**: Ev sahibi takımın maçı kazanıp kazanmayacağını tahmin etme (binary classification: home team win/loss)
2. **Regresyon (Regression)**: Maç skor farkını tahmin etme (score difference: home_score - away_score)

### 1.2 Motivasyon

NBA maç sonuçlarının tahmini, spor analitiği alanında önemli bir problemdir. Bu tahminler:
- Spor bahis endüstrisi için değerli bilgiler sağlar
- Takım performans analizlerinde kullanılabilir
- Oyuncu transfer kararlarına yardımcı olabilir
- Fan engagement için ilginç uygulamalar geliştirilebilir

### 1.3 Proje Hedefleri

- Farklı yapay sinir ağı mimarilerini (MLP, LSTM) test etmek
- Baseline modellerle (Gradient Boosting) karşılaştırma yapmak
- Feature engineering tekniklerini uygulamak
- Model performansını kapsamlı metriklerle değerlendirmek

---

## 2. Veri Seti ve Kaynaklar

### 2.1 Veri Kaynakları

Projede üç ana veri kaynağı kullanılmıştır:

1. **NocturneBear (2010-2024)**
   - Regular season box scores (3 parça)
   - Playoff box scores
   - Toplam ~17,000+ maç verisi
   - Her satır bir oyuncunun maç istatistiklerini içerir (box score formatı)

2. **NBAstuffer (2025-2026)**
   - Takım istatistikleri (`nbastuffer_2025_2026_team_stats_raw.csv`)
   - Rest days istatistikleri (`nbastuffer_2025_2026_rest_days_stats.csv`)
   - Schedule rest days istatistikleri (`nbastuffer_2025_2026_schedule_rest_days.csv`)
   - Oyuncu gamelog verileri (`player_203999_gamelog_2025_26.csv`)

3. **BDL (Ball Don't Lie) API**
   - 2025 sezonu maç verileri (opsiyonel, bu projede kullanılmadı)

### 2.2 Veri Temizleme Süreci

Veri temizleme aşamasında şu adımlar uygulanmıştır:

1. **Tarih Normalizasyonu**: Farklı formatlardaki tarihler standart `YYYY-MM-DD` formatına dönüştürüldü
2. **Takım İsmi Standartlaştırma**: 30+ takım için kısaltma/tam isim mapping'i uygulandı
3. **Box Score'dan Maç Bazlı Formata Dönüşüm**: Oyuncu bazlı box score verisi, maç bazlı formata dönüştürüldü (her maç için `home_team`, `away_team`, `home_score`, `away_score`)
4. **Eksik/Geçersiz Veri Temizleme**:
   - Eksik skor içeren satırlar kaldırıldı
   - 0-0 skorlu maçlar kaldırıldı
   - Skor değerleri 0-200 aralığında kontrol edildi
   - Duplicate maçlar kaldırıldı

### 2.3 Feature Engineering

Toplam **203 feature** oluşturulmuştur:

#### 2.3.1 Takım İstatistikleri
- Home ve away takımlar için ayrı ayrı:
  - GP (Games Played), PPG (Points Per Game), oPPG (Opponent Points Per Game)
  - PACE, oEFF (Offensive Efficiency), dEFF (Defensive Efficiency)
  - Win/Loss kayıtları, win percentage
  - SoS (Strength of Schedule), rSoS (Relative Strength of Schedule)
  - Diğer takım istatistikleri (SAR, CONS, A4F, ACH, STRK)

#### 2.3.2 Rest Days Features
- Farklı rest day senaryoları için istatistikler:
  - 1 DAY, 2 DAYS, 3 DAYS rest
  - B2B (Back-to-Back) maçlar
  - 3IN4, 4IN5 yoğun program senaryoları
- Her senaryo için: GP, Win%, AED (Average Efficiency Differential)

#### 2.3.3 Schedule Rest Days Features
- Takım bazlı genel schedule istatistikleri

#### 2.3.4 Rolling Window Features
- 5 ve 10 maçlık rolling window'lar için:
  - Win rate
  - Average score difference
  - Average points for/against
- Home ve away takımlar için ayrı ayrı hesaplandı

#### 2.3.5 ELO Ratings
- Her takım için ELO rating hesaplandı (base rating: 1500, K-factor: 20)
- `home_elo_before`, `away_elo_before`, `diff_elo` feature'ları eklendi

#### 2.3.6 Diff Features
- Home ve away takım feature'ları arasındaki farklar:
  - `diff_team_*`: Takım istatistikleri farkları
  - `diff_rest_*`: Rest days farkları
  - `diff_schedule_*`: Schedule farkları
  - `diff_roll_w*`: Rolling window farkları

#### 2.3.7 Tarih Feature'ları
- `month`: Ay (1-12)
- `day_of_week`: Haftanın günü (0-6)
- `is_weekend`: Hafta sonu mu? (0/1)
- `is_playoff`: Playoff maçı mı? (0/1)

**Not:** Injury feature'ları bu projede kaldırılmıştır (veri yetersizliği nedeniyle). Inference-time'da ayrı olarak kullanılması planlanmıştır.

### 2.4 Dataset Split Stratejisi

Veri, **random shuffle** yöntemiyle bölünmüştür:
- **Train**: %70 (random_state=30)
- **Validation**: %15
- **Test**: %15

Random shuffle kullanılmasının nedeni, zaman bazlı split'in distribution drift sorununa yol açabileceği endişesidir. Random shuffle ile tüm yıllardan veri içeren dengeli setler oluşturulmuştur.

---

## 3. Yöntem

### 3.1 Feature Engineering Pipeline

Feature engineering süreci `src/features/build_features.py` modülünde gerçekleştirilmiştir:

1. Tarih feature'ları ekleme
2. ELO rating hesaplama (kronolojik sıraya göre)
3. Rolling window feature'ları hesaplama (takım bazlı, tarih sırasına göre)
4. Diff feature'ları oluşturma
5. Numeric-only feature seçimi
6. NaN değerleri train median ile doldurma

### 3.2 Modeller

#### 3.2.1 MLP (Multi-Layer Perceptron)

**Classifier:**
- **Mimari**: 4 katmanlı fully connected network
  - Hidden units: [512, 256, 128, 64]
  - Activation: ReLU
  - Batch Normalization: Evet
  - Dropout: 0.3
  - Output: Sigmoid (binary classification)
- **Optimizer**: Adam (learning_rate=0.0005)
- **Loss**: Binary crossentropy
- **Early Stopping**: Val loss'u monitor ederek, patience=10
- **Best Model**: MLP_C3 (3 varyant arasından seçildi)

**Regressor:**
- **Mimari**: 3 katmanlı fully connected network
  - Hidden units: [256, 128, 64]
  - Activation: ReLU
  - Batch Normalization: Hayır
  - Dropout: 0.1
  - Output: Linear
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss**: Huber (delta=5.0)
- **Best Model**: MLP_R2 (2 varyant arasından seçildi)

#### 3.2.2 Sequence LSTM

- **Mimari**: LSTM tabanlı sequence model
- **Sequence Length**: N=10 (N=5 ve N=10 denendi, N=10 daha iyi performans gösterdi)
- **Real Sequence**: Evet - Takım bazında tarih sırasına göre geçmiş N maç feature'larından sequence oluşturuldu
- **Leakage Önleme**: Aynı gün maç feature'ı sequence'a dahil edilmeden önce history'den alındı (date bucket yaklaşımı)
- **Output**: Binary classification (home team win)

#### 3.2.3 Baseline: Gradient Boosting Machine (GBM)

- **Model**: Sklearn'in GradientBoostingClassifier ve GradientBoostingRegressor
- **Hyperparameters**: Default sklearn parametreleri
- **Amaç**: Neural network modellerinin performansını karşılaştırmak için referans model

### 3.3 Ön İşleme

Tüm modeller için aynı ön işleme pipeline'ı uygulanmıştır:

1. **Numeric-only Feature Seçimi**: Sadece numeric kolonlar kullanıldı (string/date kolonları hariç)
2. **All-NaN Column Drop**: Train setinde tamamen NaN olan kolonlar kaldırıldı (2 kolon)
3. **NaN Doldurma**: Train median değerleri kullanılarak NaN'ler dolduruldu (leakage önleme)
4. **StandardScaler**: Sadece train setine fit edildi, val ve test setlerine transform uygulandı

---

## 4. Deneyler ve Sonuçlar

### 4.1 Metrik Karşılaştırması

#### 4.1.1 Classification Metrikleri (Test Set)

| Model | Accuracy | F1 | ROC-AUC | LogLoss | Brier |
|-------|----------|----|---------|---------|-------|
| MLP Classifier | 0.6665 | 0.7257 | **0.7108** | 0.6143 | 0.2124 |
| LSTM Classifier | 0.6618 | 0.7208 | 0.7008 | 0.6202 | 0.2155 |
| Baseline GBM | 0.6680 | 0.7303 | **0.7176** | 0.6099 | 0.2106 |

**Gözlemler:**
- Baseline GBM en yüksek ROC-AUC (0.7176) ve en düşük Brier score (0.2106) değerlerine sahip
- MLP Classifier Baseline'a yakın performans gösteriyor (ROC-AUC: 0.7108)
- LSTM Classifier biraz daha düşük performans gösteriyor (ROC-AUC: 0.7008)

#### 4.1.2 Regression Metrikleri (Test Set)

| Model | MAE | RMSE |
|-------|-----|------|
| MLP Regressor | 10.06 | 12.91 |
| Baseline GBM | 10.04 | 12.87 |

**Gözlemler:**
- MLP ve Baseline regressor performansları çok yakın
- MAE ~10 puan, RMSE ~13 puan (makul bir tahmin hatası)

### 4.2 ROC Curves

ROC-AUC karşılaştırması:
- **Baseline GBM**: 0.7176 (en yüksek)
- **MLP Classifier**: 0.7108
- **LSTM Classifier**: 0.7008

Tüm modeller 0.70'in üzerinde ROC-AUC değerine sahip, bu da makul bir binary classification performansı gösteriyor.

### 4.3 Calibration Plot

Brier score karşılaştırması (düşük = daha iyi calibration):
- **Baseline GBM**: 0.2106 (en iyi)
- **MLP Classifier**: 0.2124
- **LSTM Classifier**: 0.2155

Tüm modeller benzer calibration performansı gösteriyor.

### 4.4 Confusion Matrix (MLP C3 - En İyi MLP Varyantı)

Test seti confusion matrix:
```
                Predicted
              Away Win  Home Win
Actual Away Win   616      549
      Home Win    362     1205
```

- True Negatives (Away Win doğru tahmin): 616
- False Positives (Away Win yanlış tahmin): 549
- False Negatives (Home Win yanlış tahmin): 362
- True Positives (Home Win doğru tahmin): 1205

**Gözlemler:**
- Model home win'i tahmin etmede daha başarılı (1205 doğru tahmin)
- Away win tahminlerinde daha fazla hata var (549 false positive)

### 4.5 Regression Scatter

MAE ve RMSE değerleri:
- MLP ve Baseline regressor performansları çok yakın
- Ortalama tahmin hatası ~10 puan (makul bir değer)

### 4.6 Loss Curves

MLP C3 classifier için loss curve:
- Train ve validation loss'lar düzgün bir şekilde azalıyor
- Overfitting belirtisi görülmüyor (early stopping etkili olmuş)

### 4.7 Ablation Analizi

**Not:** Bu projede injury feature'ları kaldırılmış ve sadece tek bir model (injury featuresız) eğitilmiştir. Bu nedenle injury ON/OFF ablation analizi yapılamamıştır. Bu durum gelecek çalışmalar bölümünde ele alınacaktır.

---

## 5. Tartışma

### 5.1 Model Performans Karşılaştırması

1. **Baseline GBM'nin Güçlü Performansı**: Gradient Boosting Machine, neural network modellerinden daha iyi performans göstermiştir. Bu, tabular data için tree-based modellerin hala güçlü bir seçenek olduğunu gösteriyor.

2. **MLP'nin Stabil Performansı**: MLP Classifier, Baseline'a yakın performans göstermiştir (ROC-AUC: 0.7108 vs 0.7176). Bu, neural network'lerin de tabular data için uygun olduğunu gösteriyor.

3. **LSTM'nin Overfit Eğilimi**: LSTM Classifier, train setinde yüksek AUC değerleri göstermiş ancak validation/test setlerinde daha düşük performans göstermiştir. Bu, sequence model'in overfitting eğiliminde olduğunu gösteriyor.

### 5.2 Feature Engineering'in Etkisi

- 203 feature ile kapsamlı bir feature seti oluşturulmuştur
- ELO ratings, rolling windows, diff features gibi domain-specific feature'lar eklenmiştir
- Bu feature'lar model performansına önemli katkı sağlamıştır

### 5.3 Dataset Split Stratejisi

Random shuffle split kullanılması, zaman bazlı split'in distribution drift sorununu önlemiştir. Ancak, gerçek dünya senaryosunda zaman bazlı split daha gerçekçi olabilir.

### 5.4 Sınırlamalar

1. **Injury Feature'larının Eksikliği**: Injury verisi yetersiz olduğu için feature engineering'den çıkarılmıştır. Bu, model performansını olumsuz etkilemiş olabilir.

2. **Veri Kapsama**: Veri 2010-2025 arası maçları kapsamaktadır. Daha eski veriler eklenebilir.

3. **Real-time Inference Pipeline**: `predict_today.py` script'inde feature mismatch sorunu vardır. Bu, gerçek zamanlı tahminler için düzeltilmesi gereken bir konudur.

---

## 6. Gelecek Çalışmalar

1. **Injury Feature'larının Entegrasyonu**: 
   - Daha kapsamlı injury verisi toplanması
   - Injury feature'larının düzgün bir şekilde feature engineering pipeline'ına eklenmesi
   - Injury ON/OFF ablation çalışması yapılması

2. **Feature Engineering İyileştirmeleri**:
   - Daha fazla domain-specific feature eklenmesi
   - Feature selection tekniklerinin uygulanması
   - Feature importance analizi

3. **Model Mimari Denemeleri**:
   - Daha derin MLP mimarileri
   - Attention mekanizmalı LSTM modelleri
   - Ensemble yöntemleri (MLP + GBM)

4. **Real-time Prediction Pipeline Düzeltmesi**:
   - `predict_today.py` script'indeki feature mismatch sorununun çözülmesi
   - Inference-time feature engineering'in eğitim pipeline'ı ile birebir eşleşmesi

5. **Zaman Bazlı Split Denemesi**:
   - Random shuffle yerine zaman bazlı split ile model performansının karşılaştırılması
   - Distribution drift analizi

---

## 7. Sonuç

Bu projede, NBA maç sonuçlarını tahmin etmek için farklı yapay sinir ağı mimarileri (MLP, LSTM) ve baseline model (GBM) test edilmiştir. Tüm modeller makul performans göstermiştir (ROC-AUC > 0.70). Baseline GBM en iyi performansı göstermiş, ancak MLP Classifier da yakın performans göstermiştir. LSTM modeli biraz daha düşük performans göstermiş ve overfitting eğilimi sergilemiştir.

Feature engineering sürecinde 203 feature oluşturulmuş ve domain-specific feature'lar (ELO, rolling windows, diff features) eklenmiştir. Bu feature'lar model performansına önemli katkı sağlamıştır.

Gelecek çalışmalarda, injury feature'larının entegrasyonu, model mimari iyileştirmeleri ve real-time prediction pipeline'ının düzeltilmesi öncelikli konulardır.

---

## Referanslar

- NocturneBear: NBA box score verileri (2010-2024)
- NBAstuffer: NBA takım ve oyuncu istatistikleri
- TensorFlow/Keras: Neural network implementasyonu
- Scikit-learn: Baseline modeller ve metrikler

---

**Rapor Tarihi**: Aralık 2025
**Proje Kodu**: [GitHub Repository](https://github.com/SpeedyV5/NBA-Game-Prediction-using-Artificial-Neural-Networks)







