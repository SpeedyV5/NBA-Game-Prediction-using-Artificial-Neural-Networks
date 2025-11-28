# ANN PROJE – NBA Game Prediction using Artificial Neural Networks

Bu proje, CENG 481 dersi kapsamında NBA maçlarının sonucunu (win/loss) ve skor farkını
yapay sinir ağları kullanarak tahmin etmeyi amaçlar.

## Ekip
- İbrahim Ersan Özdemir (202211054)
- Ekin Efe Kızıloğlu (202211043)

## Proje Yapısı

```text
ann-proje/
  data_raw/          # Ham veriler (Git'e push edilmeyecek)
  data_interim/      # Ara işlenmiş veriler
  data_processed/    # Modelde kullanılacak final CSV'ler
  notebooks/         # Jupyter/Colab notebook'ları
  src/
    data/            # Veri çekme ve temizleme script'leri
    features/        # Feature engineering kodları
    models/          # Model tanımı ve eğitim kodları
  reports/           # Rapor ve figür dosyaları
  models/            # Eğitilmiş model dosyaları
