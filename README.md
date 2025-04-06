# YOLOv8 Yangın Tespiti Projesi

Bu proje, YOLOv8 kullanarak yangın tespiti yapan bir derin öğrenme modeli eğitmek ve kullanmak için oluşturulmuştur.

## Özellikler

- YOLOv8n modeli kullanılarak yangın tespiti
- GPU ve CPU desteği
- Gerçek zamanlı webcam tespiti
- FPS göstergesi
- Yangın tespit edildiğinde görsel uyarı
- Eğitim ve test scriptleri
- Google Colab desteği

## Kurulum

1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

2. Veri seti yapısı:
```
Fire_data/
    ├── images/
    │   ├── train/  (eğitim görüntüleri)
    │   └── valid/  (doğrulama görüntüleri)
    └── labels/
        ├── train/  (eğitim etiketleri)
        └── valid/  (doğrulama etiketleri)
```

## Kullanım

### Eğitim
```bash
python train_fire_detection.py
```

### Test
```bash
python test_fire_detection.py
```

### Gerçek Zamanlı Tespit
```bash
python detect_fire_realtime.py
```

### Google Colab Kullanımı
1. Colab notebook'u açın
2. Runtime > Change runtime type > GPU seçin
3. Tüm kodları çalıştırın

## Gereksinimler

- Python 3.8+
- PyTorch
- Ultralytics
- OpenCV
- CUDA (GPU için)

## Lisans

MIT License

## Katkıda Bulunma

1. Bu repoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## İletişim

Proje Sahibi - [@yourusername](https://github.com/yourusername)

Proje Linki: [https://github.com/yourusername/fire-detection-yolov8](https://github.com/yourusername/fire-detection-yolov8) 