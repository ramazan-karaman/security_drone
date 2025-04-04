# YOLOv8 Yangın Tespiti Projesi

Bu proje, YOLOv8 kullanarak yangın tespiti yapan bir derin öğrenme modeli eğitmek ve kullanmak için oluşturulmuştur.

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

1. Model Eğitimi:
```bash
python train_fire_detection.py
```

2. Test:
```bash
python test_fire_detection.py
```

3. Gerçek Zamanlı Tespit:
```bash
python detect_fire_realtime.py
```

## Özellikler

- YOLOv8n modeli kullanılarak yangın tespiti
- CPU üzerinde eğitim ve çıkarım desteği
- Gerçek zamanlı webcam tespiti
- FPS göstergesi
- Yangın tespit edildiğinde görsel uyarı

## Notlar

- Eğitim CPU üzerinde yapılacaktır, bu nedenle eğitim süresi uzun olabilir
- Gerçek zamanlı tespit için webcam gereklidir
- Model eğitimi tamamlandıktan sonra en iyi model `fire_detection_results/fire_detection/weights/best.pt` konumunda kaydedilecektir 