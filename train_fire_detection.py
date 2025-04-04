from ultralytics import YOLO
import os

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # Başlangıç modeli olarak YOLOv8n kullanıyoruz

# Eğitim parametreleri
data_yaml = {
    'train': 'Fire_data/images/train',  # Eğitim görüntüleri
    'val': 'Fire_data/images/valid',    # Doğrulama görüntüleri
    'nc': 1,                            # Sınıf sayısı (fire)
    'names': ['fire']                   # Sınıf isimleri
}

# YAML dosyasını oluştur
with open('fire_data.yaml', 'w') as f:
    f.write(f"train: {data_yaml['train']}\n")
    f.write(f"val: {data_yaml['val']}\n")
    f.write(f"nc: {data_yaml['nc']}\n")
    f.write(f"names: {data_yaml['names']}\n")

# Modeli eğit
results = model.train(
    data='fire_data.yaml',
    epochs=100,              # Eğitim epoch sayısı
    imgsz=640,              # Görüntü boyutu
    batch=8,                # Batch size (CPU için daha küçük batch size)
    device='cpu',           # CPU kullan
    project='fire_detection_results',  # Sonuçların kaydedileceği klasör
    name='fire_detection',  # Model adı
    patience=50,            # Erken durdurma için sabır değeri
    save=True,             # En iyi modeli kaydet
    save_period=10,        # Her 10 epoch'ta bir modeli kaydet
    cache=False,           # Görüntüleri önbelleğe alma
    exist_ok=True,         # Mevcut proje klasörünü kullan
    pretrained=True,       # Önceden eğitilmiş ağırlıkları kullan
    optimizer='auto',      # Optimizer seçimi
    verbose=True,          # Detaylı çıktı
    seed=0,                # Rastgele sayı üreteci için seed
    deterministic=True     # Deterministik eğitim
)

# Eğitilmiş modeli kaydet
model.export(format='onnx')  # ONNX formatında kaydet 