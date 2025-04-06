from ultralytics import YOLO
import os
import torch
from google.colab import drive

# Google Drive'ı bağla
drive.mount('/content/drive')

# Çalışma dizinini ayarla
os.chdir('/content/drive/MyDrive/your_project_folder')  # Proje klasörünüzün yolunu buraya yazın

# GPU kullanılabilirliğini kontrol et
device = '0' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {'GPU' if device == '0' else 'CPU'}")

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
    batch=32,               # GPU için daha büyük batch size
    device=device,          # GPU kullan
    project='fire_detection_results',  # Sonuçların kaydedileceği klasör
    name='fire_detection',  # Model adı
    patience=50,            # Erken durdurma için sabır değeri
    save=True,             # En iyi modeli kaydet
    save_period=10,        # Her 10 epoch'ta bir modeli kaydet
    cache=True,            # Görüntüleri önbelleğe alma (GPU için)
    exist_ok=True,         # Mevcut proje klasörünü kullan
    pretrained=True,       # Önceden eğitilmiş ağırlıkları kullan
    optimizer='auto',      # Optimizer seçimi
    verbose=True,          # Detaylı çıktı
    seed=0,                # Rastgele sayı üreteci için seed
    deterministic=True,    # Deterministik eğitim
    workers=8,             # Veri yükleme işçi sayısı
    amp=True,              # Otomatik karışık hassasiyet (GPU için)
    cos_lr=True,           # Kosinüs öğrenme oranı zamanlaması
    close_mosaic=10,       # Son 10 epoch'ta mozaik augmentasyonu kapat
    resume=False,          # Önceki eğitimi devam ettir
    fraction=1.0,          # Veri setinin tamamını kullan
    overlap_mask=True,     # Maske örtüşmesini etkinleştir
    mask_ratio=4,          # Maske oranı
    dropout=0.0,           # Dropout oranı
    val=True,              # Doğrulama yap
    plots=True,            # Eğitim grafiklerini çiz
    rect=False,            # Dikdörtgen eğitim
    nms=True,              # NMS kullan
    agnostic_nms=False,    # Sınıf-agnostik NMS
    max_det=300,           # Maksimum tespit sayısı
    half=False,            # Yarı hassasiyet (FP16)
    dnn=False,             # OpenCV DNN kullan
    evolve=False,          # Hiperparametre evrimi
    multi_scale=False,     # Çoklu ölçek eğitimi
    single_cls=False,      # Tek sınıf eğitimi
    tracker='bytetrack.yaml',  # Nesne takipçisi
    vid_stride=1,          # Video adımı
    line_thickness=3,      # Çizgi kalınlığı
    visualize=False,       # Görselleştirme
    augment=True,          # Veri artırma
    degrees=0.0,           # Döndürme derecesi
    translate=0.1,         # Öteleme
    scale=0.5,             # Ölçeklendirme
    shear=0.0,             # Kesme
    perspective=0.0,       # Perspektif
    flipud=0.0,            # Dikey çevirme
    fliplr=0.5,            # Yatay çevirme
    mosaic=1.0,            # Mozaik artırma
    mixup=0.0,             # Mixup artırma
    copy_paste=0.0,        # Kopyala-yapıştır artırma
    auto_augment='randaugment',  # Otomatik artırma
    erasing=0.4,           # Silme artırma
    crop_fraction=1.0      # Kırpma oranı
)

# Eğitilmiş modeli kaydet
model.export(format='onnx')  # ONNX formatında kaydet 