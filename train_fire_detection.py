from ultralytics import YOLO
import os
import shutil
import torch

# GPU kullanılabilirliğini kontrol et
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device.upper()}")

# Çalışma dizinini al
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Veri seti klasör yapısını oluştur
datasets_dir = os.path.join(current_dir, 'datasets')
if not os.path.exists(datasets_dir):
    os.makedirs(datasets_dir)

# Fire_data klasörünü datasets altına kopyala
source_dir = os.path.join(current_dir, 'Fire_data')
target_dir = os.path.join(datasets_dir, 'Fire_data')
if not os.path.exists(target_dir):
    shutil.copytree(source_dir, target_dir)

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')  # Başlangıç modeli olarak YOLOv8n kullanılıyor
model.to(device)  # Modeli seçilen cihaza taşı

# Eğitim parametreleri için YAML dosyası oluştur
data_yaml = {
    'train': os.path.join(datasets_dir, 'Fire_data/images/train'),
    'val': os.path.join(datasets_dir, 'Fire_data/images/valid'),
    'nc': 1,
    'names': ['fire']
}

with open('fire_data.yaml', 'w') as f:
    f.write(f"train: {data_yaml['train']}\n")
    f.write(f"val: {data_yaml['val']}\n")
    f.write(f"nc: {data_yaml['nc']}\n")
    f.write(f"names: {data_yaml['names']}\n")

# Modeli eğit
results = model.train(
    data='fire_data.yaml',
    epochs=150,
    imgsz=640,
    batch=32 if device == 'cuda' else 8,
    device=device,
    project='fire_detection_results',
    name='fire_detection_tuned',
    patience=75,
    save=True,
    save_period=10,
    cache=True if device == 'cuda' else False,
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    verbose=True,
    seed=0,
    deterministic=True,
    workers=8 if device == 'cuda' else 4,
    amp=True if device == 'cuda' else False,
    cos_lr=True,
    close_mosaic=10,
    resume=False,
    fraction=1.0,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.1,
    val=True,
    plots=True,
    rect=False,
    nms=True,
    agnostic_nms=False,
    max_det=300,
    half=True if device == 'cuda' else False,
    dnn=False,
    multi_scale=True,
    single_cls=False,
    tracker='bytetrack.yaml',
    vid_stride=1,
    line_width=3,
    visualize=False,
    augment=True,
    degrees=5.0,
    translate=0.2,
    scale=0.6,
    shear=2.0,
    perspective=0.001,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    auto_augment='randaugment',
    erasing=0.4,
    crop_fraction=1.0
)

# Eğitilmiş modeli dışa aktar
model.export(format='onnx')
