from ultralytics import YOLO
import cv2
import os
import torch

def test_model(model_path, image_path, conf_threshold=0.25):
    # GPU kullanılabilirliğini kontrol et
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == '0' else 'CPU'}")
    
    # Modeli yükle
    model = YOLO(model_path)
    
    # Görüntüyü yükle
    img = cv2.imread(image_path)
    if img is None:
        print(f"Görüntü yüklenemedi: {image_path}")
        return
    
    # Tahmin yap (GPU kullan)
    results = model(img, conf=conf_threshold, device=device)
    
    # Sonuçları görselleştir
    for r in results:
        im_array = r.plot()  # Tahmin edilen kutuları çiz
        cv2.imshow("Fire Detection", im_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Tahmin sonuçlarını yazdır
        print(f"Tespit edilen yangın sayısı: {len(r.boxes)}")
        for box in r.boxes:
            print(f"Güven skoru: {box.conf[0]:.2f}")
            print(f"Kutu koordinatları: {box.xyxy[0]}")

if __name__ == "__main__":
    # Eğitilmiş model yolu
    model_path = "fire_detection_results/fire_detection/weights/best.pt"
    
    # Test görüntüsü yolu
    test_image = "Fire_data/images/test/fire.16.png"
    
    # Modeli test et
    test_model(model_path, test_image) 