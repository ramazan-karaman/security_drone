from ultralytics import YOLO
import cv2
import time
import torch

def detect_fire_realtime(model_path, conf_threshold=0.25):
    # GPU kullanılabilirliğini kontrol et
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == '0' else 'CPU'}")
    
    # Modeli yükle
    model = YOLO(model_path)
    
    # Webcam'i başlat
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Webcam açılamadı!")
        return
    
    # FPS hesaplama için değişkenler
    prev_time = 0
    fps = 0
    
    # GPU bellek optimizasyonu için
    torch.backends.cudnn.benchmark = True
    
    while True:
        # Görüntüyü al
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı!")
            break
        
        # Tahmin yap (GPU kullan)
        results = model(frame, conf=conf_threshold, device=device)
        
        # Sonuçları görselleştir
        for r in results:
            frame = r.plot()
            
            # FPS hesapla ve göster
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Yangın tespit edildiğinde uyarı göster
            if len(r.boxes) > 0:
                cv2.putText(frame, "YANGIN TESPIT EDILDI!", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Görüntüyü göster
        cv2.imshow("Fire Detection", frame)
        
        # 'q' tuşuna basılırsa çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Eğitilmiş model yolu
    model_path = "fire_detection_results/fire_detection/weights/best.pt"
    
    # Gerçek zamanlı yangın tespiti başlat
    detect_fire_realtime(model_path) 