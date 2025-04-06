from ultralytics import YOLO
import cv2
import torch
import time

def live_fire_detection(model_path, conf_threshold=0.25):
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
    start_time = time.time()
    frame_count = 0
    
    print("Yangın tespiti başlatıldı. Çıkmak için 'q' tuşuna basın...")
    
    while True:
        # Webcam'den görüntü al
        ret, frame = cap.read()
        if not ret:
            print("Görüntü alınamadı!")
            break
        
        # Tahmin yap
        results = model(frame, conf=conf_threshold, device=device)
        
        # Sonuçları görüntüye çiz
        annotated_frame = results[0].plot()
        
        # FPS hesapla ve göster
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Yangın tespit sayısını göster
        fire_count = len(results[0].boxes)
        cv2.putText(annotated_frame, f"Yangın Sayısı: {fire_count}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Görüntüyü göster
        cv2.imshow("Yangın Tespiti", annotated_frame)
        
        # 'q' tuşuna basıldığında döngüyü sonlandır
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Kaynakları serbest bırak
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Eğitilmiş model yolu
    model_path = "fire_detection_results/fire_detection/weights/best.pt"
    
    # Canlı tespit başlat
    live_fire_detection(model_path) 