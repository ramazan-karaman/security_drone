from ultralytics import YOLO
import cv2
import torch
import time
import os

def check_available_cameras():
    """Mevcut kameraları kontrol et"""
    available_cameras = []
    for i in range(10):  # İlk 10 kamera indeksini kontrol et
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Windows için DirectShow kullan
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def live_fire_detection(model_path, conf_threshold=0.25):
    # GPU kullanılabilirliğini kontrol et
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == '0' else 'CPU'}")
    
    # Modeli yükle
    model = YOLO(model_path)
    
    # Mevcut kameraları kontrol et
    available_cameras = check_available_cameras()
    if not available_cameras:
        print("Hiç kamera bulunamadı!")
        print("Lütfen şunları kontrol edin:")
        print("1. Webcam'inizin bağlı olduğundan emin olun")
        print("2. Webcam sürücülerinin yüklü olduğundan emin olun")
        print("3. Webcam'inizin başka bir uygulama tarafından kullanılmadığından emin olun")
        return
    
    print(f"Bulunan kameralar: {available_cameras}")
    camera_index = available_cameras[0]  # İlk bulunan kamerayı kullan
    print(f"Kamera {camera_index} kullanılıyor...")
    
    # Webcam'i başlat (Windows için DirectShow kullan)
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Kamera {camera_index} açılamadı!")
        return
    
    # Kamera özelliklerini ayarla
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # FPS hesaplama için değişkenler
    start_time = time.time()
    frame_count = 0
    
    print("Yangın tespiti başlatıldı. Çıkmak için 'q' tuşuna basın...")
    
    try:
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
                
    except KeyboardInterrupt:
        print("\nProgram kullanıcı tarafından durduruldu.")
    except Exception as e:
        print(f"Bir hata oluştu: {str(e)}")
    finally:
        # Kaynakları serbest bırak
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Eğitilmiş model yolu
    model_path = "fire_detection_results/fire_detection/weights/best.pt"
    
    # Model dosyasının varlığını kontrol et
    if not os.path.exists(model_path):
        print(f"Hata: Model dosyası bulunamadı: {model_path}")
        print("Lütfen model dosyasının doğru konumda olduğundan emin olun.")
    else:
        # Canlı tespit başlat
        live_fire_detection(model_path) 