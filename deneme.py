from ultralytics import YOLO
import cv2

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# Görüntüyü işle
results = model(r"Fire_data/images/test/fire.16.png")

# İlk sonucu al
res = results[0]

# Çıktıdaki bounding box'ları çiz ve göster
res.save(filename="detected_fire.png")  # Alternatif olarak kaydedebilirsin

# OpenCV ile gösterme
img = cv2.imread("detected_fire.png")
cv2.imshow("Fire Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
