from ultralytics import YOLO
import cv2
image = cv2.imread('image.jpg')

model = YOLO('yolov10n.pt').to('cuda')
# Run detection on an image
results = model.predict(source=image, imgsz=320, device='0')
for result in results:
    boxes = result.boxes
    for box in boxes:
        bbox = box.xyxy[0].cpu().numpy()  # Convert to NumPy array
        confidence = box.conf[0].cpu().item()  # Convert to a Python float
        class_id = box.cls[0].cpu().item()
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color=(0, 255, 0), thickness=3)

cv2.imshow("Show_results", image)
k = cv2.waitKey(0) # 0==wait forever
cv2.destroyAllWindows()

