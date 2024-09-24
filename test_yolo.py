from ultralytics import YOLO
import cv2
image = cv2.imread('image.jpg')
#Load a pre-trained YOLOv8 model (choose from yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO('yolov10n.pt').to('cuda')  # 'n' is for nano version, can be replaced with 's', 'm', 'l', 'x'
# Run detection on an image
results = model.predict(source=image, imgsz=320, device='0')
for result in results:
    boxes = result.boxes
    for box in boxes:
        bbox = box.xyxy[0].cpu().numpy()  # Convert to NumPy array
        confidence = box.conf[0].cpu().item()  # Convert to a Python float
        class_id = box.cls[0].cpu().item()
        cv2.rectangle(image, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color=(0, 255, 0), thickness=3)

cv2.imshow("lalala", image)
k = cv2.waitKey(0) # 0==wait forever
cv2.destroyAllWindows()
# # Display the detection results

# results[0].show()

