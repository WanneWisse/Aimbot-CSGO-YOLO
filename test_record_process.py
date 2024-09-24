import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import win32gui
import win32ui
import win32con
import win32api
import dxcam
from ultralytics import YOLO

#Load a pre-trained YOLOv8 model (choose from yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
model = YOLO('trained_yolo_model.pt').to('cuda')  # 'n' is for nano version, can be replaced with 's', 'm', 'l', 'x'

# Run detection on an image


def get_window_rect():
    # Get window by title
    window = gw.getWindowsWithTitle(WINDOW_TITLE)[0]
    # Return the coordinates (left, top, width, height)
    print(window)
    return window.left, window.top, window.width, window.height

def start_dxcam():
    scale_correction = 7
    dx = dxcam.create(device_idx=0, output_idx=0, output_color="BGR", max_buffer_len=64)
    left,top,width,height = get_window_rect()
    dx.start(region=(left+scale_correction,top,left+width-scale_correction,height+top), target_fps=FPS)
    #print(f"Recording window '{WINDOW_TITLE}' at ({left}, {top}) with size {width}x{height}")
    return dx


def record_window():
    cam = start_dxcam()

    while True:
        img = cam.get_latest_frame()
        frame = np.array(img)
        results = model.predict(source=frame, imgsz=320, device='0')
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].cpu().numpy()  # Convert to NumPy array
                confidence = box.conf[0].cpu().item()  # Convert to a Python float
                class_id = box.cls[0].cpu().item()
                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color=(0, 255, 0), thickness=3)

        # Display the frame with bounding boxes
        cv2.imshow("Recording with Bboxes", frame)
        # Stop recording with 'q'
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
    cam.stop()
if __name__ == "__main__":
    WINDOW_TITLE = "Counter-Strike 2"  # Replace this with the title of the window you want to record
    FPS = 20
    record_window()