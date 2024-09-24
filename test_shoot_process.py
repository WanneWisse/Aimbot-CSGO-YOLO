import cv2
import asyncio
import numpy as np
import pyautogui
import pygetwindow as gw
import win32gui
import win32ui
import win32con
import win32api
import dxcam
from ultralytics import YOLO
import time

model = YOLO('trained_yolo_model.pt').to('cuda')  


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
    return dx, width,height

def move_mouse(bbox,width,height):
    #Calculate the movement (half the screen width)
    #update logic to be faster
    width_bbox = np.sqrt((bbox[2] - bbox[0])**2)
    height_bbox = np.sqrt((bbox[3] - bbox[1])**2)

    dx = int((bbox[0] + width_bbox/2 - width/2)) * 2
    dy = int((bbox[1] + height_bbox/2 - height/2)) *2 

    #Move the mouse relative to its current position
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)
    if dx < 10 and dy < 10:
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0,0)

def record_window():
    cam,width,height = start_dxcam()

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
                if class_id == 1 or class_id == 2 or class_id == 3 or class_id == 4:
                    move_mouse(bbox,width,height)
                    cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color=(0, 255, 0), thickness=3)
                    break 

        # Display the frame with bounding boxes
        cv2.imshow("Recording with Bboxes", frame)
        # Stop recording with 'q'
        if cv2.waitKey(1) == ord("q"):
            break
    cv2.destroyAllWindows()
    cam.stop()
if __name__ == "__main__":
    WINDOW_TITLE = "Counter-Strike 2"  # Replace this with the title of the window you want to record
    FPS = 40
    record_window()