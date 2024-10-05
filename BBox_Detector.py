from ultralytics import YOLO
from camera import Camera
import numpy as np
import cv2
class BBoxDetector():
    def __init__(self):
        self.model = YOLO('trained_yolo_single.pt').to('cuda')
        self.cam = Camera() 
        self.left, self.top, self.width, self.height = self.cam.get_window_rect("Counter-Strike 2")
        self.region = self.get_window_coords(self.left, self.top, self.width, self.height)
        self.cam_dxinstance = self.cam.start_dxcam(self.region,20)
    
    def parse_frames(self):
        while True:
            img = self.cam_dxinstance.get_latest_frame()
            frame = np.array(img)
            best_box = self.box_from_frame(frame)
            if best_box != None:
                bbox = best_box.xyxy[0].cpu().numpy() 
                cv2.rectangle(frame, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color=(0, 255, 0), thickness=3)
            cv2.imshow("Recording with Bboxes", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        cv2.destroyAllWindows()
        self.cam_dxinstance.stop()

    
    def get_window_coords(self, left,top,width,height):
        start_x = left + 7
        start_y = top
        finish_x = left + width - 7
        finish_y = height + top
        return (start_x,start_y,finish_x,finish_y)
    
    def box_from_frame(self, frame):
        results = self.model.predict(source=frame, imgsz=640, device='0',verbose=False)
        best_boxes = []
        for result in results:
            boxes = list(result.boxes)
            if len(boxes)>0:
                boxes.sort(key=lambda box: box.conf[0].cpu().item(), reverse=True)
                best_box = boxes[0]
                best_boxes.append(best_box)
        if len(best_boxes)>0:
            best_boxes.sort(key=lambda box: box.conf[0].cpu().item(), reverse=True)
            return best_boxes[0]
        return None
