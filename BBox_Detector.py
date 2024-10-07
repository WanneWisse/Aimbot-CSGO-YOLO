from ultralytics import YOLO
from camera import Camera
import numpy as np
import cv2
import win32con
import win32api
import random
class BBoxDetector():
    def __init__(self,queue):
        self.model = YOLO('trained_yolo_single.pt').to('cuda')
        self.cam = Camera() 
        self.left, self.top, self.width, self.height = self.cam.get_window_rect("Counter-Strike 2")
        self.region = self.get_window_coords(self.left, self.top, self.width, self.height)
        self.cam_dxinstance = self.cam.start_dxcam(self.region,20)
        self.queue = queue
        self.center_x = self.width/2
        self.center_y = self.height/2
        self.relative_min_x = -3000
        self.relative_max_x = 3000
        self.relative_x = 0
        self.guidance_direction = -1
        self.moves = []
    
    def reverse_moves(self):
        self.moves = self.moves[::-1]
        for move in self.moves:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(move[0]), int(move[1]), 0, 0)
        self.moves = []
        
    def move_policy(self, best_bbox):
        if type(best_bbox) != type(None):
            start_x,start_y,finish_x,finish_y = best_bbox
            
            center_bbox_x = start_x + (finish_x-start_x)/2
            center_bbox_y = start_y + (finish_y-start_y)/2

            distance = np.sqrt((finish_x - self.center_x)**2 + (finish_y-self.center_x)**2)
            
            dx = center_bbox_x - self.center_x 
            dy = center_bbox_y - self.center_y 

            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)
            self.relative_x += dx 
            


            #self.moves.append((-dx,-dy))
            if dx < 5 and dy < 5:
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0)
                win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0,0)
        else:
            move_x = self.guidance_direction * 100
            print(self.relative_x)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, 0, 0, 0)
            #self.moves.append((-move_x,0))
            self.relative_x += move_x
            if self.relative_x < self.relative_min_x or self.relative_x > self.relative_max_x:
                self.guidance_direction *= -1


    def parse_frames(self):
        amount_moves = 0
        while True:
            if not self.queue.empty():
                message = self.queue.get()
                if message == True:
                    print("amount moves till kill: ", amount_moves)
                    amount_moves = 0
                    print("Switching to next bounding box")
            img = self.cam_dxinstance.get_latest_frame()
            frame = np.array(img)
            best_box = self.box_from_frame(frame)

            if best_box != None:
                best_bbox = best_box.xyxy[0].cpu().numpy()
                self.move_policy(best_bbox) 
                cv2.rectangle(frame, (int(best_bbox[0]),int(best_bbox[1])), (int(best_bbox[2]),int(best_bbox[3])), color=(0, 255, 0), thickness=3)
            else:
                self.move_policy(None)
            amount_moves+=1

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
