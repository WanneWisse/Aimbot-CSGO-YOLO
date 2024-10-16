from ultralytics import YOLO
from camera import Camera
import numpy as np
import cv2
import win32con
import win32api
import random
from policy_gradient_network import PolicyNetwork
import matplotlib.pyplot as plt
import torch
import time
import torch.optim as optim
class BBoxDetector():
    def __init__(self):
        self.model = YOLO('trained_yolo_single.pt').to('cuda')
        self.cam = Camera() 
        self.left, self.top, self.width, self.height = self.cam.get_window_rect("Counter-Strike 2")
        self.region = self.get_window_coords(self.left, self.top, self.width, self.height)
        self.cam_dxinstance = self.cam.start_dxcam(self.region,20)
        self.center_x = self.width/2
        self.center_y = self.height/2
        self.relative_min_x = -3000
        self.relative_max_x = 3000
        self.relative_min_y = -1000
        self.relative_max_y = 1000
        self.relative_x = 0
        self.relative_y = 0
        self.guidance_direction_x = -1
        self.guidance_direction_y = -1
        self.moves = []
        self.mouse_policy = PolicyNetwork([2,8,8,2])
        self.optimizer = optim.Adam(self.mouse_policy.parameters(), lr=1e-4)
    
    
    def guidance_policy(self):
    

        if self.relative_x < self.relative_min_x:
            self.guidance_direction_x = 1
        elif self.relative_x > self.relative_max_x:
            self.guidance_direction_x = -1
            
        
        if self.relative_y < self.relative_min_y:
            self.guidance_direction_y = 1
        elif self.relative_y > self.relative_max_y:
            self.guidance_direction_y = -1
            

        move_speed= 100
        move_x = self.guidance_direction_x * move_speed
        move_y = self.guidance_direction_y * move_speed
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_x, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE,0, move_y, 0, 0)
        
        self.relative_x += move_x
        self.relative_y += move_y

    def move_policy(self, best_bbox):
        start_x,start_y,finish_x,finish_y = best_bbox
        
        center_bbox_x = start_x + (finish_x-start_x)/2
        center_bbox_y = start_y + (finish_y-start_y)/2

        
        dx_pixel = center_bbox_x - self.center_x 
        dy_pixel = center_bbox_y - self.center_y 

        state = torch.tensor([dx_pixel,dy_pixel], dtype=torch.float32)
        print(state)
        

        action_means, action_stds = self.mouse_policy.forward(state)
        print(action_means)
        dx3D_distribution = torch.distributions.Normal(action_means[0], action_stds[0])
        dx3D_prediction = dx3D_distribution.sample()
        print(dx3D_prediction)

        dy3D_distribution = torch.distributions.Normal(action_means[1], action_stds[1])
        dy3D_prediction = dy3D_distribution.sample()

        dx3D_log_prob = dx3D_distribution.log_prob(dx3D_prediction)
        dy3D_log_prob = dy3D_distribution.log_prob(dy3D_prediction)

        total_log_prob = dx3D_log_prob + dy3D_log_prob

        x_move = dx3D_prediction.item()
        y_move = dy3D_prediction.item()
        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(x_move), int(y_move), 0, 0)
        self.relative_x += x_move
        self.relative_y += y_move

        return total_log_prob

        
        
    def backup_loss(self,best_bbox,log_prob_action):
        if type(best_bbox) is np.ndarray:
            start_x,start_y,finish_x,finish_y = best_bbox
                
            center_bbox_x = start_x + (finish_x-start_x)/2
            center_bbox_y = start_y + (finish_y-start_y)/2
            
            reward = -np.sqrt((center_bbox_x - self.center_x )**2 + (center_bbox_y - self.center_y)**2) 
        else:
            reward = -1000

        print(log_prob_action)
        reward = torch.tensor([reward], dtype=torch.float32)

        loss = -log_prob_action * reward
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def end_episode(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0,0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0,0)
        
    def move_random(self):
        dx = random.choice([random.randint(-100,-50),random.randint(50,100)])
        dy = random.choice([random.randint(-100,-50),random.randint(50,100)])
        self.relative_x += dx
        self.relative_y += dy

        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)

    def parse_frames(self):
        prediction_done = False
        self.losses = []
        self.episodes = 10000
        log_prob = None
        while True:
            if self.episodes < 0:
                break
            img = self.cam_dxinstance.get_latest_frame()
            frame = np.array(img)
            best_box = self.box_from_frame(frame)
            
            if best_box != None:
                best_bbox = best_box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame, (int(best_bbox[0]),int(best_bbox[1])), (int(best_bbox[2]),int(best_bbox[3])), color=(0, 255, 0), thickness=3)
                if prediction_done:
                    print("calc loss")
                    loss = self.backup_loss(best_bbox,log_prob)
                    self.losses.append(loss)
                    self.move_random()
                    prediction_done = False

                else:
                    print("predict")
                    log_prob = self.move_policy(best_bbox)
                    #time.sleep(0.1)
                    self.episodes -= 1
                    prediction_done = True

            else:
                print(1)
                if prediction_done:
                    print("calc loss")
                    loss = self.backup_loss(None,log_prob)
                    prediction_done = False
                #print("guidance")
                self.guidance_policy()

            cv2.imshow("Recording with Bboxes", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        
        cv2.destroyAllWindows()
        self.cam_dxinstance.stop()
        # 5. Visualize the loss over epochs
        plt.plot(self.losses)
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        # Save the plot to a file (e.g., a PNG image)
        plt.savefig('loss_plot.png')

    
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


bbox_detector = BBoxDetector()
bbox_detector.parse_frames()
