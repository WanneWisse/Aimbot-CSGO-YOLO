import cv2
import numpy as np
from camera import Camera
class ScoreDetector():
    def __init__(self,queue):
        self.bright = False
        self.count_next_flair = True
        self.window_check_length = 4
        self.window_check = []
        self.sum_binary_list = []
        self.threshold = 45000
        self.kills = 0
        self.cam = Camera()
        self.left, self.top, self.width, self.height = self.cam.get_window_rect("Counter-Strike 2")
        self.region = self.get_window_coords(self.left, self.top, self.width, self.height)
        self.cam_dxinstance = self.cam.start_dxcam(self.region,60)
        self.queue = queue

    def get_window_coords(self,left,top,width,height):
        start_x = int(left + 7 + width/2 - 20)
        start_y = top + height - 118
        finish_x = int(left + 7 + width/2 + 5)
        finish_y = top + height - 100
        return (start_x,start_y,finish_x,finish_y)
    
    def parse_frames(self):
        while True:
            img = self.cam_dxinstance.get_latest_frame()
            frame = np.array(img)
            kill,kills,sum_binary_list = self.detect_score_frame(frame)
            if kill:
                self.queue.put(kill)
                print(kills)
            cv2.imshow("Recording with Bboxes", frame)
            if cv2.waitKey(1) == ord("q"):
                break
        
        cv2.destroyAllWindows()      
        self.cam_dxinstance.stop()
        
        xpoints = np.array([i for i in range(len(sum_binary_list))])
        ypoints = np.array(sum_binary_list)

        plt.plot(xpoints, ypoints)
        plt.show()
        
    def detect_score_frame(self,frame):
        kill = False
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 40, 40])    # Lower bound for green
        upper_green = np.array([80, 255, 255])  # Upper bound for green

        mask = cv2.inRange(hsv, lower_green, upper_green)

        green_only = cv2.bitwise_and(frame, frame, mask=mask)

        gray_green_only = cv2.cvtColor(green_only, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_green_only, 170, 255, cv2.THRESH_BINARY)

        sum_binary = np.sum(binary_image)
        
        #window to prevent outliers
        if len(self.window_check) < self.window_check_length:
            self.window_check.append(sum_binary)
        else:
            self.window_check.append(sum_binary)
            self.window_check.pop(0)
        
        #median of window to prevent outliers
        median = np.median(self.window_check)
        #use this list for testing threshold value
        self.sum_binary_list.append(median)
        #if median larger threshold we are detecting a flair
        if median>self.threshold:
            #check if we are already in the flair (the flair takes some frames)
            if not self.bright:
                #special case for 5 kills (extra flair-> skip 1 flair)
                if self.kills == 5:
                    if self.count_next_flair == True:
                        self.count_next_flair = False
                    elif self.count_next_flair == False:
                        self.count_next_flair = True
                        self.kills +=1
                        kill = True
                else:
                    self.kills +=1
                    kill = True
                    
                #we are in a flair
                self.bright = True 
        else:
            self.bright = False
        return kill,self.kills,self.sum_binary_list