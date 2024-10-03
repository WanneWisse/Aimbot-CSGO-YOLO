import cv2
import numpy as np
import pyautogui
import pygetwindow as gw
import matplotlib.pyplot as plt
import numpy as np
import time


import dxcam

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
    dx.start(region=(int(left+scale_correction+width/2-20),top+height-118,int(left+scale_correction+width/2+5),top+height-100), target_fps=FPS)
    #print(f"Recording window '{WINDOW_TITLE}' at ({left}, {top}) with size {width}x{height}")
    return dx


def record_window():
    cam = start_dxcam()
    kills = -1
    prev = None
    bright = False
    count_next_flair = True
    binary_count = 0
    window_check_length = 4
    window_check = []
    sum_binary_list = []
    while True:
        img = cam.get_latest_frame()
        frame = np.array(img)
        # Convert the image from BGR to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the range of green color in HSV
        lower_green = np.array([40, 40, 40])    # Lower bound for green
        upper_green = np.array([80, 255, 255])  # Upper bound for green

        # Create a mask that captures only the green regions
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Apply the mask to the original image to extract the green regions
        green_only = cv2.bitwise_and(frame, frame, mask=mask)

        gray_green_only = cv2.cvtColor(green_only, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_green_only, 170, 255, cv2.THRESH_BINARY)
        if kills == -1:
            prev = binary_image.copy()
            kills = 0
        else:
            

            sum_binary = np.sum(binary_image)
            
            if len(window_check) < window_check_length:
                window_check.append(sum_binary)
            else:
                window_check.append(sum_binary)
                window_check.pop(0)
            
            median = np.median(window_check)
            sum_binary_list.append(median)
            
            
            if median>41000:
                if not bright:
                    
                    if kills == 5:
                        if count_next_flair == True:
                            count_next_flair = False
                        elif count_next_flair == False:
                            count_next_flair = True
                            kills +=1
                    else:
                        kills +=1
                    print(kills)
                    bright = True
                    binary_count = 0   
                    
                else:
                    binary_count += 1
            else:
                bright = False
              
        
        # Display the frame with bounding boxes
        cv2.imshow("Recording with Bboxes", binary_image)
        
        # Stop recording with 'q'
        if cv2.waitKey(1) == ord("q"):
            break
    

    cv2.destroyAllWindows()
    cam.stop()
    xpoints = np.array([i for i in range(len(sum_binary_list))])
    ypoints = np.array(sum_binary_list)

    plt.plot(xpoints, ypoints)
    plt.show()
if __name__ == "__main__":
    WINDOW_TITLE = "Counter-Strike 2"  # Replace this with the title of the window you want to record
    FPS = 60
    record_window()