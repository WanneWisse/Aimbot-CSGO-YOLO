Aimbot for CS2

This is aimbot is based on the yolo10 network which has very high inference speed and the win32api to click on the screen.
This project is based on the code and dataset of Ömer Faruk Günaydın! 
His code really helped me understanding all the different parts of the project, so a big shoutout to him!
Hereby a link to his dataset: https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification
The yolo network has five classes to predict, [ "none", "ct_body", "ct_head","t_body","t_head"]. 
The "none" class is there because the dataset is labeled with 1-5 while the network is trained from 0.
Todo: 
  Change how the mouse moves from current pos to head (uing RL)
  Using tensorRT to increase inference speed
![alt text](https://github.com/Aimbot-CSGO-YOLO/main/Aim_screenshot.png?raw=true)
