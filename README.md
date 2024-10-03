# Aimbot for CS2

This aimbot is based on the **YOLOv10** network, which offers a very high inference speed, and utilizes the **Win32 API** to simulate mouse clicks on the screen.

This project leverages the code and dataset provided by **Ömer Faruk Günaydın**. His contributions were instrumental in helping me understand the various components of the project, so a big shoutout to him! You can find his dataset [here](https://www.kaggle.com/datasets/merfarukgnaydn/counter-strike-2-body-and-head-classification).

## Prediction Classes
The YOLO network predicts five classes:
- **none**
- **ct_body**
- **ct_head**
- **t_body**
- **t_head**

The "none" class is included because the dataset is labeled with values 1-5, while the network is trained using a zero-based index (0-4).

## To-Do List
- Change how the mouse moves from the current position to the target head using Reinforcement Learning (RL).
- Implement TensorRT to increase inference speed.

![Aimbot Screenshot](https://github.com/WanneWisse/Aimbot-CSGO-YOLO/blob/main/Aim_screenshot.png?raw=true)
