from ultralytics import YOLO
import multiprocessing

def train_model():
    # Load the pre-trained YOLOv8 model
    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
    results = model.train(data="RoboFlow/data.yaml", epochs=300, imgsz=640,batch=32,single_cls=True,val=False)

    # Save the trained model (optional)
    model.save('trained_yolo_single.pt')

    metrics = model.val(data='RoboFlow/data.yaml')
    print(metrics)
    return results

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Optional
    train_process = multiprocessing.Process(target=train_model)
    train_process.start()
    train_process.join()