from ultralytics import YOLO
import multiprocessing

def train_model():
    # Load the pre-trained YOLOv8 model
    model = YOLO('trained_yolo_model.pt')
    results = model.train(data='datasets/data.yaml', imgsz=320, epochs=300, batch=16)

    # Save the trained model (optional)
    model.save('trained_yolo_model.pt')

    metrics = model.val(data='datasets/data.yaml')
    print(metrics)
    return results

if __name__ == '__main__':
    multiprocessing.freeze_support()  # Optional
    train_process = multiprocessing.Process(target=train_model)
    train_process.start()
    train_process.join()