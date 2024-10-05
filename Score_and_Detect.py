import multiprocessing as mp
import time
from BBox_Detector import BBoxDetector
from Score_Detector import ScoreDetector


def score_detection(queue):
    """Simulates a kill detection system."""
    score_detector = ScoreDetector()
    score_detector.parse_frames()

def bounding_box_detection(queue):
    bbox_detector = BBoxDetector()
    bbox_detector.parse_frames()
    
    
if __name__ == "__main__":
    # Create a multiprocessing Queue for communication between processes
    queue = mp.Queue()

    # Create processes
    p1 = mp.Process(target=score_detection, args=(queue,))
    p2 = mp.Process(target=bounding_box_detection, args=(queue,))

    # Start processes
    p1.start()
    p2.start()

    # Join processes (this will keep the main program running)
    p1.join()
    p2.join()