import pygetwindow as gw
import dxcam
class Camera():
    def __init__(self):
        pass

    def get_window_rect(self, window_title):
        window = gw.getWindowsWithTitle(window_title)[0]
        print(window)
        return window.left, window.top, window.width, window.height


    def start_dxcam(self,region,fps):
        dx = dxcam.create(device_idx=0, output_idx=0, output_color="BGR", max_buffer_len=64)
        dx.start(region=region, target_fps=fps)
        #print(f"Recording window '{WINDOW_TITLE}' at ({left}, {top}) with size {width}x{height}")
        return dx