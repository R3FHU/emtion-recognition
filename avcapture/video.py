import cv2
import time

class VideoCapture:
    def __init__(self, cam_id=0,start_time=None):
        self.cap = cv2.VideoCapture(cam_id)
        if not self.cap.isOpened():
            raise RuntimeError("无法打开摄像头")
        self.start_time=start_time or time.perf_counter()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        timestamp = time.perf_counter()-self.start_time
        return frame, timestamp

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()