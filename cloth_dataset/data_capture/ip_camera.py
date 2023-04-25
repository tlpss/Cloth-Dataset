import cv2
import threading



import queue
import threading

# TODO: this is not working properly... still latency... 
# so find out how to grab LATEST frame of the stream when querying at low frequency...

# class IPCamera:
#   def __init__(self, name):
#     self.cap = cv2.VideoCapture(name)
#     self.q = queue.Queue()
#     t = threading.Thread(target=self._reader)
#     t.daemon = True
#     t.start()

#   # read frames as soon as they are available, keeping only most recent one
#   def _reader(self):
#     while True:
#       ret, frame = self.cap.read()
#       if not ret:
#         break
#       if not self.q.empty():
#         try:
#           self.q.get_nowait()   # discard previous (unprocessed) frame
#         except queue.Empty:
#           pass
#       self.q.put(frame)

#   def get_rgb_image(self):
#     return self.q.get()

from PIL import Image
import requests
import numpy as np
class IPCamera: 
    """shortcut to Opencv imagecap to get latest frame from smartphone as IP camera using android app."""
    def __init__(self,ip_address: str):
        self.ip_address = ip_address
    def get_rgb_image(self):
        url = f"http://{self.ip_address}:8080/photo.jpg"
        img = Image.open(requests.get(url, stream=True).raw)
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

if __name__ == "__main__":
    import time 
    camera = IPCamera("192.168.1.11")

    while True:
        img = camera.get_rgb_image()
        #time.sleep(2)
        #time.sleep(1.0)
        img = cv2.resize(img, (640, 480))
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

