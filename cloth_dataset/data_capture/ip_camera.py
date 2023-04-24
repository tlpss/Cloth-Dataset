import cv2
from airo_camera_toolkit.utils import ImageConverter

class IPCamera:
    def __init__(self, ip_address: str) -> None:
        self.cap = cv2.VideoCapture(f"http://{ip_address}:8080/video")

    def get_rgb_image(self):
        ret, frame = self.cap.read()
        return frame


if __name__ == "__main__":
    camera = IPCamera("192.168.1.57")
    img = camera.get_rgb_image()
    print(img.shape)
    print(img.dtype)
    img = cv2.resize(img, (640, 480))
    cv2.imshow("image", img)
    cv2.imwrite("test.jpg", img)
    input("k")
    k = cv2.waitKey(50)

