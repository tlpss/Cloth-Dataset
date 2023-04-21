import cv2

class IPCamera:
    def __init__(self,ip_address: str) -> None:
        self.cap = cv2.VideoCapture(f"http://{ip_address}:8080")
    def get_rgb_image(self):
        ret,frame = self.cap.read()
        return frame

if __name__ == "__main__":
    camera = IPCamera("172.17.202.67")
    img = camera.get_rgb_image()
    print(img.shape)
    cv2.imshow("image", img)