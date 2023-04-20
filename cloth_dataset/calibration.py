import cv2
import numpy as np
from airo_typing import OpenCVIntImageType

def collect_click_points_on_image(image: OpenCVIntImageType) -> np.ndarray:
    """collects clicks and converts them to list of (x,y) coordinates in the original image size"""
    image_shape = image.shape
    image_height, image_width,_ = image_shape
    resized_shape = (1280,720)
    resized_image = cv2.resize(image, resized_shape)


    current_mouse_point = [(0, 0)]  # has to be a list so that the callback can edit it
    clicked_image_points = []

    def mouse_callback(event, x, y, flags, parm):

        if event == cv2.EVENT_LBUTTONDOWN:
            clicked_image_points.append((x, y))
            print(f"Clicked point {len(clicked_image_points)}: {x}, {y}")
        elif event == cv2.EVENT_MOUSEMOVE:
            current_mouse_point[0] = x, y

    def draw_clicked_grasp(image, clicked_image_points, current_mouse_point):
        """If we don't have tow clicks yet, draw a line between the first point and the current cursor position."""
        for point in clicked_image_points:
            image = cv2.circle(image, point, 2, (0, 255, 0), thickness=2)
        return image 

    window_name = "Camera feed"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    augmented_image = resized_image.copy()
    print("Press any key once all points are clicked.")
    while True:
        augmented_image = draw_clicked_grasp(augmented_image, clicked_image_points, current_mouse_point)

        cv2.imshow("Camera feed", augmented_image)

        if cv2.waitKey(1) != -1:
            cv2.destroyAllWindows()
            break
    
    # convert the clicked points to the original image size
    clicked_image_points = np.array(clicked_image_points).astype(np.float32)
    clicked_image_points = clicked_image_points * np.array([image_width, image_height]) / np.array(resized_shape)
    return clicked_image_points

if __name__ == "__main__":
    image = np.ones((5000, 1000,3), dtype=np.uint8)
    clicked_points = collect_click_points_on_image(image)
    print(clicked_points)