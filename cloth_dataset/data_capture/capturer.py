from dataclasses import dataclass
import datetime
from typing import List
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.calibration.fiducial_markers import detect_aruco_markers, detect_charuco_corners, get_pose_of_charuco_board, AIRO_DEFAULT_ARUCO_DICT, AIRO_DEFAULT_CHARUCO_BOARD, draw_frame_on_image
from airo_camera_toolkit.utils import ImageConverter
import cv2 
from airo_camera_toolkit.reprojection import reproject_to_frame
from airo_typing import HomogeneousMatrixType, Vector3DArrayType, Vector3DType
from airo_spatial_algebra import SE3Container
import loguru
import numpy as np 
import pathlib
from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from cloth_dataset.data_capture.deformation_sampling import sample_deformation_instructions
import imageio
import dataclasses
import json 

from cloth_dataset.keypoint_names import CLOTH_CATEGORIES, CATEGORY_TO_KEYPOINTS
np.set_printoptions(precision=3)

logger = loguru.logger

import numpy as np 
@dataclass
class DistractorConfig:
    distractor_id: int
    distractor_pose: HomogeneousMatrixType

@dataclass
class ImageMetaData:
    dataset_split: str
    location_id: int
    cloth_type: str
    cloth_id: int
    capture_date_time: str
    camera: str

@dataclass
class SceneMetaData:
    dataset_split: str
    location_id: int
    capture_date_time: str



class ClothDatasetCapturer:
    """Scope of this capturer is to capture a set of clothes (can be multiple categories) in one scene (camera setup + environment).
    Make a new object for every time you wish to do this."""

    def __init__(self, zed_camera: StereoRGBDCamera, location_id: int, root_folder: str, split: str = "train") -> None:
        self.zed_camera = zed_camera
        self.location_id = location_id
        self.root_folder = root_folder
        self.split = split
        self.data_folder = pathlib.Path(root_folder) / split / f"location_{location_id}"
        if self.data_folder.exists():
            raise ValueError(f"Data folder {self.data_folder} already exists. This should not be the case..")
        self.data_folder.mkdir(parents=True, exist_ok=False)
        for category in CLOTH_CATEGORIES:
            (self.data_folder / category).mkdir(parents=True, exist_ok=False)

        self.charuco_board = AIRO_DEFAULT_CHARUCO_BOARD
        self.aruco_dict = AIRO_DEFAULT_ARUCO_DICT


        self.zed_camera_extrinsics = None



    def _get_charuco_pose(self,image):
        intrinsics = self.zed_camera.intrinsics_matrix()
        aruco_result = detect_aruco_markers(image, self.aruco_dict)

        if not aruco_result:
            return None
        charuco_result = detect_charuco_corners(image, aruco_result, self.charuco_board)

        if not charuco_result:
            return None
        
        charuco_pose = get_pose_of_charuco_board(charuco_result, self.charuco_board, intrinsics)
        return charuco_pose
        

    def setup_camera_pose(self,desired_camera_position):
        
        #desired orientation:
        # 1. look at the origin of the marker
        # 2. "horizontal" orientation along that ray


        camera_is_in_place = False
        logger.info("Please place the camera at the desired position by comparing the spherical coordinates while keeping the camera centered on the origin of the marker. Press a key to continue with the current pose.")
        while True:
            image = self.zed_camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format
            charuco_pose = self._get_charuco_pose(image)
            logger.debug(f"charuco pose: \n {charuco_pose}")
            if charuco_pose is None:
                logger.info("No charuco board detected")
                image = cv2.resize(image, (1280,720))
                cv2.imshow("Camera pose", image)
                cv2.waitKey(1)
                continue
            charuco_center_pose_in_default_frame = SE3Container.from_euler_angles_and_translation(np.array([np.pi,0,0]), np.array([0.14,0.10,0]))
            charuco_center_pose = charuco_pose @ charuco_center_pose_in_default_frame.homogeneous_matrix
            camera_pose_in_charuco_center_frame = np.linalg.inv(charuco_center_pose) 
            camera_position_in_charuco_center_frame = camera_pose_in_charuco_center_frame[:3,3]
            logger.info(f"camera position: {camera_position_in_charuco_center_frame}")
            logger.info(f"desired camera position: {desired_camera_position}")
            if np.max(np.abs(camera_position_in_charuco_center_frame - desired_camera_position)) < 0.1 and abs(charuco_center_pose[0,3]) < 0.1 and abs(charuco_center_pose[1,3]) < 0.1:
                camera_is_in_place = True
                logger.info("Camera is in place.")

            image = draw_frame_on_image(image, charuco_center_pose, self.zed_camera.intrinsics_matrix())
            image = cv2.resize(image, (1280,720))
            image = cv2.circle(image, (640,360), 10, (0,0,255), -1)

            cv2.imshow("Camera pose", image)
            if cv2.waitKey(1) != -1:
                if not camera_is_in_place:
                    logger.error("Camera is not in place. Please try again.")
                    continue
                break
        cv2.destroyAllWindows()
        image = self.zed_camera.get_rgb_image()
        name = "charuco_board.png"
        #TODO: save calibration image & pose.

        return charuco_center_pose
    

    def camera_setup(self):
        # determine camera extrinsics
        desired_camera_position = self.sample_camera_position()
        extrinsics = self.setup_camera_pose(desired_camera_position)
        self.zed_camera_extrinsics = extrinsics           

    def capture_cloth_data_at_location(self, cloth_type:str, location_id: int, n_cloth_items: int, n_images_per_cloth_item: int):

        assert self.zed_camera_extrinsics is not None

        cloth_type = cloth_type.lower()
        assert cloth_type in CLOTH_CATEGORIES
        cloth_keypoints = CATEGORY_TO_KEYPOINTS[cloth_type]

        for cloth_id in range(n_cloth_items):

            for cloth_shot_id in range(n_images_per_cloth_item):
                # get random cloth pose
                cloth_pose_in_maker_frame = self.get_random_cloth_pose()
                print(cloth_pose_in_maker_frame)
                cloth_pose_in_camera_frame = self.zed_camera_extrinsics @ cloth_pose_in_maker_frame
                should_front_be_up = np.random.uniform() > 0.4
                logger.info(f"cloth front side shoud be up: {should_front_be_up}")
                
                # determine appropriate randomizations
                logger.info("deformation instructions:")
                deformation_instructions = sample_deformation_instructions(cloth_keypoints)
                for deformation_instruction in deformation_instructions:
                    logger.info(deformation_instruction)

                logger.info("Press any key once the cloth is positioned and deformed as required.")

                while True:
                    image = self.zed_camera.get_rgb_image()
                    image = ImageConverter.from_numpy_format(image).image_in_opencv_format
                    # draw cloth id and shot id
                    image = cv2.putText(image, f"category: {cloth_type}, cloth id: {cloth_id}, shot id: {cloth_shot_id}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    # draw cloth position
                    image = draw_cloth_pose_on_image(image, cloth_pose_in_camera_frame, self.zed_camera.intrinsics_matrix())
                    for i,deformation_instruction in enumerate(deformation_instructions):
                        image = cv2.putText(image, deformation_instruction, (10, 100 + 30*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    image = cv2.resize(image, (1280,720))
                    cv2.imshow("Cloth pose", image)
                    if cv2.waitKey(1) != -1:
                        break

                # capture image
                date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_name_prefix = f"location_id_{self.location_id}_cloth_id_{cloth_id}_{date_time}"
                image_name_prefix = str(self.data_folder / cloth_type) + "/" + image_name_prefix
                image = self.zed_camera.get_rgb_image()
                image = ImageConverter.from_numpy_format(image).image_in_opencv_format
                depth_map = self.zed_camera.get_depth_map()
                right_image = self.zed_camera.get_rgb_image("right")
                right_image = ImageConverter.from_numpy_format(right_image).image_in_opencv_format
           
                meta_data = ImageMetaData(self.split,self.location_id,cloth_type,cloth_id,date_time,"ZED2i")
                imageio.imsave(f"{image_name_prefix}_zed_rgb.png", image)
                np.save(f"{image_name_prefix}_zed_depth_map.npy", depth_map)
                imageio.imsave(f"{image_name_prefix}_zed_right.png", right_image)
                json.dump(dataclasses.asdict(meta_data), open(f"{image_name_prefix}_meta_data.json", "w"))
                
            # wait for user to press key to confirm cloth pose


        pass

    def get_random_cloth_pose(self) -> HomogeneousMatrixType:
        # get random cloth pose
        
        # sample a 2D pose for the cloth
        # has to fit in the folding area
        random_orientation = np.random.uniform(0,2*np.pi)
        position_range = 0.2
        x = np.random.uniform(-position_range,position_range)
        y = np.random.uniform(-position_range,position_range)
        return SE3Container.from_euler_angles_and_translation(np.array([0,0,random_orientation]), np.array([x,y,0])).homogeneous_matrix



    def sample_camera_position(self) -> Vector3DType:
        z_distance = np.random.uniform(0.5, 1.0)
        y_distance = np.random.uniform(-0.2, -0.6)
        x_distance = np.random.uniform(-0.1, 0.1)
        return np.array([0,-0.2,0.7])
        return np.array([x_distance, y_distance, z_distance])

    

def draw_cloth_pose_on_image(image, cloth_pose_in_camera_frame, camera_intrinsics):
    cloth_position = cloth_pose_in_camera_frame[:3,3]
    cloth_x_orientation = cloth_pose_in_camera_frame[:3,0]
    arrow_end = cloth_position + 0.1*cloth_x_orientation
    cloth_position_pixels = project_frame_to_image_plane(cloth_position, camera_intrinsics)[0]
    arrow_end_pixels = project_frame_to_image_plane(arrow_end, camera_intrinsics)[0]
    image = cv2.circle(image, (int(cloth_position_pixels[0]), int(cloth_position_pixels[1])), 10, (0,0,255), -1)
    image = cv2.arrowedLine(image, (int(cloth_position_pixels[0]), int(cloth_position_pixels[1])), (int(arrow_end_pixels[0]), int(arrow_end_pixels[1])), (0,0,255), 2)
    return image


@dataclass
class RandomizationParameters:
    pass

@dataclass
class ClothDatasetCapturerConfig:
    cloth_types: str 
    data_folder: str
    randomization_parameters: RandomizationParameters


if __name__ == "__main__":

    # determine cloth type
    # specify folder to store data to
    # configure randomization parameters
    # configure camera parameters

    # create capturer

    # start capturing
    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K,fps = 15, depth_mode= Zed2i.PERFORMANCE_DEPTH_MODE)
    capturer = ClothDatasetCapturer(camera,0,'TestDataset', 'train')
    capturer.camera_setup()
    capturer.capture_cloth_data_at_location("tshirts",0,3,2)