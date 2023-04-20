from dataclasses import dataclass
import datetime
from typing import List
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.calibration.fiducial_markers import detect_aruco_markers, detect_charuco_corners, get_pose_of_charuco_board, AIRO_DEFAULT_ARUCO_DICT, AIRO_DEFAULT_CHARUCO_BOARD, draw_frame_on_image
from airo_camera_toolkit.utils import ImageConverter
import cv2 
from cloth_dataset.calibration import collect_click_points_on_image
from airo_camera_toolkit.reprojection import reproject_to_frame
from airo_typing import HomogeneousMatrixType, Vector3DArrayType
from airo_spatial_algebra import SE3Container
import loguru
import numpy as np 
import pathlib

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

        self.charuco_board = AIRO_DEFAULT_CHARUCO_BOARD
        self.aruco_dict = AIRO_DEFAULT_ARUCO_DICT

        self.cloth_approx_rectangles = {
            "shirt": (),
            "shorts": np.array([[0.0, 0.0], [0.0, 0.0]]),
            "towels": np.array([[0.0, 0.0], [0.0, 0.0]]),

        }



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
        print("Please place the camera at the desired position by comparing the spherical coordinates while keeping the camera centered on the origin of the marker. Press a key to continue with the current pose.")
        while True:
            image = self.zed_camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format
            charuco_pose = self._get_charuco_pose(image)
            print(charuco_pose)
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
            if cv2.waitKey(1) & 0xFF == ord('s'):
                if not camera_is_in_place:
                    logger.error("Camera is not in place. Please try again.")
                    continue
                break

        image = self.zed_camera.get_rgb_image()
        name = "charuco_board.png"
        #TODO: save calibration image for all cameras.
        return charuco_pose
    
    def determine_folding_area_polygon(self, extrinsics: HomogeneousMatrixType) -> Vector3DArrayType:
        image = self.zed_camera.get_rgb_image()
        depth_map = self.zed_camera.get_depth_map()
        image = ImageConverter.from_numpy_format(image).image_in_opencv_format

        while True:
            clicked_points = collect_click_points_on_image(image)
            if len(clicked_points) != 4:
                logger.error("Please click exactly 4 points")
                continue
            confirmation = input("Is this a quadrilateral that covers the cloth folding area? (y/n)")
            if confirmation == "y":
                break
            else:
                continue
    
        print(clicked_points)
        area_corners = reproject_to_frame(clicked_points,self.zed_camera.intrinsics_matrix(),np.linalg.inv(extrinsics),depth_map)
        return area_corners

    


    def camera_setup(self):
        # determine camera extrinsics
        desired_camera_position = self.sample_camera_position()
        extrinsics = self.setup_camera_pose(desired_camera_position)
        
        self.area_corners = self.determine_folding_area_polygon(extrinsics)
            

    def capture_cloth_data_at_location(self, type:str, location_id: int, n_cloth_items: int, n_images_per_cloth_item: int):

        assert self.camera_extrinsics is not None


        for cloth_id in range(n_cloth_items):

            for cloth_shot_id in range(n_images_per_cloth_item):
                # get random cloth pose
                pose = self.get_random_cloth_pose()
                should_front_be_up = np.random.uniform() > 0.4
                print(f"cloth front side shoud be up: {should_front_be_up}")
                if cloth_shot_id == 0:
                    # perfectly unflattened cloth.
                    pass
                else:
                    # determine appropriate randomizations
                    print("deformation instructions:")
                    deformation_instructions = self.get_random_deformation_instructions()
                    for deformation_instruction in deformation_instructions:
                        print(deformation_instruction)

                input("Press a key to capture image.")
                # capture image
                date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_name = f""
                image = self.zed_camera.get_rgb_image()
                depth_map = self.zed_camera.get_depth_map()
                right_image = self.zed_camera.get_rgb_image("right")
                #TODO: save images & depth map.
                meta_data = ImageMetaData(self.split,self.location_id,type,cloth_id,date_time,"ZED2i")
                # dump meta data in json file and save image
                pass
                
            # wait for user to press key to confirm cloth pose


        pass

    def get_random_cloth_pose(self):
        # get random cloth pose
        
        # sample a 2D pose for the cloth
        # has to fit in the folding area
        random_orientation = np.random.uniform(0,2*np.pi)
        # small offsets to marker central pose.
        raise NotImplementedError

    def get_random_deformation_instructions(self) -> List[str]:
        # sample a random deformation
        return []


    def sample_camera_position(self):
        z_distance = np.random.uniform(0.5, 1.0)
        y_distance = np.random.uniform(-0.2, -0.6)
        x_distance = np.random.uniform(-0.1, 0.1)
        #return np.array([0,-0.2,0.7])
        return np.array([x_distance, y_distance, z_distance])

    



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
    capturer = ClothDatasetCapturer(camera,'Test',0, 'data')
    capturer.camera_setup()
