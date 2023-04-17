from dataclasses import dataclass
from typing import List
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.interfaces import RGBDCamera
from airo_camera_toolkit.calibration.fiducial_markers import detect_aruco_markers, detect_charuco_corners, get_pose_of_charuco_board, AIRO_DEFAULT_ARUCO_DICT, AIRO_DEFAULT_CHARUCO_BOARD
from airo_camera_toolkit.utils import ImageConverter
import cv2 
from airo_typing import HomogeneousMatrixType
import loguru

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
    distractors: List[DistractorConfig]
    capture_date: str
    camera: str
    


def _convert_cartesian_to_sperical_coordinates(x,y,z):
    r  = np.sqrt(x**2 + y**2 + z**2)
    polar_angle = np.arccos(z/r)
    azimuthal_angle = np.arctan2(y,x)
    return r, polar_angle,azimuthal_angle

def _convert_spherical_to_cartesian_coordinates(r, polar_angle, azimuthal_angle):
    x = r * np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = r * np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = r * np.cos(polar_angle)
    return x,y,z

class ClothDatasetCapturer:
    """Scope of this capturer is to capture a set of cloth items from a single category in one scene (camera setup + environment).
    Make a new object for every time you wish to do this."""

    def __init__(self, camera: RGBDCamera, cloth_type: str, location_id: int, root_folder: str) -> None:
        pass
        self._camera = camera
        self.charuco_board = AIRO_DEFAULT_CHARUCO_BOARD
        self.aruco_dict = AIRO_DEFAULT_ARUCO_DICT


    def _get_charuco_pose(self,image):
        intrinsics = self._camera.intrinsics_matrix()
        aruco_result = detect_aruco_markers(image, self.aruco_dict)

        if not aruco_result:
            return None
        charuco_result = detect_charuco_corners(image, aruco_result, self.charuco_board)

        if not charuco_result:
            return None
        
        charuco_pose = get_pose_of_charuco_board(charuco_result, self.charuco_board, intrinsics)
        return charuco_pose
        
    def camera_setup(self):
        # determine camera extrinsics
        desired_camera_position = self.sample_camera_pose()
        
        #desired orientation:
        # 1. look at the origin of the marker
        # 2. "horizontal" orientation along that ray


        camera_is_in_place = False
        print("Please place the camera at the desired position by comparing the spherical coordinates while keeping the camera centered on the origin of the marker. Press a key to continue with the current pose.")
        while True:
            image = self._camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format
            charuco_pose = self._get_charuco_pose(image)
            if charuco_pose is None:
                logger.info("No charuco board detected")
                cv2.imshow("Camera pose", image)
                cv2.waitKey(1)
                continue
            camera_pose_in_charuco_frame = np.linalg.inv(charuco_pose) 
            spherical_position_coordinates = _convert_cartesian_to_sperical_coordinates(*camera_pose_in_charuco_frame[:3,3])
            logger.info(f"Spherical coordinates of the camera position: {spherical_position_coordinates}")
            logger.info(f"desired spherical coordinates of the camera position: {desired_camera_position}")
            if np.min(np.abs(spherical_position_coordinates - desired_camera_position)) < 0.05 and abs(charuco_pose[0,3]) < 0.02 and abs(charuco_pose[1,3]) < 0.02:
                camera_is_in_place = True
                self.camera_extrinsics = charuco_pose
                logger.info("Camera is in place.")

            cv2.imshow("Camera pose", image)
            if cv2.waitKey(1) & 0xFF == ord('s'):
                if not camera_is_in_place:
                    logger.error("Camera is not in place. Please try again.")
                    continue
        
                # TODO: write the camera extrinsics to a metafile
        
            
        # determine cloth area
        #TODO:


    def capture_cloth_data_at_location(self):
        n_cloth_items = None
        n_images_per_cloth_item = None

        assert self.camera_extrinsics is not None


        for cloth_id in range(n_cloth_items):

            for cloth_shot_id in range(n_images_per_cloth_item):
                # get random cloth pose
                pose = self.get_random_cloth_pose()

                # TODO visualize stream in opencv window with the pose
                # wait for user to press key to confirm cloth pose

                if cloth_shot_id == 0:
                    # perfectly unflattened cloth.
                    pass
                else:
                    # determine appropriate randomizations

                    # visualize stream in opencv window to guide through randomization steps
                    pass

                    # sample distractors and place them on the AREA - CLOTH bbox * 1.5

                # capture image

                pass
                
            # wait for user to press key to confirm cloth pose


        pass

    def get_random_cloth_pose(self):
        # get random cloth pose
        
        # sample a 2D pose for the cloth
        # has to fit in the folding area
        pass

    def sample_camera_pose(self):
        # sample a camera pose
        # has to be in the folding area
        distance = np.random.uniform(0.5, 1.5)
        polar_angle = np.random.uniform(0, np.pi/2)
        azimuthal_angle = np.random.uniform(- np.pi/3, np.pi/3)

        return np.array([distance, polar_angle, azimuthal_angle])
        # create a pose from the spherical coordinates

    

def is_2dpoint_in_2dpolygon(point, polygon_points):
    pass


@dataclass
class RandomizationParameters:
    pass

@dataclass
class ClothDatasetCapturerConfig:
    cloth_type: str 
    data_folder: str
    randomization_parameters: RandomizationParameters


if __name__ == "__main__":

    # determine cloth type
    # specify folder to store data to
    # configure randomization parameters
    # configure camera parameters

    # create capturer

    # start capturing
    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K,fps = 15, depth_mode= Zed2i.NEURAL_DEPTH_MODE)
    capturer = ClothDatasetCapturer(camera,'Test',0, 'data')
    capturer.camera_setup()