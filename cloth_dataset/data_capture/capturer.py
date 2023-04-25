import dataclasses
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Union

import cv2
import loguru
import numpy as np
from airo_camera_toolkit.calibration.fiducial_markers import (
    AIRO_DEFAULT_ARUCO_DICT,
    AIRO_DEFAULT_CHARUCO_BOARD,
    detect_aruco_markers,
    detect_charuco_corners,
    draw_frame_on_image,
    get_pose_of_charuco_board,
)
from airo_camera_toolkit.cameras.zed2i import Zed2i
from airo_camera_toolkit.interfaces import StereoRGBDCamera
from airo_camera_toolkit.reprojection import project_frame_to_image_plane
from airo_camera_toolkit.utils import ImageConverter
from airo_spatial_algebra import SE3Container
from airo_typing import HomogeneousMatrixType, Vector3DType

from cloth_dataset.data_capture.deformation_sampling import sample_deformation_instructions
from cloth_dataset.data_capture.ip_camera import IPCamera
from cloth_dataset.keypoint_names import CATEGORY_TO_KEYPOINTS, CLOTH_CATEGORIES

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
    charuco_pose_in_camera_frame: list
    charuco_pose_in_smarthpone_frame: Union[str, list] = ""


DatasetConfig = {
    "train": {
        "tshirts": 15,
        "shorts": 8,
        "towels": 15,
        "boxershorts": 11,
    },
    "test": {
        "tshirts": 20,
        "shorts": 9,
        "towels": 20,
        "boxershorts": 11,
    },
    "development": {
        "tshirts": 2,
        "shorts": 1,
        "towels": 1,
        "boxershorts": 1,
    },
}


class ClothDatasetCapturer:
    """Scope of this capturer is to capture a set of clothes (can be multiple categories) in one scene (camera setup + environment).
    Make a new object for every time you wish to do this."""

    def __init__(
        self,
        zed_camera: StereoRGBDCamera,
        location_id: int,
        root_folder: str,
        split: str = "train",
        ip_camera: IPCamera = None,
    ) -> None:

        self.n_images_per_cloth_item = 2
        self.zed_camera_extrinsics = None


        assert split in ("train", "test", "development")
        self.location_id = location_id
        self.root_folder = root_folder
        self.split = split
        self.data_folder = pathlib.Path(root_folder) / split / f"location_{location_id}"

        if self.data_folder.exists():
            logger.warning("data folder already exists. Will overwrite existing data...")
            charuco_pose = json.load(open(self.data_folder /  "scene.json"))
            charuco_pose = np.array(charuco_pose["charuco_pose_in_camera_frame"])
            print(charuco_pose)
            charuco_center_pose_in_default_frame = SE3Container.from_euler_angles_and_translation(
                np.array([np.pi, 0, 0]), np.array([0.14, 0.10, 0])
            )
            charuco_center_pose = charuco_pose @ charuco_center_pose_in_default_frame.homogeneous_matrix
            self.zed_camera_extrinsics = charuco_center_pose
        else:
            self.data_folder.mkdir(parents=True, exist_ok=False)
            for category in CLOTH_CATEGORIES:
                (self.data_folder / category).mkdir(parents=True, exist_ok=False)

        self.charuco_board = AIRO_DEFAULT_CHARUCO_BOARD
        self.aruco_dict = AIRO_DEFAULT_ARUCO_DICT

        self.zed_camera = zed_camera

        self.ip_camera = ip_camera

    def _get_charuco_pose(self, image):
        intrinsics = self.zed_camera.intrinsics_matrix()
        aruco_result = detect_aruco_markers(image, self.aruco_dict)

        if not aruco_result:
            return None
        charuco_result = detect_charuco_corners(image, aruco_result, self.charuco_board)

        if not charuco_result:
            return None

        charuco_pose = get_pose_of_charuco_board(charuco_result, self.charuco_board, intrinsics)
        return charuco_pose

    def setup_camera_pose(self, desired_camera_position):

        # desired orientation:
        # 1. look at the origin of the marker
        # 2. "horizontal" orientation along that ray

        camera_is_in_place = False
        input("press enter to start camera pose setup. Press any key once the camera is positioned.")
        while True:
            image = self.zed_camera.get_rgb_image()
            image = ImageConverter.from_numpy_format(image).image_in_opencv_format
            charuco_pose = self._get_charuco_pose(image)
            logger.debug(f"charuco pose: \n {charuco_pose}")
            if charuco_pose is None:
                logger.info("No charuco board detected")
                image = cv2.resize(image, (1280, 720))
                cv2.imshow("Camera pose", image)
                cv2.waitKey(1)
                continue
            charuco_center_pose_in_default_frame = SE3Container.from_euler_angles_and_translation(
                np.array([np.pi, 0, 0]), np.array([0.14, 0.10, 0])
            )
            charuco_center_pose = charuco_pose @ charuco_center_pose_in_default_frame.homogeneous_matrix
            camera_pose_in_charuco_center_frame = np.linalg.inv(charuco_center_pose)
            camera_position_in_charuco_center_frame = camera_pose_in_charuco_center_frame[:3, 3]
            logger.info(f"camera position: {camera_position_in_charuco_center_frame}")
            logger.info(f"desired camera position: {desired_camera_position}")
            if (
                np.max(np.abs(camera_position_in_charuco_center_frame - desired_camera_position)) < 0.1
                and abs(charuco_center_pose[0, 3]) < 0.1
                and abs(charuco_center_pose[1, 3]) < 0.1
            ):
                camera_is_in_place = True
                logger.info("Camera is in place.")

            image = draw_frame_on_image(image, charuco_center_pose, self.zed_camera.intrinsics_matrix())
            image = cv2.resize(image, (1280, 720))
            image = cv2.circle(image, (640, 360), 10, (0, 0, 255), -1)

            cv2.imshow("Camera pose", image)
            if cv2.waitKey(1) != -1:
                if not camera_is_in_place:
                    logger.error("Camera is not in place. Please try again.")
                    continue
                break
        cv2.destroyAllWindows()
        image = self.zed_camera.get_rgb_image()
        image = ImageConverter.from_numpy_format(image).image_in_opencv_format
        name = "charuco_board.png"
        cv2.imwrite(str(self.data_folder / name), image)

        if self.ip_camera:
            ip_cam_image = self.ip_camera.get_rgb_image()
            name = "charuco_board_smartphone.png"
            cv2.imwrite(str(self.data_folder / name), ip_cam_image)



        metadata = SceneMetaData(
            self.split,
            self.location_id,
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            charuco_pose.tolist(),
        )

        input("Remove the board and press enter to continue.")
        image = self.zed_camera.get_rgb_image()
        image = ImageConverter.from_numpy_format(image).image_in_opencv_format
        cv2.imwrite(str(self.data_folder / "scene.png"), image)

        if self.ip_camera:
            ip_scene_image = self.ip_camera.get_rgb_image()
            name = "scene_smartphone.png"
            cv2.imwrite(str(self.data_folder / name), ip_scene_image)
            smartphone_charuco_pose = self._get_charuco_pose(ip_cam_image)
            if smartphone_charuco_pose is None:
                logger.error("No charuco board detected on smartphone image.")
            else:
                metadata.charuco_pose_in_smarthpone_frame = smartphone_charuco_pose.tolist()

        json.dump(dataclasses.asdict(metadata), open(str(self.data_folder / "scene.json"), "w"))

        return charuco_center_pose

    def camera_setup(self):
        # determine camera extrinsics
        desired_camera_position = self.sample_camera_position()
        extrinsics = self.setup_camera_pose(desired_camera_position)
        self.zed_camera_extrinsics = extrinsics

    def capture_cloth_data_at_location(self, cloth_type: str):

        assert self.zed_camera_extrinsics is not None

        cloth_type = cloth_type.lower()
        assert cloth_type in CLOTH_CATEGORIES
        cloth_keypoints = CATEGORY_TO_KEYPOINTS[cloth_type]

        n_cloth_items = DatasetConfig[self.split][cloth_type]
        n_images_per_cloth_item = self.n_images_per_cloth_item
        for cloth_id in range(1, n_cloth_items + 1):

            for cloth_shot_id in range(1, n_images_per_cloth_item + 1):
                # get random cloth pose
                cloth_pose_in_maker_frame = self.sample_cloth_pose()
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
                    image = cv2.putText(
                        image,
                        f"category: {cloth_type}, cloth id: {cloth_id}/{n_cloth_items}, shot id: {cloth_shot_id}/{n_images_per_cloth_item}. Cloth front side should be up: {should_front_be_up}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    # draw cloth position
                    image = draw_cloth_pose_on_image(
                        image, cloth_pose_in_camera_frame, self.zed_camera.intrinsics_matrix()
                    )
                    for i, deformation_instruction in enumerate(deformation_instructions):
                        image = cv2.putText(
                            image,
                            deformation_instruction,
                            (10, 100 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    image = cv2.resize(image, (1280, 720))
                    cv2.imshow("Cloth pose", image)
                    if cv2.waitKey(1) != -1:
                        break

                # capture & save images and depth maps
                date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                image_name_prefix = f"{date_time}"
                image_name_prefix = str(self.data_folder / cloth_type) + "/" + image_name_prefix

                image = self.zed_camera.get_rgb_image()
                image = ImageConverter.from_numpy_format(image).image_in_opencv_format

                depth_map = self.zed_camera.get_depth_map()
                depth_image = self.zed_camera.get_depth_image()
                cv2.imwrite(f"{image_name_prefix}_depth_image_zed.png", depth_image)

                right_image = self.zed_camera.get_rgb_image("right")
                right_image = ImageConverter.from_numpy_format(right_image).image_in_opencv_format

                meta_data = ImageMetaData(self.split, self.location_id, cloth_type, cloth_id, date_time, "ZED2i")
                logger.debug("started saving images")
                cv2.imwrite(f"{image_name_prefix}_rgb_zed.png", image)
                np.save(f"{image_name_prefix}_depth_map_zed.npy", depth_map)
                cv2.imwrite(f"{image_name_prefix}_rgb_zed_right.png", right_image)

                if self.ip_camera:
                    ip_cam_image = self.ip_camera.get_rgb_image()
                    cv2.imwrite(f"{image_name_prefix}_rgb_smartphone.png", ip_cam_image)
                json.dump(dataclasses.asdict(meta_data), open(f"{image_name_prefix}_meta_data.json", "w"))
                logger.debug("finished saving images")

    def sample_cloth_pose(self) -> HomogeneousMatrixType:
        random_orientation = np.random.uniform(0, 2 * np.pi)
        position_range = 0.2
        x = np.random.uniform(-position_range, position_range)
        y = np.random.uniform(-position_range, position_range)
        return SE3Container.from_euler_angles_and_translation(
            np.array([0, 0, random_orientation]), np.array([x, y, 0])
        ).homogeneous_matrix

    def sample_camera_position(self) -> Vector3DType:
        z_distance = np.random.uniform(0.5, 1.0)
        y_distance = np.random.uniform(-0.2, -0.6)
        x_distance = np.random.uniform(-0.1, 0.1)
        return np.array([x_distance, y_distance, z_distance])


def draw_cloth_pose_on_image(image, cloth_pose_in_camera_frame, camera_intrinsics):
    cloth_position = cloth_pose_in_camera_frame[:3, 3]
    cloth_x_orientation = cloth_pose_in_camera_frame[:3, 0]
    arrow_end = cloth_position + 0.1 * cloth_x_orientation
    cloth_position_pixels = project_frame_to_image_plane(cloth_position, camera_intrinsics)[0]
    arrow_end_pixels = project_frame_to_image_plane(arrow_end, camera_intrinsics)[0]
    image = cv2.circle(image, (int(cloth_position_pixels[0]), int(cloth_position_pixels[1])), 10, (0, 0, 255), -1)
    image = cv2.arrowedLine(
        image,
        (int(cloth_position_pixels[0]), int(cloth_position_pixels[1])),
        (int(arrow_end_pixels[0]), int(arrow_end_pixels[1])),
        (0, 0, 255),
        2,
    )
    return image


if __name__ == "__main__":
    # TODO: USE NEURAL DEPTH MODE

    camera = Zed2i(resolution=Zed2i.RESOLUTION_2K, fps=15, depth_mode=Zed2i.NEURAL_DEPTH_MODE)
    smartphone_camera = IPCamera("192.168.1.11")
    capturer = ClothDatasetCapturer(camera, 7, "Dataset", "test", smartphone_camera)
    capturer.camera_setup()
    capturer.capture_cloth_data_at_location("tshirts")
    capturer.capture_cloth_data_at_location("shorts")
    capturer.capture_cloth_data_at_location("towels")
    capturer.capture_cloth_data_at_location("boxershorts")
