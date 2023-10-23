import numpy as np
from airo_dataset_tools.coco_tools.split_dataset import split_coco_dataset
from airo_dataset_tools.coco_tools.transform_dataset import resize_coco_keypoints_dataset
from airo_dataset_tools.cvat_labeling.convert_cvat_to_coco import cvat_image_to_coco
from airo_dataset_tools.data_parsers.coco import CocoKeypointsDataset

from artf_clothes import DATA_DIR
from artf_clothes.scripts.keypoint_ordering import order_keypoints

np.random.seed(2023)  # fix seed to make splits reproducible


# make sure the image paths are relative to the dataset root before converting to coco.

resizing_resolutions = [(512, 256)]

coco_categories_file = DATA_DIR / "artf_clothes_coco_categories.json"
artf_dir = DATA_DIR / "aRTFClothes"

TOWEL_TRAIN_NAME = "towels-train"
TOWEL_VAL_NAME = "towels-val"
TOWEL_TEST_NAME = "towels-test"
TSHIRT_TRAIN_NAME = "tshirts-train"
TSHIRT_VAL_NAME = "tshirts-val"
TSHIRT_TEST_NAME = "tshirts-test"
ALL_TRAIN = "all-train"
ALL_TEST = "all-test"

cvat_xml_base_names = [TOWEL_TRAIN_NAME, TOWEL_TEST_NAME, TSHIRT_TRAIN_NAME, TSHIRT_TEST_NAME]

train_names = [TOWEL_TRAIN_NAME, TSHIRT_TRAIN_NAME]
val_names = [TOWEL_VAL_NAME, TSHIRT_VAL_NAME]
test_names = [TOWEL_TEST_NAME, TSHIRT_TEST_NAME]
all_names = [ALL_TRAIN, ALL_TEST]

resize_names = []
resize_names.extend(train_names)
resize_names.extend(val_names)
resize_names.extend(test_names)
# resize_names.append(all_names)


coco_jsons = []
for file_base_name in cvat_xml_base_names:
    # 1. convert cvat to  coco using the categories file
    coco_dict = cvat_image_to_coco(
        artf_dir / f"{file_base_name}.xml", coco_categories_file, add_bbox=False, add_segmentation=True
    )

    # order the keypoints.
    coco_keypoints_dataset = CocoKeypointsDataset(**coco_dict)
    category_id_to_name_dict = {category.id: category.name for category in coco_keypoints_dataset.categories}

    for annotation in coco_keypoints_dataset.annotations:
        keypoints_2D = np.array(annotation.keypoints).reshape(-1, 3)
        keypoints_2D = order_keypoints(category_id_to_name_dict[annotation.category_id], keypoints_2D, annotation.bbox)
        annotation.keypoints = keypoints_2D.flatten().tolist()

    coco_dict = coco_keypoints_dataset.dict()
    with open(artf_dir / f"{file_base_name}.json", "w") as f:
        import json

        json.dump(coco_dict, f, indent=4, sort_keys=False)

    #  split the train dataset into train and val
    if file_base_name in train_names:
        coco_train, coco_val = split_coco_dataset(coco_keypoints_dataset, [0.9, 0.1], shuffle_before_splitting=True)
        with open(artf_dir / f"{file_base_name}.json", "w") as f:
            json.dump(coco_train.dict(), f, indent=4, sort_keys=False)
        with open(artf_dir / f"{file_base_name.replace('-train','-val')}.json", "w") as f:
            json.dump(coco_val.dict(), f, indent=4, sort_keys=False)

#  created combined dataset
# TODO:


#  for the individual and the combined datasets: create resized versions
for name in resize_names:
    for resolution in resizing_resolutions:
        resize_coco_keypoints_dataset(artf_dir / f"{name}.json", resolution[0], resolution[1])
