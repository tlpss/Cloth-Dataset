from airo_dataset_tools.data_parsers.coco import CocoKeypointCategory

from cloth_dataset.keypoint_names import BOXERSHORT_KEYPOINTS, SHORTS_KEYPOINTS, TOWEL_KEYPOINTS, TSHIRT_KEYPOINTS

towel_category = CocoKeypointCategory(
    id=0, name="towel", keypoints=TOWEL_KEYPOINTS, skeleton=[[0, 1], [1, 2], [2, 3]], supercategory="cloth"
)
shorts_category = CocoKeypointCategory(
    id=1,
    name="shorts",
    keypoints=SHORTS_KEYPOINTS,
    skeleton=[[i, i + 1] for i in range(0, len(SHORTS_KEYPOINTS) - 2)],
    supercategory="cloth",
)
tshirt_category = CocoKeypointCategory(
    id=2,
    name="tshirt",
    keypoints=TSHIRT_KEYPOINTS,
    skeleton=[[i, i + 1] for i in range(0, len(TSHIRT_KEYPOINTS) - 2)],
    supercategory="cloth",
)
boxershort_category = CocoKeypointCategory(
    id=3,
    name="boxershorts",
    keypoints=BOXERSHORT_KEYPOINTS,
    skeleton=[[i, i + 1] for i in range(0, len(BOXERSHORT_KEYPOINTS) - 2)],
    supercategory="cloth",
)

categories = {
    "categories": [towel_category.dict(), tshirt_category.dict(), shorts_category.dict(), boxershort_category.dict()]
}

if __name__ == "__main__":
    from cloth_dataset import DATA_DIR

    file_path = DATA_DIR / "artf_clothes_coco_categories.json"
    with open(file_path, "w") as f:
        import json

        json.dump(categories, f, indent=4, sort_keys=False)
