CLOTH_CATEGORIES = ("towel", "shorts", "tshirt", "boxershorts")


TOWEL_KEYPOINTS = [
    "corner0",
    "corner1",
    "corner2",
    "corner3",
]

SHORTS_KEYPOINTS = [
    "waist_left",
    "waist_right",
    "pipe_right_outer",
    "pipe_right_inner",
    "crotch",
    "pipe_left_inner",
    "pipe_left_outer",
]

TSHIRT_KEYPOINTS = [
    "shoulder_left",
    "neck_left",
    "neck_right",
    "shoulder_right",
    "sleeve_right_top",
    "sleeve_right_bottom",
    "armpit_right",
    "waist_right",
    "waist_left",
    "armpit_left",
    "sleeve_left_bottom",
    "sleeve_left_top",
]

BOXERSHORT_KEYPOINTS = SHORTS_KEYPOINTS

CATEGORY_TO_KEYPOINTS = {
    "towel": TOWEL_KEYPOINTS,
    "shorts": SHORTS_KEYPOINTS,
    "tshirt": TSHIRT_KEYPOINTS,
    "boxershorts": BOXERSHORT_KEYPOINTS,
}
