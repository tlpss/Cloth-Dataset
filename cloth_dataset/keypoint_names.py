CLOTH_CATEGORIES = ("towels", "shorts", "tshirts", "boxershorts")

TOWEL_KEYPOINTS = [
    "corner_0",
    "corner_1",
    "corner_2",
    "corner_3",
]

SHORT_KEYPOINTS = [
    "left_waist",
    "right_waist",
    "right_pipe_outer",
    "right_pipe_inner",
    "crotch",
    "left_pipe_inner",
    "left_pipe_outer",
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

BOXERSHORT_KEYPOINTS = SHORT_KEYPOINTS

CATEGORY_TO_KEYPOINTS = {
    "towels": TOWEL_KEYPOINTS,
    "shorts": SHORT_KEYPOINTS,
    "tshirts": TSHIRT_KEYPOINTS,
    "boxershorts": BOXERSHORT_KEYPOINTS,
}
