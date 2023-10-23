import numpy as np

from artf_clothes.keypoint_names import TSHIRT_KEYPOINTS


def order_keypoints(cloth_type, keypoints_2D, bbox):
    if cloth_type == "towel":
        keypoints_2D = order_towel_keypoints(keypoints_2D, bbox)
    elif cloth_type == "tshirt":
        keypoints_2D = order_tshirt_keypoints(keypoints_2D, bbox)
    else:
        raise ValueError(f"cloth_type {cloth_type} not supported for reorientation")
    return keypoints_2D


def order_towel_keypoints(keypoints_2D, bbox):
    """
    Towels have a lot of symmetry. The only guarantee from the labels is that they are ordered adjacent.
    So we have to decide a starting keypoint and an order to traverse them.

    starting point is the corner closest to the topleft bbox corner.
    Order is along the adjacent keypoint that is closest to the topright bbox corner.


    Args:
        keypoints_2D (_type_): _description_
        bbox (_type_): _description_

    Returns:
        _type_: _description_
    """
    x_min, y_min, width, height = bbox

    # keypoints are in cyclical order but we need to break symmetries by having a starting point in the image viewpoint
    bbox_top_left = (x_min, y_min)

    keypoints_2D_with_visibility = np.copy(keypoints_2D)
    keypoints_2D = keypoints_2D_with_visibility[:, :2]

    # find the keypoint that is closest to the top left corner of the bounding box
    distances = [np.linalg.norm(np.array(keypoint_2D) - np.array(bbox_top_left)) for keypoint_2D in keypoints_2D]
    starting_keypoint_index = np.argmin(distances)

    # now order the keypoints in a cyclical order starting from the starting keypoint with the second keypoints being the neighbour that is
    # closest to the topright corner of the bbox

    bbox_top_right = (x_min + width, y_min)
    distances = [
        np.linalg.norm(
            np.array(keypoints_2D[(starting_keypoint_index + i) % len(keypoints_2D)]) - np.array(bbox_top_right)
        )
        for i in [-1, +1]
    ]
    direction = -1 if np.argmin(distances) == 0 else +1
    second_keypoint_index = (starting_keypoint_index + direction) % len(keypoints_2D)

    # now order the keypoints in a cyclical order starting from the starting keypoint with the second keypoints being the neighbour that is
    direction = second_keypoint_index - starting_keypoint_index

    order = [starting_keypoint_index]
    for i in range(1, len(keypoints_2D)):
        order.append((starting_keypoint_index + i * direction) % len(keypoints_2D))

    new_keypoints_2D_with_visibility = np.array([keypoints_2D_with_visibility[i] for i in order])

    return new_keypoints_2D_with_visibility


def order_tshirt_keypoints(keypoints_2D: np.ndarray, bbox: tuple):
    """
    get the two shoulder keypoints
    determine which one has smallest U coordinate, this becomes the 'left' shoulder now.
    does not necessarily match human intuition, but it is consistent and easy to figure out for the NN.
    all other left-right separations can be done afterwards, main thing here is to have association between adjacent keypoints so that even for deformed pieces the model
    estimates which shoulder is adjacent to a given armpit.


    Args:
        keypoints_2D (np.ndarray): _description_
        bbox (tuple): _description_

    Returns:
        _type_: _description_
    """

    shoulder_left_idx = TSHIRT_KEYPOINTS.index("shoulder_left")
    shoulder_right_idx = TSHIRT_KEYPOINTS.index("shoulder_right")

    neck_left_idx = TSHIRT_KEYPOINTS.index("neck_left")
    neck_right_idx = TSHIRT_KEYPOINTS.index("neck_right")

    waist_left_idx = TSHIRT_KEYPOINTS.index("waist_left")
    waist_right_idx = TSHIRT_KEYPOINTS.index("waist_right")

    shoulder_left_2D = keypoints_2D[shoulder_left_idx]
    keypoints_2D[shoulder_right_idx]

    neck_left_2D = keypoints_2D[neck_left_idx]
    neck_right_2D = keypoints_2D[neck_right_idx]

    waist_left_2D = keypoints_2D[waist_left_idx]
    waist_right_2D = keypoints_2D[waist_right_idx]

    waist_center = (waist_left_2D + waist_right_2D) / 2
    neck_center = (neck_left_2D + neck_right_2D) / 2

    vertical_vector = neck_center - waist_center
    vertical_vector /= np.linalg.norm(vertical_vector)

    waist_to_left_shoulder_vector = shoulder_left_2D - waist_center

    z_coord_of_cross_product = (
        vertical_vector[0] * waist_to_left_shoulder_vector[1] - vertical_vector[1] * waist_to_left_shoulder_vector[0]
    )

    # origin  is topleft of image
    # so positive z coord points down and implies the 'left kp' was actually on the right side of the body
    if z_coord_of_cross_product > 0:
        should_tshirt_be_flipped = True
    else:
        should_tshirt_be_flipped = False

    if should_tshirt_be_flipped:
        for idx, keypoint in enumerate(TSHIRT_KEYPOINTS):
            if "left" in keypoint:
                right_idx = TSHIRT_KEYPOINTS.index(keypoint.replace("left", "right"))
                # swap the rows in the numpy array, cannot do this as with lists
                # https://stackoverflow.com/questions/21288044/row-exchange-in-numpy
                keypoints_2D[[idx, right_idx]] = keypoints_2D[[right_idx, idx]]

    return keypoints_2D
