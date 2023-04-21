"""module to generate deformation instructions for flattened cloth items to emulate the artifacts of real world unfolding pipeline systems."""
from typing import List

import numpy as np


def sample_deformation_instructions(corner_list: list[str]) -> List[str]:
    """Samples a list of deformation instructions to apply to a flattened cloth item.

    Args:
        corner_list: list of corner names in clockwise order"""

    if np.random.uniform() > 0.6:
        # remain flattened
        return []
    deformations_to_apply = [_sample_deformation(corner_list)]

    if np.random.uniform() > 0.6:
        deformations_to_apply.append(_sample_deformation(corner_list))

    return deformations_to_apply


def _sample_deformation(corner_list: list[str]) -> str:
    deformation_options = [_sample_corner_bend_deformation, _sample_side_bend_deformation, _sample_pinch_deformation]
    return np.random.choice(deformation_options)(corner_list)


def _sample_corner_bend_deformation(corner_list: list[str]) -> str:
    # sample a random corner
    corner = np.random.choice(corner_list)
    direction = np.random.choice(["left", "center", "right"])
    distance = np.random.choice(["short", "long"])
    up_or_down = np.random.choice(["up", "down"])
    return f"bend corner <{corner}> <{up_or_down}> with direction <{direction}> and radius <{distance}>"


def _sample_side_bend_deformation(corner_list: list[str]) -> str:
    # sample a random start corner
    start_corner_id = np.random.randint(0, len(corner_list))
    end_corner_id = (start_corner_id + 1) % len(corner_list)
    distance = np.random.choice(["short", "long"])
    up_or_down = np.random.choice(["up", "down"])

    return f"bend side <{corner_list[start_corner_id]} -> {corner_list[end_corner_id]}> <{up_or_down}> with radius <{distance}>"


def _sample_pinch_deformation(*args, **kwargs) -> str:
    horizontal_position = np.random.choice(["left", "center", "right"])
    vertical_position = np.random.choice(["top", "center", "bottom"])
    return f"pinch <{horizontal_position}> <{vertical_position}>"


if __name__ == "__main__":
    print(sample_deformation_instructions(["top_left", "top_right", "bottom_left", "bottom_right"]))
