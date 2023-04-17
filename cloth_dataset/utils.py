import numpy as np 


def is_2dpoint_in_convex_2dpolygon(point, ordered_polygon_points):
    """Check if a 2D point is inside a convex 2D polygon by using half-play tests. 
    The polygon is defined by a list of points. The points are assumed to be ordered clockwise."""
    for start_id in range(len(ordered_polygon_points)):
        end_id = (start_id + 1) % len(ordered_polygon_points)
        start = ordered_polygon_points[start_id]
        end = ordered_polygon_points[end_id]
        start_to_point = point - start
        start_to_end = end - start
        # if the point is on the left side of any edge, it is outside the polygon.
        if np.dot(start_to_point, start_to_end) < 0:
            return False
    return True


if __name__ == "__main__":
    point = np.array([1.01, 0.5])
    polygon_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    print(is_2dpoint_in_convex_2dpolygon(point, polygon_points))