"""centralization.py: centralize the keypoints

centralize keypoints towards the neck. All folders, all files, all keypoints
Used keypoints:
Pose
    0 to 5      -> upper body
    15 to 18    -> head
Face & Hands    -> Use all
Set all other keypoints (legs) to none

Take Keypoint: Pose[0] as "zero" and subtract all ponts from that point, e.g.
    X_n (X coordinate of neck)
    Y_n (Y coordinate of neck)
    X_s (X coordinate of shoulder)
    Y_s (Y coordinate of shoulder)
    [X_n - X_s, Y_n - Y_s] -> write output where X_s and Y_s were

Finally set X_n, Y_n to 0

"""

class Normalize:

    def __init__(self, path_to_json_dir):
        self.path_to_json = path_to_json_dir