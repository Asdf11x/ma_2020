# 2d_vis.py: Create a visualization of 2d keypoints
# read in a json file and visualize pose, face and hands
# Paths - should be the folder where Open Pose JSON output was stored

import numpy as np
import os
import cv2
import json
from cv2 import LINE_AA
import matplotlib.colors as mcolors
import pandas as pd


class JSONVis:

    def __init__(self, width, height, FPS, path_to_json_dir):
        self.width = width
        self.height = height
        self.FPS = FPS
        self.path_to_json = path_to_json_dir
        self.json_files = [pos_json for pos_json in os.listdir(self.path_to_json) if pos_json.endswith('.json')]
        print('Found: %d json keypoint frame files in folder %s' % (len(self.json_files), self.path_to_json))

    def get_points(self, key):
        # for file in json_files:
        temp_df = json.load(open(self.path_to_json + self.file))
        temp_x_pose = temp_df['people'][0][key][0::3]
        temp_y_pose = temp_df['people'][0][key][1::3]
        return [temp_x_pose, temp_y_pose]

    def get_confidence(self, key):
        temp_df = json.load(open(self.path_to_json + self.file))
        temp_conf = temp_df['people'][0][key][2::3]
        return np.mean(temp_conf), temp_conf

    def cl(self, str):
        # getting color range 0 to 255
        # Its not rgb (red, green, blue) here, but b,g,r (blue, green, red)
        switched_colors = np.array(mcolors.to_rgb(str)).dot(255)
        switched_colors = np.array([switched_colors[2], switched_colors[1], switched_colors[0]])
        return switched_colors

    def draw_pose(self, frame, key):
        points = self.get_points(key)

        """
        Build pose: right arm is the person's right arm. not the viwer's right arm:
        [0-1]:          neck
        [1-4]:          right arm
        [1-5,6,7]:      left arm
        [1-8]:          back
        [8-11]:         right leg
        [8-12,13,14]:   left leg
        [24, 11, 22, 23]: right foot - not implemented
        [21, 14, 19, 20]: left foot - not implemented
        [17, 15, 0, 16, 18]: head
        position from: https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        """

        xs = [int(i) for i in points[0]]
        ys = [int(i) for i in points[1]]

        # neck
        cv2.line(frame, (xs[0], ys[0]), (xs[1], ys[1]), self.cl("gray"), 2, LINE_AA)
        cv2.circle(frame, (xs[0], ys[0]), 4, self.cl('white'), 3)
        cv2.circle(frame, (xs[1], ys[1]), 4, self.cl('white'), 3)
        # print("From %d, %d to %d, %d" % (xs[0], ys[0], xs[1], ys[1]))

        # back
        cv2.line(frame, (xs[1], ys[1]), (xs[8], ys[8]), self.cl('red'), 3)

        # right arm
        joints_x = xs[1:5]
        joints_y = ys[1:5]
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx+1], joints_y[idx+1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]),
                         self.cl("orange"), 3, LINE_AA)

        # left arm
        joints_x = xs[5:8]
        joints_y = ys[5:8]
        cv2.line(frame, (xs[1], ys[1]), (xs[5], ys[5]), self.cl('lime'), 3, LINE_AA)
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx + 1], joints_y[idx + 1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]), self.cl("lime"),
                         3, LINE_AA)


        # Currently removed visualization of legs
        # right leg
        """joints_x = xs[8:12]
        joints_y = ys[8:12]
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx + 1], joints_y[idx + 1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]),
                         self.cl("green"), 3, LINE_AA)"""

        # left leg
        """joints_x = xs[12:15]
        joints_y = ys[12:15]
        cv2.line(frame, (xs[8], ys[8]), (xs[12], ys[12]), self.cl('cyan'), 3, LINE_AA)
        for idx in range(len(joints_x)):
            if idx < len(joints_x) - 1 and joints_x[idx] != 0 and joints_x[idx + 1] != 0:
                # print("From %d, %d to %d, %d" % (joints_x[idx], joints_y[idx], joints_x[idx + 1], joints_y[idx + 1]))
                cv2.line(frame, (joints_x[idx], joints_y[idx]), (joints_x[idx + 1], joints_y[idx + 1]), self.cl("cyan"),
                         3, LINE_AA)"""
        # head
        cv2.line(frame, (xs[17], ys[17]), (xs[15], ys[15]), self.cl("pink"), 2, LINE_AA)
        cv2.line(frame, (xs[15], ys[15]), (xs[0], ys[0]), self.cl("magenta"), 2, LINE_AA)
        cv2.line(frame, (xs[0], ys[0]), (xs[16], ys[16]), self.cl("purple"), 2, LINE_AA)
        cv2.line(frame, (xs[16], ys[16]), (xs[18], ys[18]), self.cl("orchid"), 3, LINE_AA)

        # cv2.imshow('ImageWindow', frame)
        # cv2.waitKey()
        return frame

    def draw_face(self, frame, key, thickness):
        points = self.get_points(key)

        xs = [int(i) for i in points[0]]
        ys = [int(i) for i in points[1]]
        poly = np.array([xs, ys]).T.tolist()

        # shape
        cv2.polylines(frame, np.int32([poly[0:16]]), 0, self.cl("white"), thickness)
        # right eye brow
        cv2.polylines(frame, np.int32([poly[17:21]]), 0, self.cl("white"), thickness)
        # left eye brow
        cv2.polylines(frame, np.int32([poly[22:26]]), 0, self.cl("white"), thickness)
        # nose
        cv2.polylines(frame, np.int32([poly[27:30]]), 0, self.cl("white"), thickness)
        cv2.polylines(frame, np.int32([poly[31:35]]), 0, self.cl("white"), thickness)
        # right eye
        cv2.polylines(frame, np.int32([poly[36:41]]), 1, self.cl("white"), thickness)
        # left eye
        cv2.polylines(frame, np.int32([poly[42:47]]), 1, self.cl("white"), thickness)
        # mouth
        cv2.polylines(frame, np.int32([poly[48:59]]), 1, self.cl("white"), thickness)
        cv2.polylines(frame, np.int32([poly[60:67]]), 1, self.cl("white"), thickness)
        return frame

    def draw_hand(self, frame, key, thickness, idx):
        points = self.get_points(key)
        confidence = self.get_confidence(key)[0]
        conf_array = self.get_confidence(key)[1]

        if confidence > 0.2:
            xs = [int(i) for i in points[0]]
            ys = [int(i) for i in points[1]]
            poly = np.array([xs, ys]).T.tolist()

            # thumb
            if np.mean(conf_array[0:5]) > 0.2:
                cv2.polylines(frame, np.int32([poly[0:5]]), 0, self.cl("salmon"), thickness)
            # index finger
            if np.mean(conf_array[5:9]) > 0.2:
                cv2.polylines(frame, np.int32([[poly[0], poly[5]]]), 0, self.cl("goldenrod"), thickness)
                cv2.polylines(frame, np.int32([poly[5:9]]), 0, self.cl("goldenrod"), thickness)
            # middle finger
            if np.mean(conf_array[9:13]) > 0.2:
                cv2.polylines(frame, np.int32([[poly[0], poly[9]]]), 0, self.cl("springgreen"), thickness)
                cv2.polylines(frame, np.int32([poly[9:13]]), 0, self.cl("springgreen"), thickness)
            # ring finger
            if np.mean(conf_array[13:17]) > 0.2:
                cv2.polylines(frame, np.int32([[poly[0], poly[13]]]), 0, self.cl("navy"), thickness)
                cv2.polylines(frame, np.int32([poly[13:17]]), 0, self.cl("navy"), thickness)
            # little finger
            if np.mean(conf_array[17:21]) > 0.2:
                cv2.polylines(frame, np.int32([[poly[0], poly[17]]]), 0, self.cl("darkviolet"), thickness)
                cv2.polylines(frame, np.int32([poly[17:21]]), 0, self.cl("darkviolet"), thickness)
        else:
            print("too low at %d and %s" % (idx, key[:10]))
            
        if confidence > 0.2:
            new_frame = frame
            return new_frame
        else:
            return frame

    def draw_main(self):
        fourcc = cv2.VideoWriter_fourcc(*'MP42')
        video = cv2.VideoWriter('./json_vis.avi', fourcc, float(self.FPS), (self.width, self.height))
        idx = 0

        for file in self.json_files[:500]:
            self.file = file

            if self.get_confidence('hand_left_keypoints_2d')[0] > 0.1:
                pass


            blank_image = np.zeros((height, width, 3), np.uint8)
            blank_image_one = np.ones((height, width, 3), np.uint8)
            frame = self.draw_pose(blank_image_one, 'pose_keypoints_2d')
            alpha = 0.5  # Transparency factor.
            # Following line overlays transparent over the image
            frame = cv2.addWeighted(frame, alpha, blank_image, 1 - alpha, 0)

            frame = self.draw_face(frame, 'face_keypoints_2d', 1)
            frame = self.draw_hand(frame, 'hand_left_keypoints_2d', 2, idx)
            frame = self.draw_hand(frame, 'hand_right_keypoints_2d', 2, idx)

            video.write(frame)

            if idx % 100 == 0:
                print("Frame: %d of %d" % (idx, len(self.json_files)))

            idx += 1
        video.release()


if __name__ == '__main__':
    width = 1280
    height = 720
    FPS = 15
    path_to_json_dir = r"C:\Users\Asdf\Downloads\How2Sign_samples\openpose_output\json\_2FBDaOPYig-3-rgb_front\\"
    vis = JSONVis(width, height, FPS, path_to_json_dir)
    vis.draw_main()
