import pandas as pd
import numpy as np
import cv2
from skeletrack.bbox import Bbox
from skeletrack.bbox_set import BboxSet
import matplotlib.pyplot as plt

nice_colors = [
    (255,0,255), # Magenta
    (255,255,0), # Yellow
    (0,255,0),   # Lime
    (0,220,220), # Cyan
    (255,0,0),   # Red
    (128,0,128), # Purple
    (255,165,0), # Orange
    (255,235,205), # Blanched Almond
    (240,248,255), # Alice blue
]
for x in range(500):
    nice_colors.append((255,0,0))


class Viewer:
    def __init__(self, dataset):
        self.dataset = dataset


    def display(self, video_index, frame_index, X):
        im = self.dataset.get_image(video_index, frame_index).copy()
        for m in range(X.shape[0]):
            for l in range(X.shape[1]):
                z_guess = self.dataset.get_ground_truth_df(video_index, frame_index).iloc[0][0][2]
                xy = self.dataset.loc_to_pixel((X[m,l,0],X[m,l,1],z_guess),video_index)
                cv2.circle(im, xy, 2, nice_colors[m], 2)
        plt.rcParams["figure.figsize"] = (18,18)
        plt.imshow(im)
        plt.show()


    def display_image(self, video_index, frame_index, depth=False):
        im = self.dataset.get_image(video_index, frame_index, depth).copy()
        joints = self.dataset.get_ground_truth_df(video_index, frame_index)
        for idx, row in joints.iterrows():
            xy = (int(row['color'][0]), int(row['color'][1]))
            cv2.circle(im, xy, 2, (255, 0, 0), 2)
            cv2.putText(im, str(idx), xy, cv2.FONT_HERSHEY_PLAIN, 1, (40, 220, 255))
        plt.rcParams["figure.figsize"] = (18,18)
        plt.imshow(im)
        plt.show()


    def display_loc_on_image(self, video_index, frame_index, x, y, z, depth=False):
        """
        Display a single location or array of locations on the image
        """
        im = self.dataset.get_image(video_index, frame_index, depth).copy()
        if isinstance(x, np.ndarray) or isinstance(x, list):
            for x,y,z in zip(x,y,z):
                xy = self.dataset.loc_to_pixel((x,y,z),video_index,depth)
                cv2.circle(im, xy, 2, (255, 0, 0), 2)
        else:
            xy = self.dataset.loc_to_pixel((x,y,z),video_index,depth)
            cv2.circle(im, xy, 2, (255, 0, 0), 2)
        plt.rcParams["figure.figsize"] = (18,18)
        plt.imshow(im)
        plt.show()



    def display_bb_on_image(self, video_index, frame_index, bounding_box, depth=False):
        im = self.dataset.get_image(video_index, frame_index, depth).copy()
        if isinstance(bounding_box, BboxSet):
            bb_sets = [bounding_box.bboxes]
        elif isinstance(bounding_box, Bbox):
            bb_sets = [[bounding_box]]
        elif isinstance(bounding_box, list) and isinstance(bounding_box[0], BboxSet):
            bb_sets = [bbset.bboxes for bbset in bounding_box]
        else:
            error_string = "Invalid type for bounding_box argument, accepts: BboxSet, "
            error_string += "Bbox, or list(BboxSet). Got type {}".format(type(bounding_box))
            raise TypeError(error_string)
        for idx, bb_list in enumerate(bb_sets):
            for bounding_box in bb_list:
                min_loc = bounding_box.depth_min if depth else bounding_box.color_min
                max_loc = bounding_box.depth_max if depth else bounding_box.color_max
                cv2.rectangle(im, min_loc, max_loc, color = nice_colors[idx], thickness = 1 if depth else 2)
        plt.rcParams["figure.figsize"] = (18,18)
        plt.imshow(im)
        plt.show()
