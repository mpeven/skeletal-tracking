import numpy as np
import shapely.geometry as geom

class Bbox:
    def __init__(self, name, part_id, depth_image, xyz, box_size, projection):
        if not isinstance(xyz, np.ndarray):
            raise ValueError("xyz must be an np.ndarray")
        self.name = name
        self.id = part_id
        self.center = np.array([xyz[0], xyz[1]])
        self.z = xyz[2]
        self.im_d = depth_image
        self.im_d[self.im_d == 0] = 255
        x_delta_scaled = box_size[0]/2
        self.weight = 1.0
        y_delta_scaled = box_size[1]/2
        self.xmin, self.xmax = xyz[0]-x_delta_scaled, xyz[0]+x_delta_scaled
        self.ymin, self.ymax = xyz[1]-y_delta_scaled, xyz[1]+y_delta_scaled
        self.poly = geom.box(self.xmin, self.ymin, self.xmax, self.ymax)
        self.color_min = (int(projection['fx']*self.xmin/xyz[2] + projection['cx']),
                          int(projection['fy']*self.ymin/xyz[2] + projection['cy']))
        self.color_max = (int(projection['fx']*self.xmax/xyz[2] + projection['cx']),
                          int(projection['fy']*self.ymax/xyz[2] + projection['cy']))
        self.depth_min = (int(projection['fx_d']*self.xmin/xyz[2] + projection['cx_d']),
                          int(projection['fy_d']*self.ymin/xyz[2] + projection['cy_d']))
        self.depth_max = (int(projection['fx_d']*self.xmax/xyz[2] + projection['cx_d']),
                          int(projection['fy_d']*self.ymax/xyz[2] + projection['cy_d']))

    def __str__(self):
        return "{{{: 1.4f},{: 1.4f}}}, {{{: 1.4f},{: 1.4f}}}".format(self.xmin, self.ymin, self.xmax, self.ymax)

    def __repr__(self):
        return "(bbox: {{{: 1.4f},{: 1.4f}}}, {{{: 1.4f},{: 1.4f}}})".format(self.xmin, self.ymin, self.xmax, self.ymax)


    def size(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def get_bb_depth_matrix(self):
        """ Get the portion of the depth image inside the bounding box """
        min_x, max_x = sorted((self.depth_min[0], self.depth_max[0]))
        min_y, max_y = sorted((self.depth_min[1], self.depth_max[1]))
        bounded_im = self.im_d[min_y: max_y+1, min_x: max_x+1]
        return bounded_im

    def overlap(self, bb2):
        dx = min(self.xmax, bb2.xmax) - max(self.xmin, bb2.xmin)
        dy = min(self.ymax, bb2.ymax) - max(self.ymin, bb2.ymin)
        if (dx>=0) and (dy>=0):
            return dx*dy
        return 0

    def p_over(self, bb2):
        return self.overlap(bb2)/(min(self.size(), bb2.size()))

    def p_depth(self, bb2):
        bounded_im1 = self.get_bb_depth_matrix()
        bounded_im2 = bb2.get_bb_depth_matrix()
        print(bounded_im1.empty or bounded_im2.empty)
        mean1 = np.mean(bounded_im1)
        mean2 = np.mean(bounded_im2)
        stdev1 = np.std(bounded_im1)
        stdev2 = np.std(bounded_im2)

        half_negative_square_of_mean_difference = -1/2 * (mean1 - mean2) ** 2
        term1_power = half_negative_square_of_mean_difference / (stdev1 ** 2)
        term2_power = half_negative_square_of_mean_difference / (stdev2 ** 2)
        out = (np.exp(term1_power) + np.exp(term2_power))/2
        return out

    def prob(self, bb2, alpha):
        return alpha * self.p_over(bb2) + (1-alpha) * self.p_depth(bb2)
