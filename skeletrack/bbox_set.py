import numpy as np

class BboxSet:
    """A collection of bounding box objects"""
    def __init__(self, *args):
        self.bboxes = [args[0]]
        self.complex_shape = args[0].poly
        self.depths = {}
        xy = args[0].get_bb_depth_matrix()
        x_len, y_len = xy.shape
        for x in range(x_len):
            for y in range(y_len):
                self.depths[(x,y)] = xy[x,y]

        for bb in args[1:]:
            self.add(bb)

    def add(self, newbb):
        self.bboxes.append(newbb)
        # Update centers and weights
        num_joints = max([bbox.id+1 for bbox in self.bboxes])
        self.centers = np.zeros([num_joints, 2])
        self.weights = np.zeros([num_joints])
        for bbox in self.bboxes:
            self.centers[bbox.id] = bbox.center
            self.weights[bbox.id] = bbox.weight
        # Update shape
        self.complex_shape = self.complex_shape.union(newbb.poly)
        # Update depths
        new_depths = newbb.get_bb_depth_matrix()
        x_len, y_len = new_depths.shape
        for x in range(x_len):
            for y in range(y_len):
                self.depths[(x,y)] = new_depths[x,y]

    def __str__(self):
        out = 'Bounding Box Set:\n'
        out += '\t\n'.join(str(bb) for bb in self.bboxes)
        return out

    __repr__ = __str__

    def __iter__(self):
        return iter(self.bboxes)

    @property
    def poly(self):
        return self.complex_shape

    def size(self):
        return self.complex_shape.area

    def overlap(self, bb2):
        return self.poly.intersection(bb2.poly).area

    def p_over(self, bb2):
        return self.overlap(bb2)/(min(self.size(), bb2.size()))

    def get_bb_depth_matrix(self):
        return np.array(list(self.depths.values()))

    def p_depth(self, bb2):
        bounded_im1 = self.get_bb_depth_matrix()
        bounded_im2 = bb2.get_bb_depth_matrix()
        if len(bounded_im1) == 0 or len(bounded_im2) == 0:
            print("\n\nEMPTY BBOX\n\n")
            print(self)
            print(bb2)
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
