import numpy as np
'''
X = {M*J*2} (people x joints x dimensions)
'''

class LossFunctions:
    def __init__(self, num_joints, num_poses, alpha):
        self.num_joints = num_joints
        self.num_poses = num_poses
        self.alpha = alpha

    def z(self, X, bb_set, m, j):
        # Returns as a scalar
        full_cost = 0
        for bbox in bb_set.bboxes:
            if bbox.id == j:
                full_cost += bbox.weight * np.linalg.norm(bbox.center - X) ** 2.0
        return full_cost

    def dzdx(self, X, bb_set, m, j):
        full_gradient = np.zeros(X.shape[0])
        for bbox in bb_set.bboxes:
            if bbox.id == j:
                full_gradient = 2.0 * bbox.weight * (X - bbox.center)
        return full_gradient

    def S(self, X, bb_set, m, j):
        ''' Returns a scalar '''
        zm = np.array([z(X, bb_set, m, j) for m in range(num_people)])
        z = np.sum(zm, axis=0, keepdims=True)
        return np.sum(z * np.exp(alpha * z)) / np.sum(np.exp(alpha * z))

    def dSdz(self, X, bb_set, num_people):
        z = np.array([self.z(X, bb_set, m) for m in range(num_people)])
        denom = np.sum(np.exp(alpha * z))
        S = np.sum(z * np.exp(alpha * z)) / denom
        to_return = np.exp(alpha * z) * (1 + alpha * (z - S)) / denom
        return to_return
