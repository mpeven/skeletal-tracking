import glob, os
import numpy as np
from numpy.random import uniform as unif
import pandas as pd
from collections import namedtuple
import cv2
from skeletrack.bbox import Bbox
from skeletrack.bbox_set import BboxSet
from skeletrack.utils import math_utils
import pickle
import itertools
from bidict import bidict
from copy import deepcopy

# Set Pandas display characteristics.
import shutil
pd.set_option('display.width', shutil.get_terminal_size().columns)
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 18)
pd.set_option('display.max_colwidth', 60)

Body = namedtuple('Body', 'body_id clipped_edges hand_left_confidence hand_left_state hand_right_confidence hand_right_state is_restricted leanx leany tracking_state')
Joint = namedtuple('Joint', 'loc depth color orient track_state')
Point2D = namedtuple('Point2D', 'x y')
Point3D = namedtuple('Point3D', 'x y z')
Quaternion = namedtuple('Quaternion', 'w x y z')
JointIndexMap = pd.Series([
    'spine_base', 'spine_middle', 'neck', 'head', 'shoulder_left', 'elbow_left',
    'wrist_left', 'hand_left', 'shoulder_right', 'elbow_right', 'wrist_right',
    'hand_right', 'hip_left', 'knee_left', 'ankle_left', 'foot_left',
    'hip_right', 'knee_right', 'ankle_right', 'foot_right', 'spine',
    'left_hand_tip', 'thumb_left', 'right_hand_tip', 'thumb_right',
])


video_folder_name = "videos"
color_folder_name = "color"
depth_folder_name = "depth"
skeleton_folder_name = "skeletons"

Video = namedtuple('Video', 'color depth')

def get_paths(*args):
    return sorted(glob.glob(os.path.join(*args)))

class Dataset:
    def __init__(self):
        self.x_max = 1
        self.y_max = 1
        self.z_max = 1
        self.videos = pd.DataFrame()
        self.ground_truth_df = pd.DataFrame()
        self.detection_df = pd.DataFrame()
        self.video_df = pd.DataFrame()
        self.joint_connections = pd.DataFrame()
        self.images = {}
        self.depth_images = {}
        self.projection_values = [] # For projecting from xyz to pixel
        self.projection_matrices = None # For poses
        self.joints_of_interest_to_og_id = { # Joints we care about and their index
            1 : 'spine_middle',
            3 : 'head',
            5 : 'elbow_right',
            9 : 'elbow_left',
        }
        self.joints_of_interest = list(self.joints_of_interest_to_og_id.values())
        self.num_joints = len(self.joints_of_interest)
        self.num_poses = 10
        self.num_dimensions = 2
        self.joint_adjacency_list = list(itertools.combinations(self.joints_of_interest, 2))
        self.joint_adjacency_list_full = list(itertools.permutations(self.joints_of_interest, 2))
        self.joint_adjacency_list_id_full = list(itertools.permutations(range(self.num_joints), 2))
        self.joints_of_interest_to_id = bidict({joint_name: idx for idx, joint_name in enumerate(sorted(self.joints_of_interest))})


    def __str__(self):
        rep = "\nDataset\n-------\n"
        rep += str(self.ground_truth_df)
        return rep



    # @profile
    def load_images_ntu(self, dataset_path):
        self.video_df, self.videos = Dataset.load_video_frame_paths(dataset_path)
        if os.path.isfile('ground_truth_df.pkl'):
            self.ground_truth_df = pd.read_pickle('ground_truth_df.pkl')
        else:
            self.ground_truth_df = Dataset.load_skeleton_data(dataset_path, self.joints_of_interest)
            self.ground_truth_df.to_pickle('ground_truth_df.pkl')
        loc_array = np.array(list(self.ground_truth_df['loc']))
        self.x_min = min(loc_array[:, 0])
        self.x_max = max(loc_array[:, 0])
        self.y_min = min(loc_array[:, 1])
        self.y_max = max(loc_array[:, 1])
        self.z_min = min(loc_array[:, 2])
        self.z_max = max(loc_array[:, 2])
        self.projection_values = [self.get_projection_values(v) for v in range(len(self.videos))]
        # pickle exists
        if os.path.isfile('images.npz') and os.path.isfile('depth_images.npz'):
            print("pickle exists, loading from file")
            self.images = np.load('images.npz')
            self.depth_images = np.load('depth_images.npz')
        else:
            for frame_idx, row in self.video_df.iterrows():
                idx = str((row['video_index'],row['frame_index']))
                self.images[idx] = cv2.imread(row['color_frame_path'])
                self.depth_images[idx] = cv2.imread(row['depth_frame_path'], 0)
            np.savez('images', **self.images)
            np.savez('depth_images', **self.depth_images)

    def create_detections_ntu(self):
        det = pd.DataFrame()
        det['detection'] = Dataset.create_detections(
            self.ground_truth_df['loc'],
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max))
        det['joint_name'] = self.ground_truth_df['joint_name']
        det['person_index'] = self.ground_truth_df['person_index']
        det['frame_index'] = self.ground_truth_df['frame_index']
        det['video_index'] = self.ground_truth_df['video_index']
        def make_list(row):
            x = []
            for det in row:
                if len(det)>0:
                    for det_pt in det:
                        x.append(det_pt)
            return x
        dets = det.groupby(['frame_index','video_index','joint_name'])['detection'].apply(make_list)
        self.detection_df = dets.reset_index()


    @staticmethod
    def load_video_frame_paths(dataset_path):
        color_folders = get_paths(dataset_path, color_folder_name, r'*')
        depth_folders = get_paths(dataset_path, depth_folder_name, r'*')

        overview_df = pd.DataFrame()
        detail_df = pd.DataFrame()

        video_names = []
        video_frames = []
        for idx, (color_folder, depth_folder) in enumerate(zip(color_folders, depth_folders)):
            color_frames = glob.glob(color_folder + "/*")
            depth_frames = glob.glob(depth_folder + "/*")
            video_names.append(str(os.path.basename(color_folder)))
            assert(len(color_frames) == len(depth_frames))
            video_frames.append(len(color_frames))
            video_detail_df = pd.DataFrame()
            video_detail_df = video_detail_df.assign(frame_index=range(len(color_frames)),
                                         video_index=idx,
                                         color_frame_path=color_frames,
                                         depth_frame_path=depth_frames)
            detail_df = detail_df.append(video_detail_df)
        overview_df = overview_df.assign(name=video_names, num_frames=video_frames)
        detail_df.reset_index(inplace=True, drop=True)
        return detail_df, overview_df

    @staticmethod
    def load_skeleton_data(dataset_path, joints_of_interest):
        skeleton_df = pd.DataFrame()
        skeleton_paths = get_paths(dataset_path, skeleton_folder_name, '*')
        for video_index, skeleton_path in enumerate(skeleton_paths):
            video_index_df = pd.DataFrame()

            with open(skeleton_path, 'r') as f:
                data = f.readlines()

            for frame_idx in range(int(data.pop(0))):
                for body_idx in range(int(data.pop(0))):
                    body = Body(*map(eval, data.pop(0).split()))
                    joints = []
                    for joint_idx in range(int(data.pop(0))):
                        line =  data.pop(0).split()
                        xyz = Point3D(*map(float, line[:3]))
                        depth = Point2D(*map(float, line[3:5]))
                        color = Point2D(*map(float, line[5:7]))
                        quat = Quaternion(*map(float, line[7:11]))
                        joint = Joint(np.array(xyz), depth, color, quat, int(line[-1]))
                        joints.append(joint)
                    image_joints = pd.DataFrame(joints)
                    image_joints = image_joints.assign(
                        person_index=body_idx,
                        frame_index=frame_idx,
                        joint_index=image_joints.index,
                        joint_name=JointIndexMap[image_joints.index]
                    )
                    image_joints = image_joints[image_joints['joint_name'].isin(joints_of_interest)]
                    video_index_df = video_index_df.append(image_joints)
            video_index_df = video_index_df.assign(video_index=video_index)
            skeleton_df = skeleton_df.append(video_index_df)
        return skeleton_df



    @staticmethod
    def create_detections(locations, x_range, y_range, z_range):
        np.random.seed(7) # TESTING ONLY
        def randomize(xyz, missingness, extraness, random_extraness, stddev):
            to_return = []
            xyz = np.array(xyz)
            to_return.append(np.random.normal(xyz, stddev))

            # Extraness
            if np.random.binomial(1, 1-extraness) == 0:
                for x in range(np.random.poisson(1)):
                    to_return.append(np.random.normal(xyz, stddev))

            # Random Extraness
            if np.random.binomial(1, 1-random_extraness) == 0:
                for x in range(np.random.poisson(1)):
                    to_return.append(
                        np.array([unif(low=x_range[0], high=x_range[1]),
                                  unif(low=y_range[0], high=y_range[1]),
                                  unif(low=z_range[0], high=z_range[1])])
                    )

            # Missingness
            to_return = [] if (np.random.binomial(1, 1-missingness) == 0) else to_return
            return np.array(to_return)

        # Randomize ground truth to simulate a detector
        fake_detections = locations.apply(randomize, args=(0.001, 0.003, 0.0002, 0.0003))
        return fake_detections



    def create_bounding_boxes(self):
        x_delta = max((self.x_max-self.x_min)*0.16, (self.y_max-self.y_min)*0.16)
        y_delta = max((self.x_max-self.x_min)*0.16, (self.y_max-self.y_min)*0.16)
        scale_dict = {
            'spine_middle': (1.5, 3),
        }
        # @profile
        def make_bbox(row):
            depth_image = self.depth_images[str((row['video_index'], row['frame_index']))]
            projection = self.projection_values[row['video_index']]
            scale = (1, 1) if row['joint_name'] not in scale_dict else scale_dict[row['joint_name']]
            box_size = np.array([scale[0] * x_delta, scale[1] * y_delta])
            bboxes = []
            for xyz in row['detection']:
                if isinstance(xyz, np.ndarray):
                    part_name = row['joint_name']
                    if part_name in self.joints_of_interest_to_id:
                        part_id = self.joints_of_interest_to_id[part_name]
                    else:
                        part_id = 99
                    bboxes.append(Bbox(part_name, part_id, depth_image, xyz, box_size, projection))
            return bboxes
        self.detection_df['bounding_boxes'] = self.detection_df.apply(make_bbox, axis=1)




    def connect_joints(self):
        joint_df = pd.DataFrame(
            {name: list(self.ground_truth_df.ix[idx]['loc']) for idx, name in self.joints_of_interest_to_og_id.items()}
        )
        for ja, jb in self.joint_adjacency_list:
            self.joint_connections[(ja, jb)] = joint_df.apply(lambda x: tuple(x[jb] - x[ja]), axis=1)
        self.joint_connections = self.joint_connections.applymap(np.array)




    def create_projection_matrices(self, k, max_iterations=500):
        all_vecs = np.array([np.concatenate(v) for v in self.joint_connections.as_matrix()])
        pose_clusters = math_utils.k_means(all_vecs, k, max_iterations, visualize=False)
        projections = np.split(pose_clusters, len(self.joint_adjacency_list), axis=1)
        self.projection_matrices = {adj_joints: projections[idx] for idx, adj_joints in enumerate(self.joint_adjacency_list)}
        self.A_matrix = np.zeros((self.num_joints, self.num_joints, self.num_poses, 2))
        for i_name, j_name in self.joint_adjacency_list_full:
            i = self.joints_of_interest_to_id[i_name]
            j = self.joints_of_interest_to_id[j_name]
            if (i_name,j_name) in self.projection_matrices:
                self.A_matrix[i,j] = np.array([x[:2] for x in self.projection_matrices[(i_name,j_name)]])
                self.A_matrix[j,i] = np.array([-1. * x[:2] for x in self.projection_matrices[(i_name,j_name)]])



    def get_projected_locations(self, joint_name, joint_loc, to_joint_name):
        ''' Given joint_name -- 'head'   and    joint_loc -- head(x,y)
            this will return all projected to_joint_name -- 'spine' locations
            based on the clustered poses
        '''
        if (joint_name, to_joint_name) in self.projection_matrices:
            return np.array([joint_loc + c_loc[:2] for c_loc in self.projection_matrices[(joint_name, to_joint_name)]])
        elif (to_joint_name, joint_name) in self.projection_matrices:
            return np.array([joint_loc - c_loc[:2] for c_loc in self.projection_matrices[(to_joint_name, joint_name)]])
        else:
            raise KeyError("joint pair ({}, {}) not in adjacency list".format((joint_name, to_joint_name)))



    def get_projected_locations_by_id(self, joint_id, joint_loc, to_joint_id):
        joint_name = self.joints_of_interest_to_id.inv[joint_id]
        to_joint_name = self.joints_of_interest_to_id.inv[to_joint_id]
        return self.get_projected_locations(joint_name, joint_loc, to_joint_name)



    def get_image(self, video_index, frame_index, depth=False):
        idx = str((video_index, frame_index))
        return self.depth_images[idx] if depth else self.images[idx]



    def get_video_frame(self, video_index, frame_index):
        df_index = (self.video_df['video_index'] == video_index) & (self.video_df['frame_index'] == frame_index)
        return self.video_df[df_index].iloc[0].to_dict()



    def get_skeleton_df(self, video_index, frame_index):
        df_index = (self.detection_df['video_index'] == video_index) & (self.detection_df['frame_index'] == frame_index)
        return self.detection_df[df_index]


    def get_ground_truth_df(self, video_index, frame_index):
        df_index = (self.ground_truth_df['video_index'] == video_index) & (self.ground_truth_df['frame_index'] == frame_index)
        return self.ground_truth_df[df_index]



    def get_detections(self, video_index, frame_index):
        df = self.get_skeleton_df(video_index, frame_index)
        all_dets = {}
        for joint_name in self.joints_of_interest:
            all_dets[joint_name] = df[df['joint_name'] == joint_name]['bounding_boxes'].iloc[0]
        return all_dets


    def get_candidate_Xs(self, video_index, frame_index):
        detection_dict = self.get_detections(video_index, frame_index)
        num_dets_in_frame = sum([len(dets) for dets in detection_dict.values()])
        single_det_X = []
        for joint_name, detection_list in detection_dict.items():
            joint_id = self.joints_of_interest_to_id[joint_name]
            for det in detection_list:
                single_det_poses = np.zeros((self.num_joints, self.num_poses, self.num_dimensions))
                single_det_poses[joint_id] = det.center
                for to_joint_id in set(range(self.num_joints)) - set([joint_id]):
                    single_det_poses[to_joint_id] = self.get_projected_locations_by_id(joint_id, det.center, to_joint_id)
                single_det_poses = np.transpose(single_det_poses, [1,0,2])
                single_det_X.append(single_det_poses)
        return np.vstack(single_det_X)



    def group_bboxes(self, video_index, frame_index):
        alpha = 0.5
        dets = self.get_detections(video_index, frame_index)
        all_bbs = list(dets.values())
        amoebas = [BboxSet(bb) for bb in all_bbs[0]]
        for bbs in all_bbs[1:]:
            similarity = np.zeros((len(amoebas), len(bbs)))
            for i, am in enumerate(amoebas):
                for j, bb2 in enumerate(bbs):
                    similarity[i, j] = am.prob(bb2, alpha)

            new_amoebas = []
            for k in range(min(len(amoebas), len(bbs))):
                ameoba_max_idx, bb_max_idx = np.unravel_index(similarity.argmax(), similarity.shape)
                amoebas[ameoba_max_idx].add(bbs[bb_max_idx])
                new_amoebas.append(amoebas[ameoba_max_idx])
                similarity[ameoba_max_idx,:] = -1
                similarity[:,bb_max_idx] = -1
            amoebas = new_amoebas
        return amoebas



    def get_projection_values(self, video_index):
        ''' Values needed to project from an (x,y,z) to the depth or color image '''
        row0 = self.get_ground_truth_df(video_index, 0).iloc[0].to_dict()
        row1 = self.get_ground_truth_df(video_index, 0).iloc[1].to_dict()
        x0, y0, z0 = row0['loc']
        u0, v0 = row0['color']
        du0, dv0 = row0['depth']
        x1, y1, z1 = row1['loc']
        u1, v1 = row1['color']
        du1, dv1 = row1['depth']

        focal_fn_x = (u0 - u1)/(x0/z0 - x1/z1)
        focal_fn_y = (v0 - v1)/(y0/z0 - y1/z1)
        const_x = (u0 - focal_fn_x*x0/z0)
        const_y = (v0 - focal_fn_y*y0/z0)
        focal_fn_dx = (du0 - du1)/(x0/z0 - x1/z1)
        focal_fn_dy = (dv0 - dv1)/(y0/z0 - y1/z1)
        const_dx = (du0 - focal_fn_dx*x0/z0)
        const_dy = (dv0 - focal_fn_dy*y0/z0)

        return {
            'fx': focal_fn_x, 'fy': focal_fn_y,
            'cx': const_x,    'cy': const_y,
            'fx_d': focal_fn_dx, 'fy_d': focal_fn_dy,
            'cx_d': const_dx,    'cy_d': const_dy,
        }


    def loc_to_pixel(self, xyz, video_index, depth=False):
        p = self.projection_values[video_index]
        if depth:
            x = int(p['fx_d']*xyz[0]/xyz[2] + p['cx_d'])
            y = int(p['fy_d']*xyz[1]/xyz[2] + p['cy_d'])
        else:
            x = int(p['fx']*xyz[0]/xyz[2] + p['cx'])
            y = int(p['fy']*xyz[1]/xyz[2] + p['cy'])
        return (x,y)


    def preprocess_frame(self, X_tm1, X_tm2):
        ''' Remove people leaving the frame '''
        # Find dataset where this is necessary, none available in NTU dataset
        return X_tm1, []


    #####################
    # Data for unit tests
    #
    def make_array(self, X_dict, num_persons):
        ''' Turn dictionary joints into a numpy array '''
        X = np.zeros((num_persons, self.num_joints, 2))
        for person in range(num_persons):
            for name, loc in X_dict.items():
                X[person, self.joints_of_interest_to_id[name]] = loc
        return X


    def get_spa_data(self):
        head_loc = np.array([1.0,0.0])
        Xt1 = {
            'head': head_loc,
            'elbow_left': self.get_projected_locations(joint_name='head', joint_loc = head_loc, to_joint_name='elbow_left')[0],
            'elbow_right': self.get_projected_locations(joint_name='head', joint_loc = head_loc, to_joint_name='elbow_right')[0],
            'spine_middle': self.get_projected_locations(joint_name='head', joint_loc = head_loc, to_joint_name='spine_middle')[0]
        }
        Xt2 = deepcopy(Xt1)
        Xt3 = deepcopy(Xt1)
        Xt2['head'][0] += 0.3
        Xt2['head'][1] -= 0.3
        Xt3['spine_middle'][0] -= 0.05
        Xt3['elbow_left'][0] -= 0.2
        Xt3['elbow_left'][1] -= 0.2
        Xt3['elbow_right'][0] -= 0.5
        Xt3['head'][0] -= 0.5
        Xt4 = {
            'head': head_loc,
            'elbow_left': self.get_projected_locations(joint_name='head', joint_loc = head_loc, to_joint_name='elbow_left')[5],
            'elbow_right': self.get_projected_locations(joint_name='head', joint_loc = head_loc, to_joint_name='elbow_right')[5],
            'spine_middle': self.get_projected_locations(joint_name='head', joint_loc = head_loc, to_joint_name='spine_middle')[5]
        }
        Xt1 = self.make_array(Xt1, 1)
        Xt2 = self.make_array(Xt2, 1)
        Xt3 = self.make_array(Xt3, 1)
        Xt4 = self.make_array(Xt4, 1)
        return Xt1, Xt2, Xt3, Xt4
        # display("Costs:")
        # display(z_Espa(Xt1, 0, 0))
        # display(z_Espa(Xt2, 0, 0))
        # display(z_Espa(Xt3, 0, 0))
        # display(z_Espa(Xt4, 0, 0))
        # display(z_Espa(Xt4, 0, 5))
        # display("Gradients:")
        # display(dz_Espa_dx(Xt1, 0, 0))
        # display(dz_Espa_dx(Xt2, 0, 0))
        # display(dz_Espa_dx(Xt3, 0, 0))
        # display(dz_Espa_dx(Xt4, 0, 0))
        # display(dz_Espa_dx(Xt4, 0, 5))
        # display("Full cost function gradients:")
        # display(dEdx_Espa(Xt1, 1))
        # display(dEdx_Espa(Xt2, 1))
        # display(dEdx_Espa(Xt3, 1))
        # display(dEdx_Espa(Xt4, 1))


    def get_det_data(self):
        frame3_bboxes = self.group_bboxes(0, 3)
        Xt1 = {bb.name: bb.center for bb in frame3_bboxes[0].bboxes}
        Xt2 = {bb.name: bb.center for bb in frame3_bboxes[1].bboxes}
        Xt1 = deepcopy(Xt1)
        Xt2 = deepcopy(Xt2)
        Xt3 = deepcopy(Xt1)
        Xt4 = deepcopy(Xt2)
        Xt1['head'][0] -= 0.2
        Xt1['head'][1] += 0.1
        # Xt2['head'][0] += 0.3
        # Xt2['head'][1] -= 0.3
        Xt3['spine_middle'][0] += 0.0823
        Xt3['elbow_right'][0] += 0.02
        Xt3['elbow_left'][0] -= 0.06234
        Xt3['head'][0] -= 0.043
        Xt4['spine_middle'][0] -= 0.05
        Xt4['spine_middle'][1] += 0.0823
        Xt4['elbow_left'][0] -= 0.2
        Xt4['elbow_left'][1] += 0.1234
        Xt4['elbow_right'][0] -= 0.1
        Xt4['head'][0] -= 0.43
        Xt1 = self.make_array(Xt1, 1)
        Xt2 = self.make_array(Xt2, 1)
        Xt3 = self.make_array(Xt3, 1)
        Xt4 = self.make_array(Xt4, 1)
        return frame3_bboxes, Xt1, Xt2, Xt3, Xt4
        # all_bb_sets, Xt1, Xt2, Xt3 = data.get_det_data()
        # display("Costs:")
        # display(z_Edet(Xt1, all_bb_sets[0], 0))
        # display(z_Edet(Xt2, all_bb_sets[0], 0))
        # display(z_Edet(Xt3, all_bb_sets[0], 0))
        # display("Gradients:")
        # display(dz_Edet_dx(Xt1, 0, all_bb_sets[0]))
        # display(dz_Edet_dx(Xt2, 0, all_bb_sets[0]))
        # display(dz_Edet_dx(Xt3, 0, all_bb_sets[0]))
        # display("dSdz:")
        # display(dSdz_Edet(Xt1, all_bb_sets[0], 1))
        # display(dSdz_Edet(Xt2, all_bb_sets[0], 1))
        # display(dSdz_Edet(Xt3, all_bb_sets[0], 1))
        # display("Full cost function gradients:")
        # display(dEdx_Edet(Xt1, 1, all_bb_sets))
        # display(dEdx_Edet(Xt2, 1, all_bb_sets))
        # display(dEdx_Edet(Xt3, 1, all_bb_sets))
        # display("dE")
        # display(dE(Xt1, 1, all_bb_sets))
        # display(dE(Xt2, 1, all_bb_sets))
        # display(dE(Xt3, 1, all_bb_sets))
        # display("E")
        # display(E(Xt1, 1, all_bb_sets))
        # display(E(Xt2, 1, all_bb_sets))
        # display(E(Xt3, 1, all_bb_sets))


    def get_tra_data(self):
        frame3_bboxes = self.group_bboxes(0, 3)
        Xt1 = {bb.name: bb.center for f in frame3_bboxes for bb in f.bboxes}
        Xt1 = deepcopy(Xt1)
        Xt1 = self.make_array(Xt1, 1)
        Xt10 = Xt1.copy()
        Xt11 = Xt1.copy()
        Xt12 = Xt1.copy()
        Xt11 -= 0.2
        Xt12 += 0.2
        return Xt10, Xt11, Xt12
