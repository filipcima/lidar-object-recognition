import os
import numpy as np
import cv2
from lib.calib import Calib
from multiprocessing.dummy import Pool as ThreadPool

# 1) bin2 np.array
# 2) project spheric coordinates
# 3) save to npy

VELO_PATH = f'../dataset/{os.environ.get("KITTI_DATASET_TYPE", "training")}/velodyne'
CALIB_PATH = f'../dataset/{os.environ.get("KITTI_DATASET_TYPE", "training")}/calib'
NPY_PATH = os.path.join('..', 'dataset', 'lidar_2d')
LABELS_PATH = '../dataset/labels'

KITTI_LABELS = {
    'Car': 1,
    'Van': 1,
    'Truck': 1,
    'Pedestrian': 2,
    'Person_sitting': 2,
    'Cyclist': 3,
    'Tram': 1,
    'Misc': 0,
    'DontCare': 0
}
KITTI_COLORS = {
    1: np.array([255, 0, 0]),
    2: np.array([0, 0, 255]),
    3: np.array([0, 255, 0]),
}


def process_kitti_frame(frame_number, save_example=False):
    print(f'Processing: {frame_number}.bin')
    np_points = np.fromfile(f'{VELO_PATH}/{frame_number}.bin', np.float32).reshape((-1, 4))

    # np.save(f'{VELO_PATH}/npy/{frame_number}.npy', np.reshape(np_points, (-1, 4)))

    boxes3d = []
    with open(f'{LABELS_PATH}/{frame_number}.txt') as f:
        # print(f'GT for: {frame_number}.bin')
        for line in f.readlines():
            # print(line)
            str_list = line.strip('\n').split(' ')
            if str_list[0] == 'Cyclist' or str_list[0] == 'Car' or str_list[0] == 'Pedestrian':
                box = Box3d()
                box.set_list(str_list)
                boxes3d.append(box)

    x, y, z, intensity = np_points[:, 0], np_points[:, 1], np_points[:, 2], np_points[:, 3]
    velo_points = np_points[:, :3]

    con = in_range(x, y, z, [-45, 45])

    # calib img points to labels coords from 3rd party lib
    calib = Calib(f'{CALIB_PATH}/{frame_number}.txt')
    calibrated_img_points = calib.velo2cams(velo_points[con])
    img_points = np_points[:, :3][con]
    gt_points = np.zeros_like(calibrated_img_points.T[:, 0])
    img_intensity = intensity[con]

    for i in range(len(boxes3d)):
        v_con = within_3d_box(calibrated_img_points, boxes3d[i])
        # label points
        gt_points[v_con] = boxes3d[i].get_label()

    train_depth_map = points_to_depth_map(img_points, img_intensity)
    gt_depth_map = points_to_depth_map(img_points, gt_points, C=1)

    if save_example:
        # kernel = np.ones((2, 2), np.uint8)
        # INTENSITY
        img = cv2.normalize(train_depth_map[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'intensity-{frame_number}.png', img)

        # X
        img = cv2.normalize(train_depth_map[:, :, 1], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'x-{frame_number}.png', img)

        # Y
        img = cv2.normalize(train_depth_map[:, :, 2], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'y-{frame_number}.png', img)

        # Z
        img = cv2.normalize(train_depth_map[:, :, 3], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'z-{frame_number}.png', img)

        # DISTANCE
        img = cv2.normalize(train_depth_map[:, :, 4], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'distance-{frame_number}.png', img)

        # LABELS
        # img = cv2.normalize(gt_depth_map[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_labels = gt_depth_map[:, :, 0]
        # RGB transform
        img = np.stack((img_labels,) * 3, axis=-1)
        print('labels shape:', img_labels.shape)
        print('img:', img.shape)
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        b[img_labels == 1] = 255  # cars are blue
        g[img_labels == 4] = 255  # pedestrians are green,
        r[img_labels == 6] = 255  # cyclist are red, code is written with glue

        # img[:, :, img == k] = v
        # img[:, :, img == k] = v

        #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(f'gt-{frame_number}.png', img)

    # SAVE TO FILE
    final_data = np.append(train_depth_map, gt_depth_map, -1)
    # print(final_data.shape)  # (64, 512, 6) -> [height][width][x, y, z, intensity, distance, label]
    # print(NPY_PATH)

    np.save(str(NPY_PATH) + f"/{frame_number}.npy", final_data)
    print(f'Saved: {VELO_PATH}/npy/{frame_number}.npy')


def in_range(x, y, z, fov):
    return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180),
                          np.arctan2(y, x) < (-fov[0] * np.pi / 180))


def within_3d_box(points, box3d):
    u, v, w, b_u, b_v, b_w = box3d.get_box()
    u_dot = np.dot(u, points)
    v_dot = np.dot(v, points)
    w_dot = np.dot(w, points)
    con_b_u = np.logical_and(u_dot < b_u[0], u_dot > b_u[1])
    con_b_v = np.logical_and(v_dot < b_v[0], v_dot > b_v[1])
    con_b_w = np.logical_and(w_dot < b_w[0], w_dot > b_w[1])
    return np.logical_and(np.logical_and(con_b_u, con_b_v), con_b_w)


class Box3d:
    def __init__(self, p1=0, p2=0, p4=0, p5=0):
        """
        Init 3D bounding box with 4 points
        """
        self.init = True
        self.u = p1 - p2
        self.v = p1 - p4
        self.w = p1 - p5
        self.b_u = [np.dot(self.u, p1), np.dot(self.u, p2)]
        self.b_v = [np.dot(self.v, p1), np.dot(self.v, p4)]
        self.b_w = [np.dot(self.w, p1), np.dot(self.w, p5)]

    def set_list(self, str_list):
        """
        ref: https://github.com/NVIDIA/DIGITS/issues/992
        """
        self.label = str_list[0]
        h, w, l = np.array(str_list[8:11]).astype(float)
        x, y, z = np.array(str_list[11:14]).astype(float)
        rot = np.array(str_list[14]).astype(float)

        px = np.array([0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l, 0.5 * l, 0.5 * l, -0.5 * l, -0.5 * l])
        py = np.array([0, 0, 0, 0, -h, -h, -h, -h])
        pz = np.array([0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w, 0.5 * w, -0.5 * w, -0.5 * w, 0.5 * w])
        rot_mat = np.array([
            [np.cos(rot), 0, np.sin(rot)],
            [0, 1, 0],
            [-np.sin(rot), 0, np.cos(rot)],
        ])

        p_stack = np.array([px, py, pz])

        rot_p = np.dot(rot_mat, p_stack)
        rot_p[0, :] = rot_p[0, :] + x
        rot_p[1, :] = rot_p[1, :] + y
        rot_p[2, :] = rot_p[2, :] + z

        p1 = rot_p[:, 0]
        p2 = rot_p[:, 1]
        p4 = rot_p[:, 3]
        p5 = rot_p[:, 4]
        self.__init__(p1, p2, p4, p5)

    def get_box(self):
        """
        Returns bounding box
        """
        if self.init is None:
            print('Error using get_box without init.')
            return None
        return self.u, self.v, self.w, self.b_u, self.b_v, self.b_w

    def get_label(self):
        if self.label is None:
            print('Error using get_label without init.')
            return KITTI_LABELS['DontCare']

        return KITTI_LABELS[self.label]


def points_to_depth_map(velo_points, intensity, height=64, width=512, C=5, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
    x, y, z = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2]

    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = np.sqrt(x ** 2 + y ** 2)

    d[d == 0] = 0.000001
    r[r == 0] = 0.000001

    phi = np.radians(45.) - np.arcsin(y / r)
    phi_ = (phi / dphi).astype(int)
    phi_[phi_ < 0] = 0
    phi_[phi_ >= width] = width - 1

    theta = np.radians(2.) - np.arcsin(z / d)
    theta_ = (theta / dtheta).astype(int)
    theta_[theta_ < 0] = 0
    theta_[theta_ >= height] = height - 1

    depth_map = np.zeros((height, width, C))

    if C == 1:
        depth_map[theta_, phi_, 0] = intensity
        return depth_map

    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = intensity
    depth_map[theta_, phi_, 4] = d

    return depth_map


def main():
    # TODO: refactor or use threading
    files = [file for file in os.listdir(VELO_PATH) if file.endswith('.bin')]
    # cnt = 0
    # examples = ('004755', '004786', '004790')
    # for example in examples:
    #     process_kitti_frame(example)

    for file in files:
        frame_number = file.split('.')[0]
        process_kitti_frame(frame_number)
        # if cnt == 5:
        #     break
        # cnt += 1

    ## THREADING
    # pool = ThreadPool(10)
    # results = pool.map(parse_kitti_data, paths)


if __name__ == "__main__":
    main()
