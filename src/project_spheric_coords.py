import numpy as np
import cv2

HEIGHT = 64
WIDTH = 512

file_name = '../dataset/lidar/pcd/002399.bin.pcd'


def points_to_depth_map(velo_points, height=HEIGHT, width=WIDTH, dtheta=np.radians(0.4), dphi=np.radians(90./512.0)):
    x, y, z, i = velo_points[:, 0], velo_points[:, 1], velo_points[:, 2], velo_points[:, 3]

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

    depth_map = np.zeros((HEIGHT, WIDTH, 5))
    depth_map[theta_, phi_, 0] = x
    depth_map[theta_, phi_, 1] = y
    depth_map[theta_, phi_, 2] = z
    depth_map[theta_, phi_, 3] = i
    depth_map[theta_, phi_, 4] = d

    return depth_map


def hv_in_range(x, y, z, fov, fov_type='h'):
    d = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if fov_type == 'h':
        return np.logical_and(np.arctan2(y, x) > (-fov[1] * np.pi / 180),
                              np.arctan2(y, x) < (-fov[0] * np.pi / 180))
    elif fov_type == 'v':
        return np.logical_and(np.arctan2(z, d) < (fov[1] * np.pi / 180),
                              np.arctan2(z, d) > (fov[0] * np.pi / 180))
    else:
        raise NameError("fov type must be set between 'h' and 'v' ")


def main():
    with open(file_name, 'r') as f:
        points = f.readlines()[10:]
    np_arr = np.zeros((112537, 4))

    points = [point.rstrip().split(' ') for point in points]
    for i in range(112537):
        pts = points[i]
        np_arr[i, 0] = float(pts[0])
        np_arr[i, 1] = float(pts[1])
        np_arr[i, 2] = float(pts[2])
        np_arr[i, 3] = float(pts[3])

    # filter points for front view
    cond = hv_in_range(x=np_arr[:, 0],
                       y=np_arr[:, 1],
                       z=np_arr[:, 2],
                       fov=[-45, 45])

    np_arr_ranged = np_arr[cond]

    lidar = points_to_depth_map(np_arr_ranged)
    img = cv2.normalize(lidar[:, :, 0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('x.png', img)

    img = cv2.normalize(lidar[:, :, 1], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('y.png', img)

    img = cv2.normalize(lidar[:, :, 2], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('z.png', img)

    img = cv2.normalize(lidar[:, :, 3], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('intensity.png', img)

    img = cv2.normalize(lidar[:, :, 4], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imwrite('distance.png', img)

    # cv2.imshow("image", img)
    # cv2.waitKey();

    breakpoint()

    lidar_f = lidar.astype(np.float32)

    lidar_mask = np.reshape(
        (lidar[:, :, 4] > 0),
        [HEIGHT, WIDTH, 1]
    )

    lidar_f = (lidar_f - np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])) / np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])


if __name__ == "__main__":
    main()
