"""
 *******************************************************************************
 *                       Created by Iceberg1991 in 2018
 *
 *
 * @file    Calibration.py
 * @brief   This file tries to get R matrix and T matrix of camera2 and camera3
            (two colour cameras) in GPS/IMU coordinate system from KITTI dataset
            ( [synced+rectified] version after calibration,
            http://www.cvlibs.net/datasets/kitti/raw_data.php).
 *******************************************************************************
"""

import math
import numpy as np
from scipy import linalg
import os


class Calibration(object):

    def __init__(self, directory):
        self.dir = directory
        # r2, t2 indicate transform from imu to camera2,
        # k2 means internal parameter of camera2
        self.r2, self.t2, self.k2 = None, None, None
        # r3, t3 indicate transform from imu to camera3,
        # k3 means internal parameter of camera3
        self.r3, self.t3, self.k3 = None, None, None
        self.rad_to_deg = 180 / math.pi
        self.deg_to_rad = 1 / self.rad_to_deg

    def lidar_to_camera(self):
        f = open(os.path.join(self.dir, 'calib_velo_to_cam.txt'))
        lines = f.readlines()
        f.close()
        r, fixed_t = None, None
        for line in lines:
            if line.startswith('R:'):
                r = self.get_r_matrix(line)
            elif line.startswith('T:'):
                t = self.get_t_matrix(line)
                fixed_t = np.array([-t[2], -t[0], t[1]])
        return np.linalg.inv(r), fixed_t

    def imu_to_lidar(self):
        f = open(os.path.join(self.dir, 'calib_imu_to_velo.txt'))
        lines = f.readlines()
        f.close()
        r, t = None, None
        for line in lines:
            if line.startswith('R:'):
                r = self.get_r_matrix(line)
            elif line.startswith('T:'):
                t = self.get_t_matrix(line)
        return np.linalg.inv(r), -t

    def camera_to_camera(self):
        f = open(os.path.join(self.dir, 'calib_cam_to_cam.txt'))
        lines = f.readlines()
        f.close()
        cam2, cam3 = None, None
        for i in range(len(lines)):
            if lines[i].startswith('K_02'):
                internal = self.get_internal(lines[i])
                r = self.get_r_matrix(lines[i + 2])
                t = self.get_t_matrix(lines[i + 3])
                cam2 = [internal, r, t]
                i += 4
            if lines[i].startswith('K_03'):
                internal = self.get_internal(lines[i])
                r = self.get_r_matrix(lines[i + 2])
                t = self.get_t_matrix(lines[i + 3])
                i += 4
                cam3 = [internal, r, t]
        return cam2, cam3

    def camera_transform(self):
        f = open(os.path.join(self.dir, 'calib_cam_to_cam.txt'))
        lines = f.readlines()
        f.close()
        cameras = []
        for i in range(len(lines)):
            if lines[i].startswith('K_0'):
                internal = self.get_internal(lines[i])
                r = self.get_r_matrix(lines[i + 5])
                t = -self.get_t_matrix(lines[i + 3])
                cameras.append([internal, np.linalg.inv(r), t])
                i += 6
        return cameras

    @staticmethod
    def get_r_matrix(r_line):
        data = r_line.split(' ')[1:]
        return np.asarray(data).astype(float).reshape((3, 3))

    @staticmethod
    def get_t_matrix(t_line):
        data = t_line.split(' ')[1:]
        return np.asarray(data).astype(float)

    @staticmethod
    def get_p_matrix(p_line):
        data = p_line.split(' ')[1:]
        return np.asarray(data).astype(float).reshape((3, 4))

    @staticmethod
    def get_internal(k_line):
        data = k_line.split(' ')[1:]
        return np.asarray(data).astype(float).reshape((3, 3))

    def calibration(self):
        r_lidar_camera, t_lidar_camera = self.lidar_to_camera()
        r_imu_lidar, t_imu_lidar = self.imu_to_lidar()
        cam2, cam3 = self.camera_to_camera()
        r_cam0_cam2, t_cam0_cam2 = cam2[1], cam2[2]
        r_cam0_cam3, t_cam0_cam3 = cam3[1], cam3[2]
        self.k2 = cam2[0]
        self.r2 = r_imu_lidar @ r_lidar_camera @ r_cam0_cam2
        self.t2 = t_imu_lidar + r_imu_lidar @ (t_lidar_camera + r_lidar_camera @ t_cam0_cam2)
        self.k3 = cam3[0]
        self.r3 = r_imu_lidar @ r_lidar_camera @ r_cam0_cam3
        self.t3 = t_imu_lidar + r_imu_lidar @ (t_lidar_camera + r_lidar_camera @ t_cam0_cam3)

    def calibration_rectified(self):
        r_lidar_camera, t_lidar_camera = self.lidar_to_camera()
        r_imu_lidar, t_imu_lidar = self.imu_to_lidar()
        cam0, _, cam2, cam3 = self.camera_transform()
        r_cam0_cam2, t_cam0_cam2 = cam2[1], cam2[2]
        r_cam0_cam3, t_cam0_cam3 = cam3[1], cam3[2]
        r_00 = cam0[1]
        self.k2 = cam2[0]
        self.r2 = r_imu_lidar @ r_lidar_camera @ r_00 @ r_cam0_cam2
        self.t2 = t_imu_lidar + r_imu_lidar @ (t_lidar_camera + r_lidar_camera @ r_00 @ t_cam0_cam2)

        self.k3 = cam3[0]
        self.r3 = r_imu_lidar @ r_lidar_camera @ r_00 @ r_cam0_cam3
        self.t3 = t_imu_lidar + r_imu_lidar @ (t_lidar_camera + r_lidar_camera @ r_00 @ t_cam0_cam3)

    @staticmethod
    def pose_to_r_matrix(pose):
        phi3 = pose[0]  # theta_z, heading, yaw
        phi2 = pose[1]  # theta_y, pitch
        phi1 = pose[2]  # theta_x, roll
        phi = np.zeros((3, 3))
        phi[0, 1] = -phi3
        phi[0, 2] = phi2
        phi[1, 0] = phi3
        phi[1, 2] = -phi1
        phi[2, 0] = -phi2
        phi[2, 1] = phi1
        return linalg.expm(phi)

    def get_camera2(self, imu_position, imu_pose):
        if self.r2 is None or self.t2 is None:
            print("Error: Camera2 hasn't been calibrated yet.")
            return None
        if type(imu_position) is not np.ndarray or len(imu_position) != 3 \
                or type(imu_pose) is not np.ndarray or len(imu_pose) != 3:
            print("Error: Invalid imu position or pose (should be a 3-element numpy array)")
            return None
        r_world_imu = self.pose_to_r_matrix(imu_pose)
        x, y, z = imu_position + r_world_imu @ self.t2
        r_world_camera = r_world_imu @ self.r2
        phi = linalg.logm(r_world_camera)
        h = phi[1, 0]
        p = phi[0, 2]
        r = phi[2, 1]
        return np.asarray([x, y, z, h, p, r]).astype(float)

    def get_camera3(self, imu_position, imu_pose):
        if self.r3 is None or self.t3 is None:
            print("Error: Camera3 hasn't been calibrated yet.")
            return None
        if type(imu_position) is not np.ndarray or len(imu_position) != 3 \
                or type(imu_pose) is not np.ndarray or len(imu_pose) != 3:
            print("Error: Invalid imu position or pose (should be a 3-element numpy array)")
            return None
        r_world_imu = self.pose_to_r_matrix(imu_pose)
        x, y, z = imu_position + r_world_imu @ self.t3
        r_world_camera = r_world_imu @ self.r3
        phi = linalg.logm(r_world_camera)
        h = phi[1, 0]
        p = phi[0, 2]
        r = phi[2, 1]
        return np.asarray([x, y, z, h, p, r]).astype(float)


if __name__ == "__main__":
    calib = Calibration("/Users/test/Documents/DATA/Camera/KITTI/2011_09_28/2011_09_28_calib")
    imu_position = np.array([387, 642, 0])
    imu_pose = np.array([1, 0.8, 0.62])
    calib.calibration()
    cam2 = calib.get_camera2(imu_position, imu_pose)
    cam3 = calib.get_camera3(imu_position, imu_pose)

    tmp = 3
