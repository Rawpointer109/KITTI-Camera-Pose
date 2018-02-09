"""
 *******************************************************************************
 *                       Created by Iceberg1991 in 2018
 *
 *
 * @file    ImagePrepare.py
 * @brief   This file implements camera position and pose generation from KITTI
            dataset ( [synced+rectified] version after calibration,
            http://www.cvlibs.net/datasets/kitti/raw_data.php). Camera position
            and pose are exported to a txt configure file and a kml file.
 *******************************************************************************
"""

import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import simplekml

from Calibration import Calibration
from RelativeTransform import RelativeTransform


class ImagePrepare(object):

    def __init__(self, dir_name):
        self.deg_to_rad = np.pi / 180
        self.rad_to_deg = 180 / np.pi
        self.dir = dir_name
        self.imu_position, self.imu_pose = None, None
        self.camera2_position, self.camera2_pose = None, None
        self.camera3_position, self.camera3_pose = None, None
        self.camera2_files, self.camera3_files = [], []
        self.parse_ins()
        self.ref_lon = self.imu_position[0, 0]
        self.ref_lat = self.imu_position[0, 1]
        self.geodetic_to_relative()
        self.rename_image()
        self.get_cameras()
        self.draw_cameras()
        # self.relative_to_geodetic()
        self.write_config_file()
        self.write_kml_file()

    def rename_image(self):
        img2_dir = os.path.join(self.dir, 'Drive/image_02/data')
        for file in os.listdir(img2_dir):
            if file.endswith('.png') and file.startswith('0'):
                new_file = 'cam02_' + file
                os.rename(os.path.join(img2_dir, file), os.path.join(img2_dir, new_file))
                self.camera2_files.append(new_file)
            elif file.startswith('cam02_') and file.endswith('.png'):
                tiff_name = file[: file.rfind('.')] + '.tif'
                if not os.path.exists(os.path.join(img2_dir, tiff_name)):
                    im = Image.open(os.path.join(img2_dir, file))
                    im.save(os.path.join(img2_dir, tiff_name))
                self.camera2_files.append(tiff_name)
        img3_dir = os.path.join(self.dir, 'Drive/image_03/data')
        for file in os.listdir(img3_dir):
            if file.endswith('.png') and file.startswith('0'):
                new_file = 'cam03_' + file
                os.rename(os.path.join(img3_dir, file), os.path.join(img3_dir, new_file))
                self.camera3_files.append(new_file)
            elif file.startswith('cam03_') and file.endswith('.png'):
                tiff_name = file[: file.rfind('.')] + '.tif'
                if not os.path.exists(os.path.join(img3_dir, tiff_name)):
                    im = Image.open(os.path.join(img3_dir, file))
                    im.save(os.path.join(img3_dir, tiff_name))
                self.camera3_files.append(tiff_name)

    def geodetic_to_relative(self):
        rt = RelativeTransform(self.ref_lon, self.ref_lat)
        self.imu_position[:, 0:2] = rt.latlon_to_relative(self.imu_position[:, 0],
                                                          self.imu_position[:, 1])[:, 0:2]

    def relative_to_geodetic(self):
        rt = RelativeTransform(self.ref_lon, self.ref_lat)
        relative = rt.relative_to_latlon(self.imu_position[:, 0], self.imu_position[:, 1])
        self.imu_position[:, 0:2] = relative[:, 0:2]
        if self.camera2_position is not None:
            relative = rt.relative_to_latlon(self.camera2_position[:, 0], self.camera2_position[:, 1])
            self.camera2_position[:, 0:2] = relative[:, 0:2]
        if self.camera3_position is not None:
            relative = rt.relative_to_latlon(self.camera3_position[:, 0], self.camera3_position[:, 1])
            self.camera3_position[:, 0:2] = relative[:, 0:2]

    def parse_ins(self):
        ins_dir = os.path.join(self.dir, 'Drive/oxts/data')
        position = []
        pose = []
        for frame in os.listdir(ins_dir):
            if not frame.endswith("txt"):
                continue
            f = open(os.path.join(ins_dir, frame))
            line = f.readline()
            f.close()
            data = line.split(' ')
            lat = float(data[0])
            lon = float(data[1])
            alt = float(data[2])
            roll = float(data[3])     # in rad
            pitch = float(data[4])    # in rad
            yaw = float(data[5])      # heading, in rad
            position.append([lon, lat, alt])
            pose.append([yaw, pitch, roll])
        self.imu_position = np.asarray(position)
        self.imu_pose = np.asarray(pose)

    def get_cameras(self):
        """
        Get camera position and pose according to calibration data
        :return:
        """
        calib = Calibration(os.path.join(self.dir, 'Calibration'))
        calib.calibration_rectified()
        assert len(self.imu_position) == len(self.imu_pose)
        self.camera2_position, self.camera2_pose = [], []
        self.camera3_position, self.camera3_pose = [], []
        for i in range(len(self.imu_position)):
            cam2 = calib.get_camera2(self.imu_position[i], self.imu_pose[i])
            self.camera2_position.append(cam2[0:3])
            self.camera2_pose.append(cam2[3:])
            cam3 = calib.get_camera3(self.imu_position[i], self.imu_pose[i])
            self.camera3_position.append(cam3[0:3])
            self.camera3_pose.append(cam3[3:])
        self.camera2_position = np.asarray(self.camera2_position)
        self.camera2_pose = np.asarray(self.camera2_pose)
        self.camera3_position = np.asarray(self.camera3_position)
        self.camera3_pose = np.asarray(self.camera3_pose)

    def write_config_file(self):
        assert len(self.camera2_files) == len(self.camera2_position)
        assert len(self.camera3_files) == len(self.camera3_position)
        out_data = []
        for i in range(len(self.camera2_files)):
            x, y, z = self.camera2_position[i].astype(str)
            h, p, r = self.camera2_pose[i].astype(str)
            content = self.camera2_files[i] + ',' + x + ',' + y + ',' + z + ',' \
                      + p + ',' + r + ',' + h + ',0.01' + ',0.05\n'
            out_data.append(content)
        for i in range(len(self.camera3_files)):
            x, y, z = self.camera3_position[i].astype(str)
            h, p, r = (self.camera3_pose[i] * self.rad_to_deg).astype(str)
            content = self.camera3_files[i] + ',' + x + ',' + y + ',' + z + ',' \
                      + p + ',' + r + ',' + h + ',0.01' + ',0.05\n'
            out_data.append(content)
        out_file = os.path.join(self.dir, 'config.txt')
        f = open(out_file, 'w')
        f.writelines(out_data)
        f.close()

    def write_kml_file(self):
        assert len(self.camera2_files) == len(self.camera2_position)
        assert len(self.camera3_files) == len(self.camera3_position)
        assert len(self.camera2_position) == len(self.camera3_position)
        kml = simplekml.Kml()
        for i in range(0, len(self.camera2_position), 5):
            kml.newpoint(name="cam2_" + str(i),
                         coords=[(self.camera2_position[i, 0],
                                  self.camera2_position[i, 1])])
            kml.newpoint(name="cam3_" + str(i),
                         coords=[(self.camera3_position[i, 0],
                                  self.camera3_position[i, 1])])
        out_file = os.path.join(self.dir, 'camera_location.kml')
        kml.save(out_file)

    def draw_cameras(self):
        # plt.axis('equal')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis('equal')

        # camera 2 position and pose
        ax.scatter(self.camera2_position[:, 0],
                   self.camera2_position[:, 1], c='r')
        for i in range(0, len(self.imu_position)):
            cam = self.camera2_position[i]
            r = Calibration.pose_to_r_matrix(self.camera2_pose[i])
            target = cam + r @ np.array([0, 0, 0.3])
            ax.plot([cam[0], target[0]], [cam[1], target[1]], c='r')

        # camera 3 position and pose
        ax.scatter(self.camera3_position[:, 0],
                   self.camera3_position[:, 1],
                   c='r')
        for i in range(0, len(self.imu_position)):
            cam = self.camera3_position[i]
            r = Calibration.pose_to_r_matrix(self.camera3_pose[i])
            target = cam + r @ np.array([0, 0, 0.3])
            ax.plot([cam[0], target[0]], [cam[1], target[1]], c='r')

        # imu position and pose
        ax.scatter(self.imu_position[:, 0],
                   self.imu_position[:, 1],
                   c='b')
        for i in range(0, len(self.imu_position)):
            imu = self.imu_position[i]
            r = Calibration.pose_to_r_matrix(self.imu_pose[i])
            target = imu + r @ np.array([0.3, 0, 0])
            ax.plot([imu[0], target[0]], [imu[1], target[1]], c='b')

        # for i in range(len(self.imu_position)):
        #     plt.text(self.camera2_position[i, 0], self.camera2_position[i, 1], str(i))
        #     plt.text(self.camera3_position[i, 0], self.camera3_position[i, 1], str(i))
        #     plt.text(self.imu_position[i, 0], self.imu_position[i, 1], str(i))
        plt.show()

    def draw_cameras_3d(self):
        # plt.axis('equal')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('equal')

        # camera 2 position and pose
        ax.scatter(self.camera2_position[:, 0],
                   self.camera2_position[:, 1],
                   self.camera2_position[:, 2], c='r')
        for i in range(0, len(self.imu_position)):
            cam = self.camera2_position[i]
            r = Calibration.pose_to_r_matrix(self.camera2_pose[i])
            target = cam + r @ np.array([0, 0, 0.3])
            ax.plot([cam[0], target[0]], [cam[1], target[1]], [cam[2], target[2]], c='r')

        # camera 3 position and pose
        ax.scatter(self.camera3_position[:, 0],
                   self.camera3_position[:, 1],
                   self.camera3_position[:, 2],
                   c='r')
        for i in range(0, len(self.imu_position)):
            cam = self.camera3_position[i]
            r = Calibration.pose_to_r_matrix(self.camera3_pose[i])
            target = cam + r @ np.array([0, 0, 0.3])
            ax.plot([cam[0], target[0]], [cam[1], target[1]], [cam[2], target[2]], c='r')

        # imu position and pose
        ax.scatter(self.imu_position[:, 0],
                   self.imu_position[:, 1],
                   self.imu_position[:, 2],
                   c='b')
        for i in range(0, len(self.imu_position)):
            imu = self.imu_position[i]
            r = Calibration.pose_to_r_matrix(self.imu_pose[i])
            target = imu + r @ np.array([0.3, 0, 0])
            ax.plot([imu[0], target[0]], [imu[1], target[1]], [imu[2], target[2]], c='b')

        # for i in range(len(self.imu_position)):
        #     plt.text(self.camera2_position[i, 0], self.camera2_position[i, 1], str(i))
        #     plt.text(self.camera3_position[i, 0], self.camera3_position[i, 1], str(i))
        #     plt.text(self.imu_position[i, 0], self.imu_position[i, 1], str(i))
        plt.show()


if __name__ == "__main__":
    ip = ImagePrepare('/Users/test/work/RRM/Data/Camera/2011_09_26_04')
    tmp = 3