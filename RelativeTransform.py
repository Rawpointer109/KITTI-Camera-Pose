"""
 *******************************************************************************
 *                       Created by Iceberg1991 in 2017
 *
 *
 * @file    RelativeTransform.py
 * @brief   This file implements relative transform between Geodetic Coordinates
            (latitude, longitude, elevation) and local relative coordinates
            (x, y, z), which indicates the relative euclidean distance along
            earth surface of a given point to a predefined reference point
 *******************************************************************************
"""
import numpy as np


class RelativeTransform(object):

    def __init__(self, ref_lon, ref_lat, ref_ele=None):
        self.mRefLon = ref_lon
        self.mRefLat = ref_lat
        self.mRefEle = ref_ele
        self.mA = 6378137.0  # semi-major axis of earth
        self.mFlat = 0.0033528131778969  # flattening of earth, 1 / 298.257
        self.mEsquare = self.mFlat * (2 - self.mFlat)  # square of first eccentricity ratio
        self.mPi = 3.141592653589793  # PI
        self.mDegToRad = 0.0174532925199433  # PI/180
        self.mRadToDeg = 57.29577951308233  # 1/Ang_Hud
        self.mMr = self.meridian_radius(self.mRefLat)
        self.mPr = self.parallel_radius(self.mRefLat)

    def latlon_to_relative(self, lon, lat, ele=None):
        """
        Convert geodetic longitude/latitude to relative coordinates
        :param lon: numpy array (n * 1)
        :param lat: numpy array (n * 1)
        :param ele: none or numpy array (n * 1)
        :return:
        """
        dlon = lon - self.mRefLon
        dlon[dlon < -180] += 360
        dlon[dlon > 180] -= 360
        dx = dlon * self.mDegToRad * self.mPr
        dy = (lat - self.mRefLat) * self.mDegToRad * self.mMr
        if ele is not None:
            dz = ele - self.mRefEle
            if len(dx) == 1:
                return [dx[0], dy[0], dz[0]]
            else:
                return np.vstack((dx, dy, dz)).T
        if len(dx) == 1:
            return [dx[0], dy[0]]
        else:
            return np.vstack((dx, dy)).T

    def relative_to_latlon(self, x, y, z=None):
        dlon = x / self.mPr * self.mRadToDeg
        dlat = y / self.mMr * self.mRadToDeg
        lon = self.mRefLon + dlon
        lon[lon < -180] += 360
        lon[lon > 180] -= 360
        lat = self.mRefLat + dlat
        if z is not None and self.mRefEle is not None:
            ele = self.mRefEle + z
            if len(lon) == 1:
                return [lon[0], lat[0], ele[0]]
            else:
                return np.vstack((lon, lat, ele)).T
        if len(lon) == 1:
            return [lon[0], lat[0]]
        else:
            return np.vstack((lon, lat)).T

    def relative_to_relative(self, from_lon, from_lat, to_lon, to_lat, x, y):
        pr_target = self.parallel_radius(to_lat)
        pr_current = self.parallel_radius(from_lat)
        m_target = self.meridian_radius(to_lat)
        m_current = self.meridian_radius(from_lat)
        coe_x = pr_target / pr_current
        coe_y = m_target / m_current
        p = RelativeTransform(to_lon, to_lat)
        offset = p.latlon_to_relative(from_lon * np.ones(1), from_lat * np.ones(1))
        return [offset[0] + coe_x * x, offset[1] + coe_y * y]

    def meridian_radius(self, phi):
        sinphi = np.sin(phi * self.mDegToRad)
        return self.mA * (1 - self.mEsquare) / np.power(1 - self.mEsquare * sinphi * sinphi, 1.5)

    def parallel_radius(self, phi):
        sinphi = np.sin(phi * self.mDegToRad)
        cosphi = np.cos(phi * self.mDegToRad)
        return self.mA * cosphi / np.sqrt(1 - self.mEsquare * sinphi * sinphi)


if __name__ == '__main__':
    from_lon, from_lat = -83.133544921875, 42.36328125
    rt = RelativeTransform(from_lon, from_lat)
    to_lon, to_lat = -83.111572265625, 42.352294921875
    x, y = 2057.36, -894.932
    lon, lat = rt.relative_to_latlon(x * np.ones(1), y * np.ones(1))

    x1, y1 = rt.relative_to_relative(from_lon, from_lat, to_lon, to_lat, x, y)
    x2, y2 = rt.relative_to_relative(to_lon, to_lat, from_lon, from_lat, x1, y1)
    assert x == x2
    assert y == y2