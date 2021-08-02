#!/usr/bin/env python3
import math
import os
from configparser import ConfigParser
from itertools import product
from typing import Dict, List, NamedTuple

import cv2
import numpy as np

Scale = NamedTuple('Scale', [('x', int), ('y', int)])
class Fov(Scale): pass


Point2d = NamedTuple('Point2d', [('x', float), ('y', float)])
Point3d = NamedTuple('Point3d', [('x', float), ('y', float), ('z', float)])
# body coordinate system
Point_bcs = NamedTuple('Point_bcs', [('yaw', float), ('pitch', float), ('roll', float)])
# horizontal coordinate system
Point_hcs = NamedTuple('Point_hcs', [('r', float), ('azimuth', float), ('elevation', float)])


class Config:
    project: str
    data_path: str
    projection: str
    scale: str
    pattern_list: List[str]
    pattern: Scale
    viewport: dict
    unit: str
    columns_name: Dict[str, Dict]
    p_res_x: int
    p_res_y: int
    fov_x: int
    fov_y: int
    yaw_col: str
    pitch_col: str
    roll_col: str

    def __init__(self, config_file: str):
        self.config_file = config_file
        config = ConfigParser()
        config.read(config_file)
        self.configure(config)

        self.project = f'results/{self.project}'
        os.makedirs(self.project, exist_ok=True)

        self.fov = Scale(self.fov_x, self.fov_y)
        self.proj_res = Scale(self.p_res_x, self.p_res_y)

    def configure(self, config: ConfigParser) -> None:
        """
        This function convert itens under [main] sections of a config file
        created using ConfigParser in attributes of this class.
        :param config: the config filename
        :return: None
        """
        config = config['main']
        for item in config:
            try:
                value = float(config[item])
            except ValueError:
                value = config[item]
            setattr(self, item, value)


class Plane:
    normal: Point3d
    relation: str

    def __init__(self, normal=Point3d(0, 0, 0), relation='<'):
        self.normal = normal
        self.relation = relation  # With viewport


class View:
    center = Point_hcs(1, 0, 0)

    def __init__(self, fov=Scale(0, 0)):
        """
        The viewport is the a region of sphere created by the intersection of
        four planes that pass by center of sphere and the Field of view angles.
        Each plane split the sphere in two hemispheres and consider the viewport
        overlap.
        Each plane was make using a normal vectors (x1i+y1j+z1k) and the
        equation of the plane (x1x+y2y+z1z=0)
        If we rotate the vectors, so the viewport is roted too.
        :param fov: Field-of-View in degree
        :return: None
        """
        self.fov = fov
        fovx = np.deg2rad(fov.x)
        fovy = np.deg2rad(fov.y)

        self.p1 = Plane(Point3d(-np.sin(fovy / 2), 0, np.cos(fovy / 2)))
        self.p2 = Plane(Point3d(-np.sin(fovy / 2), 0, -np.cos(fovy / 2)))
        self.p3 = Plane(Point3d(-np.sin(fovx / 2), np.cos(fovx / 2), 0))
        self.p4 = Plane(Point3d(-np.sin(fovx / 2), -np.cos(fovx / 2), 0))

    def __iter__(self):
        return iter([self.p1, self.p2, self.p3, self.p4])

    def get_planes(self):
        return [self.p1, self.p2, self.p3, self.p4]


class Viewport:
    position: Point_bcs
    projection: np.ndarray

    def __init__(self, fov: Scale) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = fov
        self.default_view = View(fov)
        self.new_view = View(fov)

    def set_position(self, position: Point_bcs) -> View:
        """
        Set a new position to viewport using aerospace's body coordinate system
        and make the projection. Return numpy.ndarray.
        :param position:
        :return:
        """
        self.position = position
        self._rotate()
        return self

    def _rotate(self):
        """
        Rotate the normal planes of viewport using matrix of rotation and Tait–Bryan
        angles in Z-Y-X order. Refer to Wikipedia.
        :return: A new View object with the new position.
        """
        view: View = self.default_view
        new_position: Point_bcs = self.position

        new_view = View(view.fov)
        mat = rot_matrix(new_position)
        new_view.center = Point_hcs(1, new_position[0], new_position[1])

        # For each plane in view
        for default_plane, new_plane in zip(view, new_view):
            normal = default_plane.normal
            roted_normal = mat @ normal
            new_plane.normal = Point3d(roted_normal[0], roted_normal[1], roted_normal[2])

        self.new_view = new_view

    def project(self, scale: str):
        res = Scale(*splitx(scale))
        self.projection = self._project_viewport(res)
        return self

    def _project_viewport(self, res: Scale) -> np.ndarray:
        """
        Project the sphere using ERP. Where is Viewport the
        :param res: The resolution of the Viewport
        :return: a numpy.ndarray with one deep color
        """
        projection = np.ones((res.y, res.x), dtype=np.uint8) * 255
        for j, i in product(range(res.y), range(res.x)):
            point_hcs = proj2sph(Point2d(i, j), res)
            point_cart = hcs2cart(point_hcs)
            if self.is_viewport(point_cart):
                projection.itemset((j, i), 0)  # by the docs, it is more efficient than projection[j, i] = 0
        return projection

    def is_viewport(self, point: Point3d) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * x + y1 * y + z1 * z < 0
        If True, the "point" is on the viewport
        :param point: A 3D Point in the space.
        :return: A boolean
        """
        view = self.new_view
        is_in = True
        for plane in view.get_planes():
            result = (plane.normal.x * point.x
                      + plane.normal.y * point.y
                      + plane.normal.z * point.z)
            teste = (result < 0)

            # is_in só retorna true se todas as expressões forem verdadeiras
            is_in = is_in and teste
        return is_in

    def show(self) -> None:
        cv2.imshow('imagem', self.projection)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, file_path):
        cv2.imwrite(file_path, self.projection)
        print('save ok')


def splitx(string: str) -> tuple:
    return tuple(map(int, string.split('x')))


def rot_matrix(new_position: Point_bcs):
    """
    Create rotation matrix using Tait–Bryan angles in Z-Y-X order.
    See Wikipedia.
    :param new_position: A new position using the Body Coordinate System.
    :return:
    """
    yaw = np.deg2rad(new_position.yaw)
    pitch = np.deg2rad(-new_position.pitch)
    roll = np.deg2rad(-new_position.roll)

    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    sr = np.sin(roll)

    mat = np.array(
        [[cy * cp, cy * sp * sr - cr * sy, sy * sr + cy * cr * sp],
         [cp * sy, cy * cr + sy * sp * sr, cr * sy * sp - cy * sr],
         [-sp, cp * sr, cp * cr]])

    return mat


def proj2sph(point: Point2d, res: Scale) -> Point3d:
    """
    Convert a 2D point of ERP projection coordinates to Horizontal Coordinate
    System in degree
    :param point: A point in ERP projection
    :param res: The resolution of the projection
    :return: A 3D Point on the sphere
    """
    # Only ERP Projection
    radius = 1
    azimuth = (point.x / res.x) * 360 - 180
    elevation = 90 - (point.y / res.y) * 180
    return Point_hcs(radius, azimuth, elevation)


def hcs2cart(position: Point_hcs):
    """
    Horizontal Coordinate system to Cartesian coordinates. Equivalent to sph2cart in Matlab.
    https://www.mathworks.com/help/matlab/ref/sph2cart.html
    :param position: The coordinates in Horizontal Coordinate System
    :return: A Point3d in cartesian coordinates
    """
    az = position.azimuth / 180 * math.pi
    el = position.elevation / 180 * math.pi
    r = position.r

    cos_az = math.cos(az)
    cos_el = math.cos(el)
    sin_az = math.sin(az)
    sin_el = math.sin(el)

    x = r * cos_el * cos_az
    y = r * cos_el * sin_az
    z = r * sin_el

    return Point3d(x, y, z)


if __name__ == '__main__':
    viewport = Viewport(Fov(x=120, y=90))
    viewport.set_position(Point_bcs(yaw=0, pitch=0, roll=0))
    viewport.project('400x200')
    viewport.show()
    print('Viewport = Fov(120x90), (yaw=0, pitch=0, roll=0)')
