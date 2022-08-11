#!/usr/bin/env python3
import math
import os
from configparser import ConfigParser
from itertools import product
from typing import Dict, List, NamedTuple, Union
from pathlib import Path
import cv2
import numpy as np


class Projection:
    proj_uv: np.ndarray = None
    proj_tp: np.ndarray = None
    proj_xyv: np.ndarray = None

    def __init__(self, scale: str):
        self.scale = Scale(scale)
        self.proj_img = np.zeros(self.scale.shape)

        self.mn2uv()
        self.uv2tp()
        self.tp2xyz()

    def mn2uv(self):
        if self.proj_uv is None:
            self.proj_uv = np.zeros(self.scale.shape)
        for n, m in self.scale:
            u = (m + 0.5) / self.scale.W
            v = (n + 0.5) / self.scale.H
            self.proj_uv[n, m] = Point_uv(u, v)
        return self.proj_uv

    def uv2tp(self):
        if self.proj_tp is None:
            self.proj_tp = np.zeros(self.scale.shape)
        for n, m in self.scale:
            u, v = self.proj_uv[n, m]
            theta = (0.5 - v) * np.pi
            phi = (u - 0.5) * 2 * np.pi
            self.proj_tp[n, m] = Point_tp(theta, phi)
        return self.proj_tp

    def tp2xyz(self):
        if self.proj_xyv is None:
            self.proj_xyv = np.zeros(self.scale.shape)
        for n, m in self.scale:
            theta, phi = self.proj_tp[n, m]
            x = np.cos(theta) * np.cos(phi)
            y = np.sin(theta)
            z = -np.cos(theta) * np.sin(phi)
            self.proj_xyv[n, m] = Point3d(x, y, z)
        return self.proj_xyv

    def get_tp_pix(self, theta, phi):
        v = - theta / np.pi + 0.5


Point_tp = NamedTuple('Point_tp', [('theta', float), ('phi', float)])
Point_uv = NamedTuple('Point_uv', [('u', float), ('v', float)])
# Scale = NamedTuple('Scale', [('x', int), ('y', int)])


class Scale:
    def __init__(self, scale: str):
        self.W, self.H = splitx(scale)

    def __str__(self):
        return f'{self.W}x{self.H}'

    def __iter__(self):
        return zip(map(range, self.shape))

    @property
    def shape(self):
        return self.H, self.W

    @shape.setter
    def shape(self, mn_tuple: tuple):
        self.H, self.W = mn_tuple


Point2d = NamedTuple('Point2d', [('x', float), ('y', float)])
Point3d = NamedTuple('Point3d', [('x', float), ('y', float), ('z', float)])
Point_bcs = NamedTuple('Point_bcs', [('yaw', float), ('pitch', float), ('roll', float)])  # body coordinate system
Point_hcs = NamedTuple('Point_hcs', [('r', float), ('azimuth', float), ('elevation', float)])  # horizontal coordinate system
class Fov(Scale): pass
class Pattern(Scale): pass


class Config:
    project: Union[str, Path]
    data_path: str
    projection: str
    scale: Union[str, Scale]
    pattern_list: List[str]
    pattern: Union[str, Pattern]
    viewport: dict
    unit: str
    columns_name: Dict[str, Dict]
    p_res_y: int
    proj_res: str
    fov_x: int
    fov_y: int
    fov: Union[str, Fov]
    yaw_col: str
    pitch_col: str
    roll_col: str

    def __init__(self, config_file: str):
        self.config_file = config_file
        config = ConfigParser()
        config.read(config_file)
        self.configure(config)

        self.project = Path(f'results/{self.project}')
        self.project.mkdir(parents=True, exist_ok=True)

        self.fov = Fov(self.fov)
        self.proj_scale = Scale(self.proj_res)

    def configure(self, config: ConfigParser) -> None:
        """
        This function convert itens under [main] sections of a config file
        created using ConfigParser in attributes of this class.
        :param config: the config filename
        :return: None
        """
        config = config['main']
        for item in config:
            import ast
            value = ast.literal_eval(config[item])
            setattr(self, item, value)


class Plane:
    normal: Point3d
    relation: str

    def __init__(self, normal=Point3d(0, 0, 0), relation='<'):
        self.normal = normal
        self.relation = relation  # With viewport


class View:
    center = Point_hcs(1, 0, 0)

    def __init__(self, fov='0x0'):
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
        self.fov = Fov(fov)
        fovx = np.deg2rad(self.fov.H)
        fovy = np.deg2rad(self.fov.W)

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

    def __init__(self, fov: str) -> None:
        """
        Viewport Class used to extract view pixels in projections.
        :param fov:
        """
        self.fov = Fov(fov)
        self.default_view = View(fov)
        self.new_view = View(fov)

    def set_position(self, position: Point_bcs):
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

        new_view = View(f'{view.fov}')
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
        p = Projection(str(res))
        projection = np.ones(res.shape, dtype=np.uint8) * 255
        for j, i in res:
            point_hcs = proj2sph(Point2d(i, j), res)
            point_cart = hcs2cart(point_hcs)
            if self.is_viewport(point_cart):
                projection.itemset((j, i), 0)  # by the docs, it is more efficient than projection[j, i] = 0
        return projection

    def is_viewport(self, point: Point3d) -> bool:
        """
        Check if the plane equation is true to viewport
        x1 * m + y1 * n + z1 * z < 0
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


class Tiling:
    def __init__(self, tiling: str, proj_res: str, fov: str):
        self.fov_x, self.fov_y = splitx(fov)
        self.viewport = Viewport(fov)
        self.m, self.n = splitx(tiling)
        self.proj_res = Scale(proj_res)
        self.proj_w, self.proj_h = splitx(proj_res)
        self.total_tiles = self.m * self.n
        self.tile_w = self.proj_res.W / self.m
        self.tile_h = self.proj_res.H / self.n

    def __str__(self):
        return f'({self.m}x{self.n}@{proj_res}.'

    def idx2mn(self, idx):
        tile_x = idx % self.m
        tile_y = idx // self.m
        return tile_x, tile_y

    def get_border(self, idx) -> list:
        """
        :param idx: indice do tile.
        :return: list with all border pixels coordinates
        """
        tile_m, tile_n = self.idx2mn(idx)
        width = self.tile_w
        high = self.tile_h

        x_i = width * tile_m  # first row
        x_f = width * (1 + tile_m) - 1  # last row
        y_i = high * tile_n  # first line
        y_f = high * (1 + tile_n) - 1  # last line
        '''
        [(x_i, y_i), (x_f, y_f)]
        plt.imshow(tiling.arr),plt.show
        '''

        border = list(zip(range(x_i, x_f), [y_i] * width))  # upper border
        border.extend(list(zip(range(x_i, x_f), [y_f] * width)))  # botton border
        border.extend(list(zip([x_i] * width, range(y_i, y_f))))  # left border
        border.extend(list(zip([x_f] * width, range(y_i, y_f))))  # right border

        return border

    def get_vptiles(self, position: Point_bcs):
        """
        1. seta o viewport na posição position
        2. para cada 'tile'
        2.1. pegue a borda do 'tile'
        2.2. para cada 'ponto' da borda
        2.2.1. se 'ponto' pertence ao viewport
        2.2.1.1. marcar tile
        2.2.1.2. break
        3. retorna tiles marcados
        """
        self.viewport.set_position(position)
        tiles = []
        self.arr = np.ones((self.proj_h, self.proj_w))
        for idx in range(self.total_tiles):
            border = self.get_border(idx)
            for (x, y) in border:
                point = self._unproject(Point2d(x, y))

                if self.viewport.is_viewport(point):
                    tiles.append(idx)
                    self.arr[y, x] = 0
                    # break
        return tiles

    def _unproject(self, point: Point2d):
        """
            Convert a 2D point of ERP projection coordinates to Horizontal Coordinate
            System (Only ERP Projection)
            :param point: A point in ERP projection
            :return: A 3D Point on the sphere
            """
        proj_scale = Scale(f'{self.proj_res}')
        point_hcs = proj2sph(point, proj_scale)
        point = hcs2cart(point_hcs)
        return point


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
    azimuth = - 180 + (point.x / res.W) * 360
    elevation = 90 - (point.y / res.H) * 180

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


def get_viewport(viewport: Viewport, projejction: Projection):
    for n, m in projejction.scale:
        point_tp = projejction.proj_tp[n, m]
        if viewport.is_viewport(point_tp):
            pass


if __name__ == '__main__':
    viewport = Viewport('120x90')
    viewport.set_position(Point_bcs(yaw=0, pitch=0, roll=0))
    viewport.project('400x200')
    viewport.show()
    viewport.save('viewport.png')
    print('Viewport = Fov(120x90), (yaw=0, pitch=0, roll=0)')

