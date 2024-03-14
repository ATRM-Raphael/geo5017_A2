import os
import numpy as np
from matplotlib import pyplot as plt


class Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z

    def __str__(self):
        return "x:{}, y:{}, z:{}".format(self.x, self.y, self.z)


# Parent class for Slice_x, Slice_y and Slice_z
class Slice:
    def __init__(self, points=None):
        if points is None:
            self.points = set()


class Slice_x(Slice):
    def __init__(self, x_max, points=None):
        super().__init__(points)
        self.x_max = x_max

    def is_inside(self, point):
        return point.x < self.x_max


class Slice_y(Slice):
    def __init__(self, y_max, points=None):
        super().__init__(points)
        self.y_max = y_max

    def is_inside(self, point):
        return point.y < self.y_max


class Slice_z(Slice):
    def __init__(self, z_max, points=None):
        super().__init__(points)
        self.z_max = z_max

    def is_inside(self, point):
        return point.z < self.z_max


# Parent class for Slices_x, Slices_y and Slices_z
class Slices:
    def __init__(self, X, Y, Z, points, slice_layer, feture_type):
        self.feature_type = feture_type
        self.slice_layer = slice_layer
        self.points = points

        self.X_MIN, self.Y_MIN, self.Z_MIN = min(X), min(Y), min(Z)
        self.X_MAX, self.Y_MAX, self.Z_MAX = max(X), max(Y), max(Z)

        self.X_STEP = (self.X_MAX - self.X_MIN) / slice_layer
        self.Y_STEP = (self.Y_MAX - self.Y_MIN) / slice_layer
        self.Z_STEP = (self.Z_MAX - self.Z_MIN) / slice_layer

        self.step_volume = self.X_STEP * self.Y_STEP * self.Z_STEP
        self.bbox_volume = (self.X_MAX - self.X_MIN) * (self.Y_MAX - self.Y_MIN) * (self.Z_MAX - self.Z_MIN)

    def fill_points(self, slices):
        for point in self.points:
            for i, slice in enumerate(slices):
                if slice.is_inside(point):
                    pass
                elif slice.is_inside(point) and i+1 == len(slices):
                    slices[i].points.add(point)
                else:
                    slices[i - 1].points.add(point)
                    break


class Slices_x(Slices):
    def __init__(self, X, Y, Z, points, slice_layer, feture_type):
        super().__init__(X, Y, Z, points, slice_layer, feture_type)

    def get_feature(self):
        slices_x = [Slice_x(self.Y_MIN + (i + 1) * self.Y_STEP) for i in range(self.slice_layer)]
        Slices_x.fill_points(self, slices_x)

        slices_x_den = [len(slice.points) / self.bbox_volume for slice in slices_x]
        return slices_x_den


class Slices_y(Slices):
    def __init__(self, X, Y, Z, points, slice_layer, feture_type):
        super().__init__(X, Y, Z, points, slice_layer, feture_type)

    def get_feature(self):
        slices_y = [Slice_y(self.Y_MIN + (i + 1) * self.Y_STEP) for i in range(self.slice_layer)]
        Slices_y.fill_points(self, slices_y)

        slices_y_den = [len(slice.points) / self.bbox_volume for slice in slices_y]
        return slices_y_den


class Slices_z(Slices):
    def __init__(self, X, Y, Z, points, slice_layer, feture_type):
        super().__init__(X, Y, Z, points, slice_layer, feture_type)

    def get_feature(self):
        slices_z = [Slice_z(self.Z_MIN + (i + 1) * self.Z_STEP) for i in range(self.slice_layer)]
        Slices_z.fill_points(self, slices_z)

        slices_z_den = [len(slice.points) / self.bbox_volume for slice in slices_z]
        return slices_z_den







if __name__ == '__main__':
    pass