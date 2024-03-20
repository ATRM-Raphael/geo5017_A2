import os
import numpy as np


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
    def __init__(self, X, Y, Z, points, slice_number, feature_type):
        self.feature_type = feature_type
        self.slice_number = slice_number
        self.points = points

        self.X_MIN, self.Y_MIN, self.Z_MIN = min(X), min(Y), min(Z)
        self.X_MAX, self.Y_MAX, self.Z_MAX = max(X), max(Y), max(Z)

        self.X_STEP = (self.X_MAX - self.X_MIN) / slice_number
        self.Y_STEP = (self.Y_MAX - self.Y_MIN) / slice_number
        self.Z_STEP = (self.Z_MAX - self.Z_MIN) / slice_number

        self.step_volume = self.X_STEP * self.Y_STEP * self.Z_STEP
        self.bbox_volume = (self.X_MAX - self.X_MIN) * (self.Y_MAX - self.Y_MIN) * (self.Z_MAX - self.Z_MIN)

    def fill_points(self, slices):
        for point in self.points:
            for i, slice in enumerate(slices):
                if slice.is_inside(point):
                    slice.points.add(point)
                else:
                    pass


class Slices_x(Slices):
    def __init__(self, X, Y, Z, points, slice_number, feature_type):
        super().__init__(X, Y, Z, points, slice_number, feature_type)

    def get_feature(self):
        slices_x = [Slice_x(self.X_MIN + (i + 1) * self.X_STEP) for i in range(self.slice_number)]
        Slices_x.fill_points(self, slices_x)

        slices_x_den = [len(slice.points) / self.bbox_volume for slice in slices_x]
        slices_x_num = [len(slice.points) for slice in slices_x]
        if self.feature_type == "density":
            return slices_x_den
        elif self.feature_type == "number":
            return slices_x_num
        elif self.feature_type == "both":
            return slices_x_den + slices_x_num


class Slices_y(Slices):
    def __init__(self, X, Y, Z, points, slice_number, feature_type):
        super().__init__(X, Y, Z, points, slice_number, feature_type)

    def get_feature(self):
        slices_y = [Slice_y(self.Y_MIN + (i + 1) * self.Y_STEP) for i in range(self.slice_number)]
        Slices_y.fill_points(self, slices_y)

        slices_y_den = [len(slice.points) / self.bbox_volume for slice in slices_y]
        slices_y_num = [len(slice.points) for slice in slices_y]
        if self.feature_type == "density":
            return slices_y_den
        elif self.feature_type == "number":
            return slices_y_num
        elif self.feature_type == "both":
            return slices_y_den + slices_y_num


class Slices_z(Slices):
    def __init__(self, X, Y, Z, points, slice_number, feature_type):
        super().__init__(X, Y, Z, points, slice_number, feature_type)

    def get_feature(self):
        slices_z = [Slice_z(self.Z_MIN + (i + 1) * self.Z_STEP) for i in range(self.slice_number)]
        Slices_z.fill_points(self, slices_z)

        slices_z_den = [len(slice.points) / self.bbox_volume for slice in slices_z]
        slices_z_num = [len(slice.points) for slice in slices_z]
        if self.feature_type == "density":
            return slices_z_den
        elif self.feature_type == "number":
            return slices_z_num
        elif self.feature_type == "both":
            return slices_z_den + slices_z_num


def get_feature(file_path, slice_number_x, slice_number_y, slice_number_z, feature_type):
    with open(file_path, mode="r") as file:
        lines = [line.strip().split() for line in file.readlines()]
        coors = np.array(lines).astype(float)
        X, Y, Z = coors[:, 0], coors[:, 1], coors[:, 2]
        points = [Point(coor[0], coor[1], coor[2]) for coor in coors]

    slices_x = Slices_x(X, Y, Z, points, slice_number_x, feature_type)
    slices_y = Slices_y(X, Y, Z, points, slice_number_y, feature_type)
    slices_z = Slices_z(X, Y, Z, points, slice_number_z, feature_type)
    return slices_x.get_feature() + slices_y.get_feature() + slices_z.get_feature()


if __name__ == '__main__':
    all_file_path = "../pointclouds-500"
    file_names = sorted(os.listdir(all_file_path), key=lambda x: int(x.split('.')[0]))

    slices_number = list(np.logspace(start=1, stop=int(np.log(100) / np.log(1.25)), num=15, base=1.25))
    slices_number = [int(num) for num in slices_number]

    os.chdir('../result_both')

    for i in range(len(slices_number)):
        n = slices_number[i]
        print(f"Slice: {n}")
        X = []  # 500 feature vectors
        for j, file_name in enumerate(file_names):
            file_path = f"{all_file_path}/{file_name}"
            features = get_feature(file_path, n, n, n, "both")
            X.append(features)
            # just to show the progress
            if j % 50 == 0:
                print(j)
            elif j == 499:
                print(j)
        X = np.array(X)
        np.save(f"X_{i}_{n}.npy", X, allow_pickle=True, fix_imports=True)

    Y = [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100 + [4] * 100  # 500 labels
    Y = np.array(Y)
    np.save("y.npy", Y, allow_pickle=True, fix_imports=True)