import os
import numpy as np


class Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class UrbanObject:
    def __init__(self, X, Y, Z, points, slice_number, feature_type):
        self.feature_type = feature_type
        self.slice_number = slice_number
        self.points = points

        self.X_MIN, self.Y_MIN, self.Z_MIN = min(X), min(Y), min(Z)
        self.X_MAX, self.Y_MAX, self.Z_MAX = max(X), max(Y), max(Z)

        self.X_STEP = (self.X_MAX - self.X_MIN) / slice_number
        self.Y_STEP = (self.Y_MAX - self.Y_MIN) / slice_number
        self.Z_STEP = (self.Z_MAX - self.Z_MIN) / slice_number

        self.bbox_volume = (self.X_MAX - self.X_MIN) * (self.Y_MAX - self.Y_MIN) * (self.Z_MAX - self.Z_MIN)

    def is_inside(self, slice_type, point, slice_index):
        if slice_type == 'x':
            return point.x < self.X_MIN + (slice_index + 1) * self.X_STEP
        elif slice_type == 'y':
            return point.y < self.Y_MIN + (slice_index + 1) * self.Y_STEP
        elif slice_type == 'z':
            return point.z < self.Z_MIN + (slice_index + 1) * self.Z_STEP

    def fill_points(self, slices, slice_type):
        for point in self.points:
            for i, slice in enumerate(slices):
                if self.is_inside(slice_type, point, i):
                    slice.add(point)

    def get_slices(self, slice_type):
        slice_number = self.slice_number
        slices = [set() for _ in range(slice_number)]
        self.fill_points(slices, slice_type)
        return slices

    def construct_features(self, slices):
        slices_point_density = [len(slice) / self.bbox_volume for slice in slices]
        slices_point_numbers = [len(slice) for slice in slices]
        if self.feature_type == "density":
            return slices_point_density
        elif self.feature_type == "number":
            return slices_point_numbers
        elif self.feature_type == "both":
            return slices_point_density + slices_point_numbers

    def get_feature(self):
        features = []
        for slice_type in ['x', 'y', 'z']:
            slices = self.get_slices(slice_type)
            features += self.construct_features(slices)
        return features


def get_all_features(file_path, slice_number, feature_type):
    with open(file_path, mode="r") as file:
        lines = [line.strip().split() for line in file.readlines()]
        coors = np.array(lines).astype(float)
        X, Y, Z = coors[:, 0], coors[:, 1], coors[:, 2]
        points = [Point(coor[0], coor[1], coor[2]) for coor in coors]

    urban_object = UrbanObject(X, Y, Z, points, slice_number, feature_type)
    return urban_object.get_feature()


if __name__ == '__main__':
    all_file_path = "../pointclouds-500"
    file_names = sorted(os.listdir(all_file_path), key=lambda x: int(x.split('.')[0]))

    # these numbers come from the feature_curve.py, us these to generate all corresponding feature files,
    # to speed up the feature curve plotting (no need to regenerate features every time).
    slices_number = list(np.logspace(start=1, stop=int(np.log(100) / np.log(1.25)), num=15, base=1.25))
    slices_number = [int(num) for num in slices_number]

    os.chdir('../result_both')

    for i in range(len(slices_number)):
        n = slices_number[i]
        print(f"Slice: {n}")
        X = []  # 500 feature vectors
        for j, file_name in enumerate(file_names):
            file_path = f"{all_file_path}/{file_name}"
            features = get_all_features(file_path, n, "both")
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