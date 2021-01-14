"""
Functions for processing grasps in different representations.
"""

import numpy as np

import matplotlib.pyplot as plt
import provider
from skimage.draw import polygon
from skimage.feature import peak_local_max
import cv2


def _bb_text_to_no(l, offset=(0, 0)):
    # Read Cornell grasping dataset.
    x, y = l.split()
    return [int(round(float(y))) - offset[0], int(round(float(x))) - offset[1]]


def _bb_text_to_float(l, offset=(0, 0)):
    # Read Cornell grasping force dataset.
    x, y = l.split()
    return [round(float(x), 3) - offset[0], round(float(y), 3) - offset[1]]


def _bb_text_to_no_3num(l, offset=(0, 0)):
    # Read Cornell grasping dataset for force.
    x, y, _ = l.split()
    return [(round(float(y))) - offset[0], (round(float(x))) - offset[1]]


class BoundingBoxes(object):
    """
    Convenience class for storing a set of grasps represented by bounding boxes.
    """
    def __init__(self, bbs=None):
        if bbs:
            self.bbs = bbs
        else:
            self.bbs = []

    def __getitem__(self, item):
        return self.bbs[item]

    def __iter__(self):
        return self.bbs.__iter__()

    def __getattr__(self, attr):
        # Call a function on all bbs
        if hasattr(BoundingBox, attr) and callable(getattr(BoundingBox, attr)):
            return lambda *args, **kwargs: list(map(lambda bb: getattr(bb, attr)(*args, **kwargs), self.bbs))
        else:
            raise AttributeError("Couldn't find function %s in BoundingBoxes or BoundingBox" % attr)

    @classmethod
    def load_from_array(cls, arr):
        bbs = []
        # print(arr.shape)   maybe 5 bbox
        # print(arr.shape[0])
        for i in range(arr.shape[0]):
            # print(i)
            bbp = arr[i, :, :].squeeze()
            # print(bbp)
            if bbp.max() == 0:
                break
            else:
                bbs.append(BoundingBox(bbp))
        return cls(bbs)

    @classmethod
    def load_from_file(cls, fname, id=None):
        bbs = []
        count_bbx = 0
        # print('id', id )
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3 = f.readline(), f.readline(), f.readline()
                try:
                    bb = np.array([
                        _bb_text_to_no(p0),
                        _bb_text_to_no(p1),
                        _bb_text_to_no(p2),
                        _bb_text_to_no_3num(p3)
                    ])
                    # print(bb.shape)
                    count_bbx += 1
                    # print(count_bbx)
                    bbs.append(BoundingBox(bb))
                    if count_bbx ==id:
                        bbs=[(BoundingBox(bb))]
                        break
                except ValueError:
                    # Some files contain weird values, ignore them.
                    continue
                # if count_bbx >1:
                #     break
        return cls(bbs), count_bbx

    @classmethod
    def load_from_file_force(cls, fname):
        bbs = []
        with open(fname) as f:
            while True:
                # Load 4 lines at a time, corners of bounding box.
                p0 = f.readline()
                if not p0:
                    break  # EOF
                p1, p2, p3, p4 = f.readline(), f.readline(), f.readline(), f.readline()
                # print(p0, p1, p2, p3, p4)
                try:
                    bb = np.array([
                        _bb_text_to_no(p0),
                        _bb_text_to_no(p1),
                        _bb_text_to_no(p2),
                        _bb_text_to_no(p3),
                        _bb_text_to_float(p4)
                    ])
                    # print(bb.shape)
                    bbs.append(BoundingBox(bb))
                except ValueError:
                    # Some files contain weird values, ignore them.
                    continue
        return cls(bbs)

    @classmethod
    def save_tactile(cls, fname, fwrite, value, id):
        bbs = []
        # print('id', id )
        count_bbx = 0
        with open(fname) as f:
            with open(fwrite, 'a') as f2:
                while True:
                    # Load 4 lines at a time, corners of bounding box.
                    p0 = f.readline()
                    if not p0:
                        break  # EOF
                    p1, p2, p3 = f.readline(), f.readline(), f.readline()
                    count_bbx += 1
                    if count_bbx == id:
                        f2.write(p0)
                        # p1 = f.readline()
                        f2.write(p1)
                        # p2 = f.readline()
                        f2.write(p2)
                        # p3 = f.readline()
                        f2.write(p3)
                        f2.write(value+'\n')

    def append(self, bb):
        self.bbs.append(bb)

    def copy(self):
        new_bbs = BoundingBoxes()
        for bb in self.bbs:
            new_bbs.append(bb.copy())
        return new_bbs

    def show(self, ax=None, shape=None):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            ax.imshow(np.zeros(shape))
            ax.axis([0, shape[1], shape[0], 0])
            self.plot(ax)
            plt.show()
        else:
            self.plot(ax)

    def draw(self, shape, position=True, angle=True, width=True, force=True):
        # print('--------------------------')

        if position:
            pos_out = np.zeros(shape)
        else:
            pos_out = None
        if angle:
            ang_out = np.zeros(shape)
        else:
            ang_out = None
        if width:
            width_out = np.zeros(shape)
        else:
            width_out = None

        if force:
            force_out = np.zeros(shape)
        else:
            force_out = None

        for bb in self.bbs:
            rr, cc = bb.compact_polygon_coords(shape)

            # print(len(rr), len(cc))
            # print('DEBUG grasp', bb.length, bb.width, bb.width/bb.length)
            # print(bb.length, bb.force)
            if position:
                pos_out[rr, cc] = 1.0
            if angle:
                ang_out[rr, cc] = bb.angle
            if width:
                width_out[rr, cc] = bb.length*0.7
            if force:
                force_out[rr, cc] = bb.force
            # test = (bb.force/2+0.5)*5
            # print((4.999-test)/0.45)

            # cv2.imshow("pos_out", np.uint8(pos_out * 255.0))
            # cv2.imshow("ang_out", np.uint8(ang_out * 255.0))
            # cv2.imshow("width_out", np.uint8(width_out * 255.0))
            # cv2.waitKey(0)

        # print(pos_out)

            # print(pos_out.shape)
        # provider.show_image(image=ang_out)
        return pos_out.astype('float16'), ang_out.astype('float16'), width_out.astype('float16'), force_out.astype('float16')

    def to_array(self, pad_to=0):
        a = np.stack([bb.points for bb in self.bbs])
        if pad_to:
           if pad_to > len(self.bbs):
               a = np.concatenate((a, np.zeros((pad_to - len(self.bbs), 4, 2))))
        return a.astype(np.int)

    @property
    def center(self):
        points = [bb.points for bb in self.bbs]
        return np.mean(np.vstack(points), axis=0).astype(np.int)


class BoundingBox:
    """
    Class for manipulating grasps represented as bounding boxes (i.e. 4 corners of a rectangle)
    """
    def __init__(self, points):
        # self.points = points
        # print(self.points.shape)  #(4,2)

        # new
        self.points = points[:4, :]
        self.forces = points[4:, :]
        # print(self.points)
        # print(self.forces)
        # print(self.force)
        # raw_input()

    def __str__(self):
        return str(self.points)

    @property
    def angle(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return (np.arctan2(-dy, dx) + np.pi/2) % np.pi - np.pi/2

    @property
    def as_grasp(self):
        return Grasp(self.center, self.angle, self.length, self.width)

    @property
    def center(self):
        return self.points.mean(axis=0).astype(np.int)

    @property
    def length(self):
        dx = self.points[1, 1] - self.points[0, 1]
        dy = self.points[1, 0] - self.points[0, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def width(self):
        dy = self.points[2, 1] - self.points[1, 1]
        dx = self.points[2, 0] - self.points[1, 0]
        return np.sqrt(dx ** 2 + dy ** 2)

    @property
    def force(self):
        left = self.forces[0, 0]
        right = self.forces[0, 1]
        mean = 0.5*(left+right)/5.0
        mean_normal = (mean - 0.5)/0.5
        return mean_normal

    def polygon_coords(self, shape=None):
        return polygon(self.points[:, 0], self.points[:, 1], shape)

    def compact_polygon_coords(self, shape=None):
        return Grasp(self.center, self.angle, self.length/3, self.width).as_bb.polygon_coords(shape)

    def iou(self, bb, angle_threshold=np.pi/6):
        if abs(self.angle - bb.angle) % np.pi > angle_threshold:
            return 0

        rr1, cc1 = self.polygon_coords()
        rr2, cc2 = polygon(bb.points[:, 0], bb.points[:, 1])

        try:
            r_max = max(rr1.max(), rr2.max()) + 1
            c_max = max(cc1.max(), cc2.max()) + 1
        except:
            return 0

        canvas = np.zeros((r_max, c_max))
        canvas[rr1, cc1] += 1
        canvas[rr2, cc2] += 1
        union = np.sum(canvas > 0)
        if union == 0:
            return 0
        intersection = np.sum(canvas == 2)
        return intersection * 1.0/union

    def copy(self):
        # print(type(self.points))
        # print(type(self.forces))
        return BoundingBox(np.concatenate((self.points,self.forces), axis=0).copy())

    def offset(self, offset):
        self.points += np.array(offset).reshape((1, 2))

    def rotate(self, angle, center):
        R = np.array(
            [
                [np.cos(-angle), np.sin(-angle)],
                [-1 * np.sin(-angle), np.cos(-angle)],
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(R, (self.points - c).T)).T + c).astype(np.int)

    def plot(self, ax, color=None):
        points = np.vstack((self.points, self.points[0]))
        ax.plot(points[:, 1], points[:, 0], color=color)

    def zoom(self, factor, center):
        T = np.array(
            [
                [1.0/factor, 0],
                [0, 1.0/factor]
            ]
        )
        c = np.array(center).reshape((1, 2))
        self.points = ((np.dot(T, (self.points - c).T)).T + c).astype(np.int)


class Grasp:
    """
    Class for manipulating grasps represented by a centre pixel, angle and width.
    """
    def __init__(self, center, angle, length=60, width=30):
        self.center = center
        self.angle = angle  # Positive angle means rotate anti-clockwise from horizontal.
        self.length = length
        self.width = width

    @property
    def as_bb(self):
        xo = np.cos(self.angle)
        yo = np.sin(self.angle)

        y1 = self.center[0] + self.length/2.0 * yo
        x1 = self.center[1] - self.length/2.0 * xo
        y2 = self.center[0] - self.length/2.0 * yo
        x2 = self.center[1] + self.length/2.0 * xo

        return BoundingBox(np.array(
            [
             [y1 - self.width/2.0 * xo, x1 - self.width/2.0 * yo],
             [y2 - self.width/2.0 * xo, x2 - self.width/2.0 * yo],
             [y2 + self.width/2.0 * xo, x2 + self.width/2.0 * yo],
             [y1 + self.width/2.0 * xo, x1 + self.width/2.0 * yo],
             ]
        ).astype(np.int))

    def max_iou(self, bbs):
        self_bb = self.as_bb
        max_iou = 0
        for bb in bbs:
            iou = self_bb.iou(bb)
            max_iou = max(max_iou, iou)
        return max_iou

    def plot(self, ax, color=None):
        self.as_bb.plot(ax, color)


def detect_grasps(point_img, ang_img, width_img=None, no_grasps=1, ang_threshold=5):
    """
    Detect Grasps in a GG-CNN output.
    """
    local_max = peak_local_max(point_img, min_distance=20, threshold_abs=0.2, num_peaks=no_grasps)
    # print(np.shape(local_max))
    # print(local_max)
    # exit()
    grasps = []

    for grasp_point_array in local_max:
        grasp_point = tuple(grasp_point_array)

        grasp_angle = ang_img[grasp_point]
        if ang_threshold > 0:
            if grasp_angle > 0:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].max()
            else:
                grasp_angle = ang_img[grasp_point[0] - ang_threshold:grasp_point[0] + ang_threshold + 1,
                                      grasp_point[1] - ang_threshold:grasp_point[1] + ang_threshold + 1].min()

        g = Grasp(grasp_point, grasp_angle)
        if width_img is not None:
            g.length = width_img[grasp_point]
            g.width = g.length/2.0

        grasps.append(g)

    return grasps

