from MinimumBoundingBox import MinimumBoundingBox
import cv2
import numpy as np
import os
import provider
import copy
from skimage.draw import circle
import math
from matplotlib import patches


_mask_patten ="/media/aarons/hdd/dataset/rgbd-dataset/rgbd-dataset/soda_can/soda_can_1"
img_id = "soda_can_1_1_28_mask.png"
#
#
#
_mask_patten = "/media/aarons/hdd/dataset/rgbd-dataset/rgbd-dataset/lemon/lemon_1"
img_id = "lemon_1_1_155_mask.png"
# _mask_patten = '/media/aarons/hdd/dataset/rgbd-dataset/rgbd-dataset/lightbulb/lightbulb_2'
# img_id ="lightbulb_2_1_16_mask.png"
# _mask_patten ="/media/aarons/hdd/dataset/YCB/ycb/030_fork/masks"
# img_id = "NP1_0_mask.pbm"


class BoundingBox:
    def __init__(self, name='newBBX', vis=False):
        self.lineThickness = 2
        self.target_value = 255   # mask value: white
        self.mask = None
        self.vis = vis
        self.name = name
        if self.target_value == 255:
            self.other_value = 0
        else:
            self.other_value = 255

    def load_data_from_file(self, path):
        self.mask = cv2.imread(path, -1)

    def load_image(self, img):
        self.mask = img

    def rotate_zoom(self):
        pass

    def draw(self, shape, mask_points, center, ort, length, bbx_length, pos_out=None, ang_out=None, width_out=None, newbbx=True):
        pos_img = []

        # draw a ellipse
        g_ell_center = center
        # print("link", length)True
        # print("length", bbx_length)

        g_ell_width = length*1.8
        # g_ell_width = max(length, bbx_length/3)*2.0
        g_ell_height = 0.5*g_ell_width

        # print(g_ell_width, g_ell_height)
        g_ell_angle = -math.degrees(ort)
        g_ellipse = patches.Ellipse(g_ell_center,g_ell_width, g_ell_height, g_ell_angle)

        if newbbx:
            pos_out = np.zeros(shape)
            ang_out = np.zeros(shape)
            width_out = np.zeros(shape)

        for point in mask_points:
            # dis = math.sqrt((point[0] - center[0]) ** 2 + (point[1] - center[1]) ** 2)
            # if dis <= length*0.5:
            #     pos_img.append(point)
            if g_ellipse.contains_point(point,radius=0.0):
                pos_img.append(point)
        # print(len(mask_points))
        # print(len(pos_img))
        # pos_img = mask_points
        if self.target_value==255.0:
            rr, cc = self.get_mask_points(pos_img)
            pos_out[rr, cc] = 1.0
            ang_out[rr, cc] = ort
            print(ort)
            width_out[rr, cc] = length
        return pos_out, ang_out, width_out


    def get_distance(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def get_mask_points(self, mask_points):
        rr=[]
        cc=[]
        for i in range(len(mask_points)):
            rr.append(mask_points[i][1])
            cc.append(mask_points[i][0])
        return rr, cc

    def test_center(self, center, mask_points, edge_point):
        outlier = False
        center_test = list(center)
        center_test = [int(center_test[0]), int(center_test[1])]
        # print(center_test)
        if center_test  not in mask_points:
            # print("outlier! out of mask", center_test)
            outlier = True
            return  outlier
        mindis = None
        for edge in edge_point:
            dis = math.sqrt((center[0] - edge[0]) ** 2 + (center[1] - edge[1]) ** 2)
            if mindis == None:
                mindis = dis
        if mindis <3:
            outlier = True
            # print("outlier! distance")
        return outlier

    def symbol(self, x, points):
        k = (points[1][1]-points[0][1])/(points[1][0]-points[0][0]+0.001)
        return x[0]*k -k*points[0][0] + points[0][1] - x[1]

    def split_mask(self, maskpoints, link):
        mask_top = copy.deepcopy(self.mask)
        mask_bottom = copy.deepcopy(self.mask)
        for point in maskpoints:
            if self.symbol(point, link) >0:
                mask_bottom[point[1],point[0]] = 0
            else:
                mask_top[point[1],point[0]] = 0
        return mask_top, mask_bottom

    def subBBX(self, image, pos_out, ang_out, width_out, main_edge, name):
        new_edges = []
        if self.target_value == 255.0:
            color = (1, 1, 1)  # white
        else:
            color = (0, 0, 0)
        mask_points, edge_point = self.extract_point(image)
        bounding_box = MinimumBoundingBox(mask_points)
        corner = list(bounding_box.corner_points)
        center = list(bounding_box.rectangle_center)
        vertex = self.norm_bbx(corner, center)
        outlier = self.test_center(center, mask_points, edge_point)
        if outlier:
            return outlier, pos_out, ang_out, width_out
        for edge in edge_point:
            if edge in main_edge:
                new_edges.append(edge)
        link, short_dis, orientation = self.find_width(new_edges, center)
        if link is None:
            outlier = True
            return outlier, pos_out, ang_out, width_out
        pos_out, ang_out, width_out = self.draw(image.shape, mask_points,center, orientation,
                                                short_dis, bounding_box.length_parallel,
                                                pos_out, ang_out, width_out, newbbx=False)

        if self.vis:
            #  for visulization
            cv2.line(image, (int(link[0][0]), int(link[0][1])), (int(link[1][0]), int(link[1][1])), (0, 0, 0), self.lineThickness)
            cv2.line(image, (int(vertex[0][0]), int(vertex[0][1])), (int(vertex[1][0]), int(vertex[1][1])), color,
                     self.lineThickness)
            cv2.line(image, (int(vertex[1][0]), int(vertex[1][1])), (int(vertex[2][0]), int(vertex[2][1])), color,
                     self.lineThickness)
            cv2.line(image, (int(vertex[2][0]), int(vertex[2][1])), (int(vertex[3][0]), int(vertex[3][1])), color,
                     self.lineThickness)
            cv2.line(image, (int(vertex[3][0]), int(vertex[3][1])), (int(vertex[0][0]), int(vertex[0][1])), color,
                     self.lineThickness)

            rr, cc = circle(center[1], center[0], 5)
            image[rr, cc] = 0
            provider.show_image(image=image, name=name)
        return outlier, pos_out, ang_out, width_out

    def get_BBX(self):
        # global mask
        mask_points, edge_point = self.extract_point(self.mask)
        if self.target_value == 255.0:
            color = (1, 1, 1)  # white
        else:
            color = (0, 0, 0)        # black
        if len(mask_points)<3:
            return None, None, None, False, None

        bounding_box = MinimumBoundingBox(mask_points)
        # print(bounding_box)

        # print(bounding_box.area)
        corner = list(bounding_box.corner_points)
        center = list(bounding_box.rectangle_center)
        vertex = self.norm_bbx(corner, center)  # sort the four corner from left top in clockwise
        # print(corner, list(center))
        outlier = self.test_center(center, mask_points, edge_point)
        if outlier:
            return None, None, None, outlier, None
        link, short_dis, orientation = self.find_width(edge_point, center) #find the width for grasping [[x1,y1],[x2,y2]]


        if link is None:
            print("No best link find!")
            return None, None, None, False, link


        pos_out, ang_out, width_out = self.draw(self.mask.shape,mask_points,center,orientation, short_dis, bounding_box.length_parallel)
        # provider.show_image(ang_out, name='ang_out')
        # provider.show_image(pos_out, name='pos_out')

        mask_top, mask_bottom = self.split_mask(mask_points, link)
        outlier_sub, pos_out, ang_out, width_out = self.subBBX(mask_top, pos_out, ang_out, width_out, edge_point, name='top')
        # provider.show_image(ang_out, name='top_width_out')
        outlier_sub, pos_out, ang_out, width_out = self.subBBX(mask_bottom, pos_out, ang_out, width_out, edge_point, name='bottom')
        # provider.show_image(ang_out, name='bottom_width_out')



        if self.vis:
            #  for visulization
            cv2.line(self.mask, (int(link[0][0]), int(link[0][1])), (int(link[1][0]), int(link[1][1])), (0, 0, 0), self.lineThickness)
            cv2.line(self.mask, (int(vertex[0][0]), int(vertex[0][1])), (int(vertex[1][0]), int(vertex[1][1])), color,
                     self.lineThickness)
            cv2.line(self.mask, (int(vertex[1][0]), int(vertex[1][1])), (int(vertex[2][0]), int(vertex[2][1])), color,
                     self.lineThickness)
            cv2.line(self.mask, (int(vertex[2][0]), int(vertex[2][1])), (int(vertex[3][0]), int(vertex[3][1])), color,
                     self.lineThickness)
            cv2.line(self.mask, (int(vertex[3][0]), int(vertex[3][1])), (int(vertex[0][0]), int(vertex[0][1])), color,
                     self.lineThickness)

            rr, cc = circle(center[1], center[0], 5)
            self.mask[rr, cc] = 0
            provider.show_image(image=(self.mask), name= self.name)
            # provider.show_image(image=mask_top, name='mask_top')
            # provider.show_image(image=mask_bottom, name='mask_bottom')
        # return [int(center[0]), int(center[1])], short_dis,
        return  pos_out.astype(np.float32), ang_out.astype(np.float32), width_out.astype(np.float32), outlier, link

    def norm_bbx(self, points, center):
        pointlist = []

        for point in points:
            pointlist.append(list(point))
        # print(pointlist)
        # print(center)
        l_top = pointlist[0]
        # if
        # print(l_top)
        dis = []
        dis.append( math.sqrt((l_top[0] - pointlist[1][0]) ** 2 + (l_top[1] - pointlist[1][1]) ** 2))
        dis.append(math.sqrt((l_top[0] - pointlist[2][0]) ** 2 + (l_top[1] - pointlist[2][1]) ** 2))
        dis.append(math.sqrt((l_top[0] - pointlist[3][0]) ** 2 + (l_top[1] - pointlist[3][1]) ** 2))
        diagonal_length = np.max([dis[0], dis[1], dis[2]])

        for i in range(len(dis)):
            if dis[i] == diagonal_length:
                point_3 = pointlist[i+1]
        pointlist.remove(l_top)
        pointlist.remove(point_3)
        # diagonal_point =
        # print([dis1, dis2, dis3])
        # print(np.median([dis[0], dis[1], dis[2]]))
        # for i in range(len(points)):
            # if points[i][0] < center[0] and points[i][1] < center[1]:
            #       l_top = points[i]                                                                 
            # if points[i][0] > center[0] and points[i][1] < center[1]:
            #     r_top = points[i]
            # if points[i][0] > center[0] and points[i][1] > center[1]:
            #     r_bottom = points[i]
            # if points[i][0] < center[0] and points[i][1] > center[1]:
            #     l_bottom = points[i]
        return [l_top, pointlist[0], point_3, pointlist[1]]

    def extract_point(self, image):
        if len(image.shape) > 2:
            print("Not a mask image!")
            return
        else:
            mask_set = []
            edge_set = []
            # print('other value', self.other_value)
            for i in range(1, image.shape[0]-1):
                for j in range(1, image.shape[1]-1):
                    if image[i, j] > 0:
                        mask_set.append([j, i])
                        if image[i + 1, j] == self.other_value or image[i - 1, j] == self.other_value or image[
                            i, j - 1] == self.other_value or image[i, j + 1] == self.other_value:
                            edge_set.append([j, i])
            # print('mask set', len(mask_set))
            return mask_set, edge_set

    def find_width(self, edge, center):
        np.float32(edge)
        np.float32(center)
        # print(len(edge))
        link_set = []
        best_link = None
        short_dis = None
        for point1 in edge:
            other_set = copy.deepcopy(edge)
            other_set.remove(point1)
            # print(len(other_set))
            for point2 in other_set:
                angle1_val = (point1[1] - center[1]) / (point1[0] - center[0] +0.001)
                angle2_val = (point2[1] - center[1]) / (point2[0] - center[0] +0.001)
                # angle1 = math.atan2(point[1]-center[1], point[0]-center[0])
                # angle1 = math.atan((point[1]-center[1])/(point[0]-center[0]))

                if abs(angle2_val - angle1_val) < 0.2 and (point1[1] - center[1]) * (point2[1] - center[1]) < 0 \
                        and (point1[0] - center[0]) * (point2[0] - center[0]) < 0:
                    # print((point1[1]-center[1])*(point2[1]-center[1]))
                    link_set.append([point2, point1])
        for link in link_set:
            dis = math.sqrt((link[0][0] - link[1][0]) ** 2 + (link[0][1] - link[1][1]) ** 2)
            if short_dis is None:
                short_dis = dis
                best_link = link
            else:
                if dis < short_dis:
                    short_dis = dis
                    best_link = link
        if best_link:
            # angle =math.atan(float(best_link[0][1]-best_link[1][1])/float(best_link[0][0]-best_link[1][0])+0.0001)  # in degree

            dx = float(best_link[0][0]-best_link[1][0])
            dy = float(best_link[0][1]-best_link[1][1])

            angle = (math.atan2(-dy,dx) + np.pi / 2) % np.pi - np.pi / 2
        else:
            angle = None
        return best_link, short_dis, angle



# bbx = BoundingBox()
# bbx.load_data_from_file(os.path.join(_mask_patten, img_id))
# pos_out, ang_out, width_out = bbx.get_BBX()


