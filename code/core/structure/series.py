import torch
from collections import Counter
from typing import List
from .dicom import DICOM, lazy_property


class Series(list):
    """
    将DICOM的序列，按照dim的方向，根据image_position对DICOM进行排列
    """
    def __init__(self, dicom_list: List[DICOM]):
        # 遍历所有图像，获取图像切开的方向
        planes = [dicom.plane for dicom in dicom_list]
        # 记录每个方向上的图像数量
        plane_counter = Counter(planes)
        # 获取数量最多的那个切开方向
        self.plane = plane_counter.most_common(1)[0][0]
        # 是否为水平切开
        if self.plane == 'transverse':
            dim = 2
        # 是否为左右切开
        elif self.plane == 'sagittal':
            dim = 0
        # 是否为前后切开
        elif self.plane == 'coronal':
            dim = 1
        else:
            dim = None
        # 取出切开方向为当前方向的图像
        dicom_list = [dicom for dicom in dicom_list if dicom.plane == self.plane]
        # 如果有图像
        if dim is not None:
            # 将图像根据image_position进行排列
            dicom_list = sorted(dicom_list, key=lambda x: x.image_position[dim], reverse=True)
        super().__init__(dicom_list)
        # 设置实例ID
        self.instance_uids = {d.instance_uid: i for i, d in enumerate(self)}
        # 设置中间帧ID
        self.middle_frame_uid = None

    def __getitem__(self, item) -> DICOM:
        # 判断item是否为字符串
        if isinstance(item, str):
            # 通过实例ID查找到对应的图像并返回
            item = self.instance_uids[item]
        return super().__getitem__(item)

    @lazy_property
    def t_type(self):
        # 对序列描述值进行计数
        t_type_counter = Counter([d.t_type for d in self])
        # 返回数量最多的那种序列描述值
        return t_type_counter.most_common(1)[0][0]

    @lazy_property
    def mean(self):
        output = 0
        i = 0
        # 遍历图像
        for dicom in self:
            # 获取图像的均值
            mean = dicom.mean
            if mean is None:
                continue
            # 累计输出
            output = i / (i + 1) * output + mean / (i + 1)
            i += 1
        return output

    @property
    def middle_frame(self) -> DICOM:
        """
        会被修改的属性不应该lazy
        :return:
        """
        # 判断中间帧ID是否非空
        if self.middle_frame_uid is not None:
            # 非空则返回ID
            return self[self.middle_frame_uid]
        else:
            # 空则返回图像数量数量-1整除2，即最中间那张图片
            return self[(len(self) - 1) // 2]

    # 设置输入的实例ID为中间帧ID
    def set_middle_frame(self, instance_uid):
        self.middle_frame_uid = instance_uid

    @property
    def image_positions(self):
        positions = []
        # 遍历图像
        for dicom in self:
            # 累计每张图像的位置
            positions.append(dicom.image_position)
        # 返回所有图像的位置
        return torch.stack(positions, dim=0)

    @property
    def image_orientations(self):
        orientations = []
        # 遍历图像
        for dicom in self:
            # 累计每张图像的方向
            orientations.append(dicom.image_orientation)
        # 返回所有图像的方向
        return torch.stack(orientations, dim=0)

    @property
    def unit_normal_vectors(self):
        vectors = []
        # 遍历图像
        for dicom in self:
            # 累计每张图像的单位法向量
            vectors.append(dicom.unit_normal_vector)
        # 返回所有图像的单位法向量
        return torch.stack(vectors, dim=0)

    @lazy_property
    def series_uid(self):
        # 记录所有图像的序列ID
        study_uid_counter = Counter([d.series_uid for d in self])
        # 返回数量最多的那个序列ID
        return study_uid_counter.most_common(1)[0][0]

    @lazy_property
    def study_uid(self):
        # 记录所有图像的检查ID
        study_uid_counter = Counter([d.study_uid for d in self])
        # 返回数量最多的那个检查ID
        return study_uid_counter.most_common(1)[0][0]

    @lazy_property
    def series_description(self):
        # 记录所有图像的序列描述
        study_uid_counter = Counter([d.series_description for d in self])
        # 返回数量最多的那个序列描述
        return study_uid_counter.most_common(1)[0][0]

    def point_distance(self, coord: torch.Tensor):
        """
        点到序列中每一张图像平面的距离，单位为毫米
        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为NxM的矩阵或者长度为M的向量，M是序列的长度
        """
        # 调用dicom.py中的函数计算距离
        return torch.stack([dicom.point_distance(coord) for dicom in self], dim=1).squeeze()

    def k_nearest(self, coord: torch.Tensor, k, max_dist) -> List[List[DICOM]]:
        """

        :param coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :param k:
        :param max_dist: 如果距离大于max dist，那么返回一个None
        :return:
        """
        # 首先计算所有点的距离
        distance = self.point_distance(coord)
        # 获取距离张量从小到大排列后每个数值原来的索引
        indices = torch.argsort(distance, dim=-1)
        # 若只有一个距离
        if len(indices.shape) == 1:
            # 返回在范围内的图像
            return [[self[i] if distance[i] < max_dist else None for i in indices[:k]]]
        else:
            # 返回在范围内符合条件的k个图像
            return [[self[i] if row_d[i] < max_dist else None for i in row[:k]]
                    for row, row_d in zip(indices, distance)]

