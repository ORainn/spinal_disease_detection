import os
import random
from collections import Counter
from multiprocessing import Pool, cpu_count
from typing import Dict, Union
from tqdm import tqdm
import torch
from torchvision.transforms import functional as tf
from .dicom import DICOM, lazy_property
from .series import Series
from ..data_utils import read_annotation


class Study(dict):
    def __init__(self, study_dir, pool=None):
        # 图像列表
        dicom_list = []
        # 非None则启动多进程
        if pool is not None:
            # 存储异步结果
            async_results = []
            # 遍历给出的文件夹
            for dicom_name in os.listdir(study_dir):
                # 获取每张图像的绝对路径
                dicom_path = os.path.join(study_dir, dicom_name)
                # # 异步非阻塞地向数组中加入读取到灰度图信息
                async_results.append(pool.apply_async(DICOM, (dicom_path, )))

            # 遍历异步执行结果
            for async_result in async_results:
                # 等待所有数据读取完毕
                async_result.wait()
                # 获取图片
                dicom = async_result.get()
                # 将图片加入图像列表
                dicom_list.append(dicom)
        else:
            # 遍历给出的文件夹
            for dicom_name in os.listdir(study_dir):
                # 获取每张图像的绝对路径
                dicom_path = os.path.join(study_dir, dicom_name)
                # 获取图片
                dicom = DICOM(dicom_path)
                # 将图片加入图像列表
                dicom_list.append(dicom)

        dicom_dict = {}
        # 遍历图像列表
        for dicom in dicom_list:
            # 获取当前图像的序列ID
            series_uid = dicom.series_uid
            # 判断图像字典中是否有该序列ID
            if series_uid not in dicom_dict:
                # 初始化，在该序列ID下放入这张图像
                dicom_dict[series_uid] = [dicom]
            else:
                # 若已有该序列ID，则在后面加上这张图像
                dicom_dict[series_uid].append(dicom)

        super().__init__({k: Series(v) for k, v in dicom_dict.items()})

        # 初始化T2矢状面影像对应的ID
        self.t2_sagittal_uid = None
        # 初始化T2轴状位影像对应的ID
        self.t2_transverse_uid = None
        # 通过平均值最大的来剔除压脂像
        # 初始化T2矢状面影像对应的最大平均值
        max_t2_sagittal_mean = 0
        # 初始化T2轴状位影像对应的最大平均值
        max_t2_transverse_mean = 0
        # 遍历序列ID及其对应的序列
        for series_uid, series in self.items():
            # 判断当前序列的图像方向
            # 若为矢状面影像并且序列描述值为T2，可以确定该图像为T2矢状面影像
            if series.plane == 'sagittal' and series.t_type == 'T2':
                # 求出当前序列的均值
                t2_sagittal_mean = series.mean
                # 与当前最大的均值比较，进行替换
                if t2_sagittal_mean > max_t2_sagittal_mean:
                    max_t2_sagittal_mean = t2_sagittal_mean
                    # 替换为均值最大的影像对应的序列ID
                    self.t2_sagittal_uid = series_uid
            # 若为轴状位影像并且序列描述值为T2，可以确定该图像为T2轴状位影像
            if series.plane == 'transverse' and series.t_type == 'T2':
                # 求出当前序列的均值
                t2_transverse_mean = series.mean
                # 与当前最大的均值比较，进行替换
                if t2_transverse_mean > max_t2_transverse_mean:
                    max_t2_transverse_mean = t2_transverse_mean
                    # 替换为均值最大的影像对应的序列ID
                    self.t2_transverse_uid = series_uid
        # 若没有获取到T2矢状面影像的ID，就用T1矢状面影像来代替
        if self.t2_sagittal_uid is None:
            # 遍历序列ID及其对应的序列
            for series_uid, series in self.items():
                # 判断当前序列的图像方向
                # 若为矢状面影像并且序列描述值为T1，可以确定该图像为T1矢状面影像
                if series.plane == 'sagittal' and series.t_type != 'T1':
                    # 求出当前序列的均值
                    t2_sagittal_mean = series.mean
                    # 与当前最大的均值比较，进行替换
                    if t2_sagittal_mean > max_t2_sagittal_mean:
                        max_t2_sagittal_mean = t2_sagittal_mean
                        # 替换为均值最大的影像对应的序列ID
                        self.t2_sagittal_uid = series_uid
        # 若没有获取到T2轴状位影像的ID，就用没有序列标注的轴状位影像来代替
        if self.t2_transverse_uid is None:
            # 遍历序列ID及其对应的序列
            for series_uid, series in self.items():
                # 判断当前序列的图像方向
                # 若为轴状位影像，就用该图像代替T2轴状位影像
                if series.plane == 'transverse':
                    # 求出当前序列的均值
                    t2_transverse_mean = series.mean
                    # 与当前最大的均值比较，进行替换
                    if t2_transverse_mean > max_t2_transverse_mean:
                        max_t2_transverse_mean = t2_transverse_mean
                        # 替换为均值最大的影像对应的序列ID
                        self.t2_transverse_uid = series_uid

    @lazy_property
    def study_uid(self):
        # 为所有检查ID进行计数
        study_uid_counter = Counter([s.study_uid for s in self.values()])
        # 返回数量最多的检查ID
        return study_uid_counter.most_common(1)[0][0]

    @property
    def t2_sagittal(self) -> Union[None, Series]:
        """
        会被修改的属性不应该lazy
        :return: T2矢状面影像对应的ID
        """
        if self.t2_sagittal_uid is None:
            return None
        else:
            return self[self.t2_sagittal_uid]

    @property
    def t2_transverse(self) -> Union[None, Series]:
        """
        会被修改的属性不应该lazy
        :return: T2轴状位影像对应的ID
        """
        if self.t2_transverse_uid is None:
            return None
        else:
            return self[self.t2_transverse_uid]

    @property
    def t2_sagittal_middle_frame(self) -> Union[None, DICOM]:
        """
        会被修改的属性不应该lazy
        :return: T2矢状面中间帧影像
        """
        if self.t2_sagittal is None:
            return None
        else:
            return self.t2_sagittal.middle_frame

    def set_t2_sagittal_middle_frame(self, series_uid, instance_uid):
        # 断言序列ID存在
        assert series_uid in self
        # 设置T2矢状面影像ID为序列ID
        self.t2_sagittal_uid = series_uid
        # 设置T2矢状面中间帧影像的ID
        self.t2_sagittal.set_middle_frame(instance_uid)

    def t2_transverse_k_nearest(self, pixel_coord, k, size, max_dist, prob_rotate=0,
                                max_angel=0) -> (torch.Tensor, torch.Tensor):
        """
        T2轴状位距离最近的k张图像
        :param pixel_coord: (M, 2)
        :param k: 要选取的图像数
        :param size: 截取图像的大小
        :param max_dist: 最大距离范围
        :param prob_rotate: 旋转概率
        :param max_angel: 最大旋转角度
        :return: 图像张量(M, k, 1, height, width)，masks(M， k)
            masks: 为None的位置将被标注为True
        """
        # 若k值不大于0 或者没有T2轴状位对应的图像ID，直接创建k个图像并返回
        if k <= 0 or self.t2_transverse is None:
            # padding
            # 新建给定大小的零填充张量
            images = torch.zeros(pixel_coord.shape[0], k, 1, *size)
            # 新建给定大小的零填充bool类型张量
            masks = torch.zeros(*images.shape[:2], dtype=torch.bool)
            # 返回直接生成的k个图像和mask
            return images, masks
        # 将T2矢状面中间帧图像上的像素坐标转换成人坐标系上的坐标
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        # 返回在范围内距离该图像最近的k个图像
        dicoms = self.t2_transverse.k_nearest(human_coord, k, max_dist)
        images = []
        masks = []
        # 遍历人坐标系上的坐标以及所有图像中的序列
        for point, series in zip(human_coord, dicoms):
            temp_images = []
            temp_masks = []
            # 遍历每张图像
            for dicom in series:
                # 若图像为空
                if dicom is None:
                    # mask标注为True
                    temp_masks.append(True)
                    # 创建零填充张量
                    image = torch.zeros(1, *size)
                # 若图像不为空
                else:
                    # mask标注为False
                    temp_masks.append(False)
                    # 将人坐标系中的点投影到图像上，获取像素坐标
                    projection = dicom.projection(point)
                    # 获取旋转后的image tensor和distance map
                    image, projection = dicom.transform(
                        projection, size=[size[0]*2, size[1]*2], prob_rotate=prob_rotate, max_angel=max_angel,
                        tensor=False
                    )
                    # 裁剪图像
                    image = tf.crop(
                        image, int(projection[0]-size[0]//2), int(projection[1]-size[1]//2), size[0], size[1])
                    # 图像转为张量
                    image = tf.to_tensor(image)
                # 将处理好的图像添加到列表中
                temp_images.append(image)
            # 将张量列表堆叠为一个张量
            temp_images = torch.stack(temp_images, dim=0)
            # 将该张量添加到图像列表中
            images.append(temp_images)
            # 将mask添加到列表中
            masks.append(temp_masks)
        # 将图像列表堆叠成一个张量
        images = torch.stack(images, dim=0)
        # 将mask张量转为bool类型
        masks = torch.tensor(masks, dtype=torch.bool)
        # 返回图像张量及其对应的mask张量
        return images, masks

    def transform(self, v_coords, d_coords, transverse_size, sagittal_size=None, k_nearest=0,
                  max_dist=6, prob_rotate=0, max_angel=0, sagittal_shift=0):
        """
        生成study的训练数据
        :param v_coords: 垂直坐标
        :param d_coords: 水平坐标
        :param transverse_size: 轴状位图像大小
        :param sagittal_size:矢状面图像大小
        :param k_nearest: 选取的图像数量
        :param max_dist: 选取的图像距离当前图像的最大距离
        :param prob_rotate: 旋转概率
        :param max_angel: 最大旋转角度
        :param sagittal_shift: 矢状面偏移量
        :return: 训练用轴状位、矢状面图像、距离张量、像素距离以及mask
        """
        # 随机对中间帧进行偏移操作
        # 获取矢状面中间帧的实例ID
        dicom_idx = self.t2_sagittal.instance_uids[self.t2_sagittal_middle_frame.instance_uid]
        # 随机在[dicom_idx-sagittal_shift, dicom_idx+sagittal_shift]范围内选取一个新的实例ID
        dicom_idx = random.randint(dicom_idx-sagittal_shift, dicom_idx+sagittal_shift)
        # 将新的ID转为矢状面图像
        dicom: DICOM = self.t2_sagittal[dicom_idx]
        # 计算原先的坐标在偏移后的帧上的投影坐标
        # 得到像素坐标
        pixel_coord = torch.cat([v_coords, d_coords], dim=0)
        # 将像素坐标转为人坐标系上的坐标
        human_coord = self.t2_sagittal_middle_frame.pixel_coord2human_coord(pixel_coord)
        # 计算得到在偏移后的帧上的投影坐标
        pixel_coord = dicom.projection(human_coord)
        # 将被选中的矢状图转换成张量
        sagittal_image, pixel_coord, distmap = dicom.transform(
            pixel_coord, sagittal_size, prob_rotate, max_angel, distmap=True)

        # 因为锥体的轴状图太少了，所以只提取椎间盘的轴状图
        transverse_image, t_mask = self.t2_transverse_k_nearest(
            d_coords, k=k_nearest, size=transverse_size, max_dist=max_dist,
            prob_rotate=prob_rotate, max_angel=max_angel
        )
        # 返回训练用图像、距离张量、像素距离以及mask
        return transverse_image, sagittal_image, distmap, pixel_coord, t_mask


# 给定文件夹路径，创建Study
def _construct_studies(data_dir, multiprocessing=False):
    # 初始化Study字典
    studies: Dict[str, Study] = {}
    # 判断是否使用多进程
    if multiprocessing:
        pool = Pool(cpu_count())
    else:
        pool = None
    # 遍历文件夹，通过进度条显示遍历进度
    for study_name in tqdm(os.listdir(data_dir), ascii=True):
        # 获取子文件夹路径
        study_dir = os.path.join(data_dir, study_name)
        # 初始化一个Study
        study = Study(study_dir, pool)
        # 将该Study填入字典中
        studies[study.study_uid] = study

    if pool is not None:
        pool.close()
        pool.join()
    # 返回字典
    return studies


# 记录标注有误的中间帧数量
def count_error_middle_frame(studies, annotation):
    # 初始化字典，记录每种错误情况下的ID
    counter = {
        't2_sagittal_not_found': [],
        't2_sagittal_miss_match': [],
        't2_sagittal_middle_frame_miss_match': []
    }
    # 遍历标注字典的key
    for k in annotation.keys():
        # 判断该标注信息是否已被加载到Study字典中
        if k[0] in studies:
            # 获取该Study
            study = studies[k[0]]
            # 若没有T2矢状面的图像
            if study.t2_sagittal is None:
                # “T2矢状面图像未找到”情况加上对应的StudyID
                counter['t2_sagittal_not_found'].append(study.study_uid)
            # 若T2矢状面图像的ID与记录的不一致
            elif study.t2_sagittal_uid != k[1]:
                # “T2矢状面图像匹配错误”情况加上对应的StudyID
                counter['t2_sagittal_miss_match'].append(study.study_uid)
            else:
                # 获取T2矢状面图像
                t2_sagittal = study.t2_sagittal
                # 获取T2矢状面图像的Z轴方向ID
                gt_z_index = t2_sagittal.instance_uids[k[2]]
                # 获取T2矢状面图像中间帧
                middle_frame = t2_sagittal.middle_frame
                # 获取中间帧的Z轴方向ID
                z_index = t2_sagittal.instance_uids[middle_frame.instance_uid]
                # 判断T2矢状面图像的Z轴方向ID是否与中间帧的Z轴方向ID一致
                if abs(gt_z_index - z_index) > 1:
                    # “T2矢状面图像中间帧匹配错误”情况加上对应的StudyID
                    counter['t2_sagittal_middle_frame_miss_match'].append(study.study_uid)
    # 返回计数字典
    return counter


# 设置中间帧
def set_middle_frame(studies: Dict[str, Study], annotation):
    # 遍历标注字典的key
    for k in annotation.keys():
        # 判断某Study是否在字典中
        if k[0] in studies:
            # 获取该Study
            study = studies[k[0]]
            # 为该Study设置T2矢状面中间帧
            study.set_t2_sagittal_middle_frame(k[1], k[2])


def construct_studies(data_dir, annotation_path=None, multiprocessing=False):
    """
    方便批量构造study的函数
    :param data_dir: 存放study的文件夹
    :param multiprocessing: 多进程处理
    :param annotation_path: 如果有标注，那么根据标注来确定定位帧
    :return: 批量构造的study、读取的标注信息、有误的study标注数量
    """
    # 先生成Study字典
    studies = _construct_studies(data_dir, multiprocessing)

    # 使用annotation制定正确的中间帧
    if annotation_path is None:
        return studies
    else:
        # 读取标注信息
        annotation = read_annotation(annotation_path)
        # 记录错误中间帧数量
        counter = count_error_middle_frame(studies, annotation)
        # 设置中间帧
        set_middle_frame(studies, annotation)
        # 返回批量构造的study、读取的标注信息、有误的study标注数量
        return studies, annotation, counter
