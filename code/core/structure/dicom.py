import random

import SimpleITK as sitk
import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image

from ..data_utils import resize, rotate, gen_distmap
from ..dicom_utils import DICOM_TAG


# 使用懒特性延迟初始化，提升计算性能
def lazy_property(func):
    # 设置属性名称
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        # 判断是否未设置属性名称
        if not hasattr(self, attr_name):
            # 设置属性名称
            setattr(self, attr_name, func(self))
        # 返回已设置好名称的属性名称
        return getattr(self, attr_name)
    # 返回懒特性
    return _lazy_property


def str2tensor(s: str) -> torch.Tensor:
    """
    将以“\\”等分割开的字符串拆分成张量
    :param s: numbers separated by '\\', eg.  '0.71875\\0.71875 '
    :return: 1-D tensor
    """
    # 调用自带方法返回拆分后生成的tensor
    return torch.tensor(list(map(float, s.split('\\'))))


# 对输入张量进行单位化，获取单位向量
def unit_vector(tensor: torch.Tensor, dim=-1):
    # 求出最后一维上所有取值的平方根
    norm = (tensor ** 2).sum(dim=dim, keepdim=True).sqrt()
    # 除以平方根完成单位化
    return tensor / norm

# 获取图像的单位法向量
def unit_normal_vector(orientation: torch.Tensor):
    # 将输入张量最后一维的数据取出[1, 2, 0]位置的数据
    temp1 = orientation[:, [1, 2, 0]]
    # 将输入张量最后一维的数据取出[2, 0, 1]位置的数据
    temp2 = orientation[:, [2, 0, 1]]
    # 两个临时张量相乘，并获取计算结果的前两列
    output = temp1 * temp2[[1, 0]]
    # 获取乘积前两列之差
    output = output[0] - output[1]
    # 调用上面的方法对张量进行单位化，返回单位法向量
    return unit_vector(output, dim=-1)


class DICOM:
    """
    解析dicom文件
    属性：
        study_uid：检查ID
        series_uid：序列ID
        instance_uid：图像ID
        series_description：序列描述，用于区分T1、T2等
        pixel_spacing: 长度为2的向量，像素的物理距离，单位是毫米
        image_position：长度为3的向量，图像左上角在人坐标系上的坐标，单位是毫米
        image_orientation：2x3的矩阵，第一行表示图像从左到右的方向，第二行表示图像从上到下的方向，单位是毫米？
        unit_normal_vector: 长度为3的向量，图像的单位法向量，单位是毫米？
        image：PIL.Image.Image，图像
    注：人坐标系，规定人体的左边是X轴的方向，从面部指向背部的方向表示y轴的方向，从脚指向头的方向表示z轴的方向
    """

    def __init__(self, file_path):
        # 设置文件路径
        self.file_path = file_path
        self.error_msg = ''
        # 新建SimpleITK的图像文件读取器
        reader = sitk.ImageFileReader()
        # 加载私有的元信息
        reader.LoadPrivateTagsOn()
        # GDCMImageIO是一个读取和写入DICOM v3和ACR/NEMA图像的ImageIO类。在这里GDCMImageIO对象被创建并与ImageFileReader相连
        reader.SetImageIO('GDCMImageIO')
        # 设置读取器读取的路径
        reader.SetFileName(file_path)
        try:
            # 读取图像信息
            reader.ReadImageInformation()
        except RuntimeError:
            pass

        try:
            # 获取检查ID
            self.study_uid = reader.GetMetaData(DICOM_TAG['studyUid'])
        except RuntimeError:
            self.study_uid = ''

        try:
            # 获取序列ID
            self.series_uid: str = reader.GetMetaData(DICOM_TAG['seriesUid'])
        except RuntimeError:
            self.series_uid = ''

        try:
            # 获取图像ID
            self.instance_uid: str = reader.GetMetaData(DICOM_TAG['instanceUid'])
        except RuntimeError:
            self.instance_uid = ''

        try:
            # 获取序列描述，用于区分T1、T2等
            self.series_description: str = reader.GetMetaData(DICOM_TAG['seriesDescription'])
        except RuntimeError:
            self.series_description = ''

        try:
            # 获取像素的物理距离，是长度为2的向量，单位是毫米
            self._pixel_spacing = reader.GetMetaData(DICOM_TAG['pixelSpacing'])
        except RuntimeError:
            self._pixel_spacing = None

        try:
            # 获取长度为3的坐标向量，图像左上角在人坐标系上的坐标，单位是毫米
            self._image_position = reader.GetMetaData(DICOM_TAG['imagePosition'])
        except RuntimeError:
            self._image_position = None

        try:
            # 获取2x3的方向矩阵，第一行表示图像从左到右的方向，第二行表示图像从上到下的方向，单位是毫米
            self._image_orientation = reader.GetMetaData(DICOM_TAG['imageOrientation'])
        except RuntimeError:
            self._image_orientation = None

        try:
            # 执行读取器读取dicom序列
            image = reader.Execute()
            # 将SimpleITK对象转换为ndarray
            array = sitk.GetArrayFromImage(image)[0]
            # 因为我们要将图像写为jpg格式，因此需要转换为UInt8像素类型以及重新调整图像强度（默认为[0,255]）
            # 虽然SimpleITK有现成的方法，但是在python 3.7和ubuntu 20.04环境下可能因为未知原因崩溃，因此在这里就自己写了
            # 首先转为float64数据类型
            array = array.astype(np.float64)
            # 使用极差标准化重新将ndarray的值映射到[0,255]上
            array = (array - array.min()) * (255 / (array.max() - array.min()))
            # 转为UInt8数据类型
            array = array.astype(np.uint8)
            # 将ndarray转化为PIL Image并设置
            self.image: Image.Image = tf.to_pil_image(array)
        except RuntimeError:
            self.image = None

    @lazy_property
    def pixel_spacing(self):
        # 判断像素点间距是否为None
        if self._pixel_spacing is None:
            # 新建一个空值填充的长度为2的张量并返回
            return torch.full([2, ], fill_value=np.nan)
        else:
            # 使用前面定义好的函数处理读取到的像素点间距并返回张量
            return str2tensor(self._pixel_spacing)

    @lazy_property
    def image_position(self):
        # 判断图像坐标是否为None
        if self._image_position is None:
            # 新建一个空值填充的长度为3的张量并返回
            return torch.full([3, ], fill_value=np.nan)
        else:
            # 使用前面定义好的函数处理读取到的图像坐标并返回张量
            return str2tensor(self._image_position)

    @lazy_property
    def image_orientation(self):
        # 判断图像方向是否为None
        if self._image_orientation is None:
            # 新建一个空值填充的大小为2×3的张量并返回
            return torch.full([2, 3], fill_value=np.nan)
        else:
            # 调用前面定义好的函数，首先处理读取到的图像方向并reshape为大小为2×3的张量，之后对该张量进行单位化
            return unit_vector(str2tensor(self._image_orientation).reshape(2, 3))

    @lazy_property
    def unit_normal_vector(self):
        # 判断图像方向是否为None
        if self.image_orientation is None:
            # 新建一个空值填充的长度为3的张量并返回
            return torch.full([3, ], fill_value=np.nan)
        else:
            # 调用前面定义好的函数对图像方向求出单位法向量并返回
            return unit_normal_vector(self.image_orientation)

    @lazy_property
    def t_type(self):
        # 判断序列描述值是否包含“T1”和“T2”，包含就返回
        if 'T1' in self.series_description.upper():
            return 'T1'
        elif 'T2' in self.series_description.upper():
            return 'T2'
        else:
            return None

    @lazy_property
    def plane(self):
        # 判断单位法向量是否为空
        if torch.isnan(self.unit_normal_vector).all():
            return None
        # 单位法向量非空则判断Z方向单位向量值是否大于0.75
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 0., 1.])).abs() > 0.75:
            # 轴状位，水平切开
            return 'transverse'
        # Z方向单位向量值小于等于.75则判断Y方向单位向量值是否大于0.75
        elif torch.matmul(self.unit_normal_vector, torch.tensor([1., 0., 0.])).abs() > 0.75:
            # 矢状位，左右切开
            return 'sagittal'
        # Y方向单位向量值小于等于.75则判断X方向单位向量值是否大于0.75
        elif torch.matmul(self.unit_normal_vector, torch.tensor([0., 1., 0.])).abs() > 0.75:
            # 冠状位，前后切开
            return 'coronal'
        else:
            # 不知道
            return None

    @lazy_property
    def mean(self):
        # 判断图像是否为None
        if self.image is None:
            return None
        else:
            # 非空则将图像转化为张量并返回
            return tf.to_tensor(self.image).mean()

    @property
    def size(self):
        """
        :return: width and height
        """
        # 判断图像是否为None
        if self.image is None:
            return None
        else:
            # 非空则将图像的大小返回
            return self.image.size

    def pixel_coord2human_coord(self, coord: torch.Tensor) -> torch.Tensor:
        """
        将图像上的像素坐标转换成人坐标系上的坐标
        :param coord: 像素坐标，Nx2的矩阵或者长度为2的向量
        :return: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        """
        # 先将像素坐标与像素点间距相乘得到像素点的物理坐标，之后与图像的方向向量相乘得到具体方向上的物理坐标，最后加上图像的位置得到人坐标系上精确的物理坐标
        return torch.matmul(coord.cpu() * self.pixel_spacing, self.image_orientation) + self.image_position

    def point_distance(self, human_coord: torch.Tensor) -> torch.Tensor:
        """
        点到图像平面的距离，单位为毫米
        :param human_coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return: 长度为N的向量或者标量
        """
        # 先将人坐标系上的物理坐标减去图像的位置，得到相对的物理坐标，之后与单位法向量相乘，取绝对值后就得到了关键点的物理距离
        return torch.matmul(human_coord.cpu() - self.image_position, self.unit_normal_vector).abs()

    def projection(self, human_coord: torch.Tensor) -> torch.Tensor:
        """
        将人坐标系中的点投影到图像上，并输出像素坐标
        :param human_coord: 人坐标系坐标，Nx3的矩阵或者长度为3的向量
        :return:像素坐标，Nx2的矩阵或者长度为2的向量
        """
        # 先将人坐标系上的物理坐标减去图像的位置，得到相对的物理坐标，之后与转置后的方向矩阵相乘得到余弦值
        cos = torch.matmul(human_coord - self.image_position, self.image_orientation.transpose(0, 1))
        # 将余弦值➗像素间距离后，四舍五入并返回
        return (cos / self.pixel_spacing).round()

    def transform(self, pixel_coord: torch.Tensor,
                  size=None, prob_rotate=0, max_angel=0, distmap=False, tensor=True) -> (torch.Tensor, torch.Tensor):
        """
        返回image tensor和distance map
        :param pixel_coord: 像素坐标
        :param size: 图像大小
        :param prob_rotate: 旋转可能性
        :param max_angel: 最大旋转角
        :param distmap: 是否返回distmap
        :param tensor: 如果True，那么返回图片的tensor，否则返回Image
        :return: 图像、像素坐标和距离向量
        """
        # 首先获取到图像和像素间距
        image, pixel_spacing = self.image, self.pixel_spacing
        # 判断图像大小是否非空
        if size is not None:
            # 调用前面定义好的函数对图像、像素间距和像素坐标矩阵进行resize
            image, pixel_spacing, pixel_coord = resize(size, image, pixel_spacing, pixel_coord)
        # 当最大旋转角大于0并且随机数小于旋转可能性时
        if max_angel > 0 and random.random() <= prob_rotate:
            # 随机取一个旋转角度
            angel = random.randint(-max_angel, max_angel)
            # 调用前面定义好的函数对图像和像素坐标进行旋转
            image, pixel_coord = rotate(image, pixel_coord, angel)
        # 判断tensor取值
        if tensor:
            # 返回图片的tensor
            image = tf.to_tensor(image)
        # 对像素坐标张量进行四舍五入，并将其转换为long类型
        pixel_coord = pixel_coord.round().long()
        # 判断distmap取值
        if distmap:
            # 调用定义好的函数生成distmap
            distmap = gen_distmap(image, pixel_spacing, pixel_coord)
            # 返回图像、像素坐标和距离向量
            return image, pixel_coord, distmap
        else:
            # 返回图像和像素坐标
            return image, pixel_coord
