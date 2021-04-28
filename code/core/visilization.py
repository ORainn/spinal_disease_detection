from typing import Union
import torch
import torchvision.transforms.functional as tf
from PIL import Image


# 将输入的图像转换为张量
def to_tensor(image):
    # 判断输入的图像是不是PIL.Image格式
    if isinstance(image, Image.Image):
        # 转为张量
        return tf.to_tensor(image)
    else:
        # 深拷贝输入图像并返回一个新的张量，不参与梯度计算
        return image.clone().detach()


def visilize_coord(image: Union[Image.Image, torch.Tensor], *coords: torch.Tensor, _range=10) -> Image.Image:
    """
    显示坐标标注信息
    关于annotation的结构请参考read_annotation
    :param image:
    :param coords:
    :param _range:
    :return:
    """
    # 先将图像转换为张量
    image = to_tensor(image)
    # 遍历所有坐标
    for coord in coords:
        # 遍历所有点
        for point in coord:
            # 注意，image和tensor存在转置关系
            # 将关键点附近10个像素组成的正方形空间全部设为0
            image[0, int(point[1]-_range):int(point[1]+_range), int(point[0]-_range):int(point[0]+_range)] = 0
    # 将处理好的图像转为PIL Image并返回
    return tf.to_pil_image(image)


def visilize_distmap(image: Union[Image.Image, torch.Tensor], *distmaps: torch.Tensor, max_dist=8) -> Image.Image:
    """
    显示物理距离信息
    关于label的结构请参考gen_label
    :param image:
    :param distmaps:
    :param max_dist:
    :return:
    """
    # 先将图像转换为张量
    image = to_tensor(image)
    # 遍历所有物理距离
    for distmap in distmaps:
        # 将物理距离小于阈值的位置设为0
        image[(distmap < max_dist).sum(dim=0).bool().unsqueeze(0)] = 0
    # 将处理好的图像转为PIL Image并返回
    return tf.to_pil_image(image)


def visilize_annotation(image, *annotations, _range=10):
    """
    显示标注文件中的信息
    :param image:
    :param annotations:
    :param _range:
    :return:
    """
    # 先将图像转换为张量
    image = to_tensor(image)
    # 遍历所有标注数据
    for annotation in annotations:
        # 遍历每条标注数据中的所有点
        for point in annotation['data'][0]['annotation'][0]['data']['point']:
            # 获取标注数据中每个点的坐标
            coord = point['coord']
            # 注意，image和tensor存在转置关系
            # 将关键点附近10个像素组成的正方形空间全部设为0
            image[0, int(coord[1]-_range):int(coord[1]+_range), int(coord[0]-_range):int(coord[0]+_range)] = 0
    # 将处理好的图像转为PIL Image并返回
    return tf.to_pil_image(image)
