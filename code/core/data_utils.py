import json
import math
import os
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm

from .dicom_utils import read_one_dcm

# 填充值
PADDING_VALUE: int = 0


def read_dcms(dcm_dir, error_msg=False) -> (Dict[Tuple[str, str, str], Image.Image], Dict[Tuple[str, str, str], dict]):
    """
    读取文件夹内的所有dcm文件
    :param dcm_dir: 待读取文件夹
    :param error_msg: 是否打印错误信息
    :return: 包含图像信息的字典，和包含元数据的字典
    """
    # 初始化路径数组
    dcm_paths = []
    # 遍历待读取文件夹路径
    for study in os.listdir(dcm_dir):
        # 获取子文件夹角绝对路径
        study_path = os.path.join(dcm_dir, study)
        # 遍历子文件夹中的图像
        for dcm_name in os.listdir(study_path):
            # 获取子文件夹中图像的绝对路径
            dcm_path = os.path.join(study_path, dcm_name)
            # 向路径数组中添加路径
            dcm_paths.append(dcm_path)

    # 使用with启动多进程
    with Pool(cpu_count()) as pool:
        # 存储异步结果
        async_results = []
        # 遍历数据路径数组
        for dcm_path in dcm_paths:
            # 异步非阻塞地向数组中加入读取到的元数据和灰度图等信息
            async_results.append(pool.apply_async(read_one_dcm, (dcm_path,)))

        # 存储图片和元数据
        images, metainfos = {}, {}
        # 遍历异步执行结果，通过进度条显示遍历进度
        for async_result in tqdm(async_results, ascii=True):
            # 等待所有数据读取完毕
            async_result.wait()
            try:
                # 获取元数据和图片
                metainfo, image = async_result.get()
            except RuntimeError as e:
                # 打印错误信息
                if error_msg:
                    print(e)
                continue
            # 从元数据中获取三个ID：检查UID、序列实例号UID（唯一标记不同序列的号码）、实例UID
            key = metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            # 从元数据中删除已获取的ID
            del metainfo['studyUid'], metainfo['seriesUid'], metainfo['instanceUid']
            # 将图片转为PIL Image，并存储至image数组中三个ID对应的位置
            images[key] = tf.to_pil_image(image)
            # 将其余元数据存入字典
            metainfos[key] = metainfo

    # 返回包含图像信息的字典，和包含元数据的字典
    return images, metainfos


def get_spacing(metainfos: Dict[Tuple[str, str, str], dict]) -> Dict[Tuple[str, str, str], torch.Tensor]:
    """
    从元数据中获取像素点间距的信息
    :param metainfos: 元数据
    :return: 像素点间距信息
    """
    output = {}
    # 遍历图像元数据信息
    for k, v in metainfos.items():
        # 获取像素点间距
        spacing = v['pixelSpacing']
        # 拆分像素点间距
        spacing = spacing.split('\\')
        # 将拆分后的字符串数组转为浮点数list
        spacing = list(map(float, spacing))
        # 在输出字典对应的key位置赋值为张量类型的间距信息
        output[k] = torch.tensor(spacing)
    # 返回像素点间距信息
    return output


# 读取记录着脊椎位置标注id的json文件，将这些标签读入字典
with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_vertebra_id.json'), 'r') as file:
    SPINAL_VERTEBRA_ID = json.load(file)

# 读取记录着椎间盘位置标注id的json文件，将这些标签读入字典
with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_disc_id.json'), 'r') as file:
    SPINAL_DISC_ID = json.load(file)

# 断言记录id的两个字典无交集
assert set(SPINAL_VERTEBRA_ID.keys()).isdisjoint(set(SPINAL_DISC_ID.keys()))

# 读取记录着脊椎疾病标注id的json文件，将这些标签读入字典
with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_vertebra_disease.json'), 'r') as file:
    SPINAL_VERTEBRA_DISEASE_ID = json.load(file)

# 读取记录着椎间盘疾病标注id的json文件，将这些标签读入字典
with open(os.path.join(os.path.dirname(__file__), 'static_files/spinal_disc_disease.json'), 'r') as file:
    SPINAL_DISC_DISEASE_ID = json.load(file)


def read_annotation(path) -> Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]]:
    """
    :param path: 标注文件路径
    :return: 字典的key是（studyUid，seriesUid，instance_uid）
             字典的value是两个矩阵，第一个矩阵对应锥体，第一个矩阵对应椎间盘
             矩阵每一行对应一个脊柱的位置，前两列是位置的坐标(横坐标, 纵坐标)，之后每一列对应一种疾病
             坐标为0代表缺失
             ！注意图片的坐标和tensor的坐标是转置关系的
    """
    # 根据输入的路径读取标注文件
    with open(path, 'r') as annotation_file:
        # non_hit_count用来统计为被编码的标记的数量，用于预警
        non_hit_count = {}
        # 存储标注信息的字典
        annotation = {}
        # 将json格式的标注文件加载并遍历
        for x in json.load(annotation_file):
            # 获取studyUid
            study_uid = x['studyUid']
            # 断言每条标注信息的'data'长度为1
            assert len(x['data']) == 1, (study_uid, len(x['data']))
            # 读取'data'数组的第一个标注数据
            data = x['data'][0]
            # 获取instanceUid
            instance_uid = data['instanceUid']
            # 获取seriesUid
            series_uid = data['seriesUid']
            # 断言每条标注数据的'annotation'长度为1
            assert len(data['annotation']) == 1, (study_uid, len(data['annotation']))
            # 获取data['annotation'][0]['data']['point']数据，即具体的坐标和标注信息
            points = data['annotation'][0]['data']['point']
            # 定义一个大小为脊椎位置标注id数量*3，数值大小为PADDING_VALUE的张量，记录脊椎标签数据
            vertebra_label = torch.full([len(SPINAL_VERTEBRA_ID), 3],
                                        PADDING_VALUE, dtype=torch.long)
            # 定义一个大小为椎间盘位置标注id数量*3，数值大小为PADDING_VALUE的张量，记录椎间盘标签数据
            disc_label = torch.full([len(SPINAL_DISC_ID), 3],
                                    PADDING_VALUE, dtype=torch.long)
            # 遍历图像中每个关键点具体的坐标和标注信息
            for point in points:
                # 读取位置标注
                identification = point['tag']['identification']
                # 判断位置标注是否在SPINAL_VERTEBRA_ID中
                if identification in SPINAL_VERTEBRA_ID:
                    # 读取位置标注信息对应的椎体ID
                    position = SPINAL_VERTEBRA_ID[identification]
                    # 读取疾病标注
                    diseases = point['tag']['vertebra']
                    # 在脊椎标签数据张量对应位置前两列填入坐标
                    vertebra_label[position, :2] = torch.tensor(point['coord'])
                    # 遍历当前图像标注信息中的疾病
                    for disease in diseases.split(','):
                        # 判断疾病标注是否在SPINAL_VERTEBRA_DISEASE_ID中
                        if disease in SPINAL_VERTEBRA_DISEASE_ID:
                            # 读取疾病标注信息对应的ID
                            disease = SPINAL_VERTEBRA_DISEASE_ID[disease]
                            # 在脊椎标签数据张量对应位置第三列填入疾病
                            vertebra_label[position, 2] = disease
                # 若位置标注在SPINAL_DISC_ID中
                elif identification in SPINAL_DISC_ID:
                    # 读取位置标注信息对应的椎间盘ID
                    position = SPINAL_DISC_ID[identification]
                    # 读取疾病标注
                    diseases = point['tag']['disc']
                    # 在椎间盘标签数据张量对应位置前两列填入坐标
                    disc_label[position, :2] = torch.tensor(point['coord'])
                    # 遍历当前图像标注信息中的疾病
                    for disease in diseases.split(','):
                        # 判断疾病标注是否在SPINAL_DISC_DISEASE_ID中
                        if disease in SPINAL_DISC_DISEASE_ID:
                            # 读取疾病标注信息对应的ID
                            disease = SPINAL_DISC_DISEASE_ID[disease]
                            # 在椎间盘标签数据张量对应位置第三列填入疾病
                            disc_label[position, 2] = disease
                # 判断当前位置标注是否已在non_hit_count中
                elif identification in non_hit_count:
                    # 若存在则自增1
                    non_hit_count[identification] += 1
                else:
                    # 若不存在则初始化对应标注位置数量为1
                    non_hit_count[identification] = 1
            # 向存储信息的字典中加入键值对，三个ID对应两个标注矩阵
            annotation[study_uid, series_uid, instance_uid] = vertebra_label, disc_label
    # 判断non_hit_count是否大于0，即是否有过标注记录
    if len(non_hit_count) > 0:
        print(non_hit_count)
    # 返回存储图像中所有关键点的标注字典
    return annotation


def resize(size: Tuple[int, int], image: Image.Image, spacing: torch.Tensor, *coords: torch.Tensor):
    """
    :param size: [height, width]，height对应纵坐标，width对应横坐标
    :param image: 图像
    :param spacing: 像素点间距
    :param coords: 标注是图像上的坐标，[[横坐标,纵坐标]]，横坐标从左到有，纵坐标从上到下
    :return: resize之后的image，spacing，annotation
    """
    # image.size是[width, height]
    # 高度缩放比例=缩放后高度/缩放前高度
    height_ratio = size[0] / image.size[1]
    # 宽度缩放比例=缩放后宽度/缩放前宽度
    width_ratio = size[1] / image.size[0]
    # 宽高缩放比例张量
    ratio = torch.tensor([width_ratio, height_ratio])
    # 像素点间距÷缩放比例
    spacing = spacing / ratio
    # 图像上的每个关键点坐标都×缩放比例
    coords = [coord * ratio for coord in coords]
    # 调用方法对图片进行缩放
    image = tf.resize(image, size)
    # 输出缩放后的图像、像素点间距以及标注信息
    output = [image, spacing] + coords
    return output


def rotate_point(points: torch.Tensor, angel, center: torch.Tensor) -> torch.Tensor:
    """
    将points绕着center顺时针旋转angel度
    :param points: size of（*， 2）
    :param angel: 旋转角度
    :param center: size of（2，）
    :return: 旋转后的关键点tensor
    """
    # 判断角度是否为0
    if angel == 0:
        # 为0则不旋转直接返回
        return points
    # 计算角度
    angel = angel * math.pi / 180
    while len(center.shape) < len(points.shape):
        # 在center的第一维前面再加一维使得center和points大小一致
        center = center.unsqueeze(0)
    # 求余弦值
    cos = math.cos(angel)
    # 求正弦值
    sin = math.sin(angel)
    # 计算偏置张量
    rotate_mat = torch.tensor([[cos, -sin], [sin, cos]], dtype=torch.float32, device=points.device)
    # 计算关键点距旋转中心点距离，得到关键点相对距离
    output = points - center
    # 计算高维矩阵乘法，相对距离×偏置张量
    output = torch.matmul(output, rotate_mat)
    # 再把旋转中心点坐标加回来得到关键点新的绝对坐标，并返回
    return output + center


def rotate_batch(points: torch.Tensor, angels: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    将一个batch的点，按照不同的角度和中心旋转
    :param points: (num_batch, num_points, 2)
    :param angels: (num_batch,)
    :param centers: (num_batch, 2)
    :return: 旋转后的关键点batch
    """
    # 为旋转中心tensor升维到与关键点tensor大小一致
    centers = centers.unsqueeze(1)
    # 计算关键点距旋转中心点距离，得到关键点相对距离
    output = points - centers
    # 计算角度
    angels = angels * math.pi / 180
    # 计算余弦值
    cos = angels.cos()
    # 计算正弦值
    sin = angels.sin()
    # 计算偏置张量
    rotate_mats = torch.stack([cos, sin, -sin, cos], dim=1).reshape(angels.shape[0], 1, 2, 2)
    # 在最后一个维度前增加一个维度，使大小与偏置张量保持一致
    output = output.unsqueeze(-1)
    # 计算高维矩阵乘法，相对距离×偏置张量
    output = output * rotate_mats
    # 将刚才升维的那一列求和
    output = output.sum(dim=-1)
    # 再把旋转中心点坐标加回来得到关键点新的绝对坐标，并返回
    return output + centers


def rotate(image: Image.Image, points: torch.Tensor, angel: int) -> (Image.Image, torch.Tensor):
    # 计算旋转中心
    center = torch.tensor(image.size, dtype=torch.float32) / 2
    # 调用函数返回旋转后的图像+图像张量
    return tf.rotate(image, angel), rotate_point(points, angel, center)


def gen_distmap(image: torch.Tensor, spacing: torch.Tensor, *gt_coords: torch.Tensor, angel=0):
    """
    先将每个像素点的坐标顺时针旋转angel之后，再计算到标注像素点的物理距离
    :param image: height * weight
    :param gt_coords: size of（*， 2）
    :param spacing: 关键点间距
    :param angel: 旋转角度
    :return:
    """
    # 过滤掉无穷大的图像
    coord = torch.where(image.squeeze() < np.inf)
    # 注意需要反转横纵坐标，反转后÷2得到中心点坐标
    center = torch.tensor([image.shape[2], image.shape[1]], dtype=torch.float32) / 2
    # 将坐标张量reshape
    coord = torch.stack(coord[::-1], dim=1).reshape(image.size(1), image.size(2), 2)
    # 调用函数旋转关键点得到旋转后的坐标张量
    coord = rotate_point(coord, angel, center)
    # 保存物理距离的张量
    dists = []
    # 遍历所有坐标
    for gt_coord in gt_coords:
        # 获得旋转后的坐标
        gt_coord = rotate_point(gt_coord, angel, center)
        dist = []
        # 遍历旋转后的坐标
        for point in gt_coord:
            # 记录物理距离
            dist.append((((coord - point) * spacing) ** 2).sum(dim=-1).sqrt())
        # 将这些距离连接在一起
        dist = torch.stack(dist, dim=0)
        # 存入张量
        dists.append(dist)
    # 判断是否只有一个坐标
    if len(dists) == 1:
        return dists[0]
    else:
        return dists


def gen_mask(coord: torch.Tensor):
    # 判断坐标张量中，每个坐标是否是原点，并返回bool类型张量
    return (coord.index_select(-1, torch.arange(2, device=coord.device)) != PADDING_VALUE).any(dim=-1)
