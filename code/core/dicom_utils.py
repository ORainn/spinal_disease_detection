import json
import os
from collections import OrderedDict
import SimpleITK as sitk


def dicom_metainfo(dicm_path, list_tag):
    """
    获取dicom的元数据信息
    :param dicm_path: dicom文件地址
    :param list_tag: 标记名称列表,比如['0008|0018',]
    :return:
    """
    # 获取SimpleITK的图像文件读取器
    reader = sitk.ImageFileReader()
    # 加载私有的元信息
    reader.LoadPrivateTagsOn()
    # 设置读取器读取的路径
    reader.SetFileName(dicm_path)
    # 读取图像信息
    reader.ReadImageInformation()
    # 返回传入的list中要查找的tags对应的元数据信息
    return [reader.GetMetaData(t) for t in list_tag]


def dicom2array(dcm_path):
    """
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    :param dcm_path: dicom文件
    :return:
    """
    # 获取SimpleITK的图像文件读取器
    image_file_reader = sitk.ImageFileReader()
    # GDCMImageIO是一个读取和写入DICOM v3和ACR/NEMA图像的ImageIO类。在这里GDCMImageIO对象被创建并与ImageFileReader相连
    image_file_reader.SetImageIO('GDCMImageIO')
    # 设置读取器读取的路径
    image_file_reader.SetFileName(dcm_path)
    # 读取图像信息
    image_file_reader.ReadImageInformation()
    # 执行读取器读取dicom序列
    image = image_file_reader.Execute()
    # 判断图像每个像素是否只有一个元素：
    if image.GetNumberOfComponentsPerPixel() == 1:
        # 将图像的灰度值归一化到0-255
        image = sitk.RescaleIntensity(image, 0, 255)
        # 根据图像的光度计参数判断图像是否是灰度图
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            # 若图像为灰度图，则对图像灰度值进行反转
            image = sitk.InvertIntensity(image, maximum=255)
        # 将图像写为jpg格式，并且需要重新调整图像强度（默认为[0,255]），因为JPEG格式的显示需要转换为UInt8像素类型
        image = sitk.Cast(image, sitk.sitkUInt8)
    # 将SimpleITK对象转换为ndarray
    img_x = sitk.GetArrayFromImage(image)[0]
    # 返回灰度图(np.array)
    return img_x


# 读取记录着DICOM图像标签的json文件，将这些标签读入有序字典
with open(os.path.join(os.path.dirname(__file__), 'static_files/dicom_tag.json'), 'r') as file:
    DICOM_TAG = json.load(file, object_hook=OrderedDict)


# 调用上面的dicom_metainfo()方法，传入DICOM标签有序字典读取对应的图像元数据，并返回一一对应的有序字典
def dicom_metainfo_v2(dicm_path: str) -> dict:
    # 读取元数据
    metainfo = dicom_metainfo(dicm_path, DICOM_TAG.values())
    # 遍历DICOM标签有序字典和元数据返回有序字典
    return {k: v for k, v in zip(DICOM_TAG.keys(), metainfo)}


# 遍历有序字典DICOM_TAG中的标签，读取传入文件路径对应图片，对应标签的元数据，为元数据字典对应的标签下赋值，并返回遍历后的字典，以及错误信息
def dicom_metainfo_v3(dicom_path: str) -> (dict, str):
    metainfo = {}
    error_msg = ''
    # 遍历有序字典DICOM_TAG中的标签
    for k, v in DICOM_TAG.items():
        try:
            # 读取传入文件路径对应图片，对应标签的元数据
            temp = dicom_metainfo(dicom_path, [v])[0]

        except RuntimeError as e:
            temp = None
            error_msg += str(e)
        # 为元数据字典对应的标签下赋值
        metainfo[k] = temp
    # 返回遍历后的字典，以及错误信息
    return metainfo, error_msg


# 调用上面的dicom_metainfo_v2()方法和dicom2array()方法，传入图像路径，并返回存储图像元数据的有序字典以及转化后的灰度图
def read_one_dcm(dcm_path):
    return dicom_metainfo_v2(dcm_path), dicom2array(dcm_path)
