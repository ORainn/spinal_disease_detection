import math
from copy import deepcopy
from typing import Tuple
import torch
import torchvision.transforms.functional as tf
from .loss import DisLoss
from ..structure import Study
from ..key_point import extract_point_feature, KeyPointModel
from ..data_utils import SPINAL_VERTEBRA_ID, SPINAL_VERTEBRA_DISEASE_ID, SPINAL_DISC_ID, SPINAL_DISC_DISEASE_ID


# 获取脊椎和椎间盘关键点与疾病的字典
VERTEBRA_POINT_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_ID.items()}
VERTEBRA_DISEASE_INT2STR = {v: k for k, v in SPINAL_VERTEBRA_DISEASE_ID.items()}
DISC_POINT_INT2STR = {v: k for k, v in SPINAL_DISC_ID.items()}
DISC_DISEASE_INT2STR = {v: k for k, v in SPINAL_DISC_DISEASE_ID.items()}


# 使用定义好的关键点模型作为疾病分类模型基础
class DiseaseModelBase(torch.nn.Module):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int]):
        super().__init__()
        # 设置矢状面图像的大小
        self.sagittal_size = sagittal_size
        # 设置脊椎疾病的数量
        self.num_vertebra_diseases = len(SPINAL_VERTEBRA_DISEASE_ID)
        # 设置椎间盘疾病的数量
        self.num_disc_diseases = len(SPINAL_DISC_DISEASE_ID)
        # 引入关键点模型
        self.backbone = deepcopy(kp_model)

    @property
    # 获取输出通道数
    def out_channels(self):
        return self.backbone.out_channels

    @property
    # 获取识别到的脊椎关键点数量
    def num_vertebra_points(self):
        return self.backbone.num_vertebra_points

    @property
    # 获取识别到的椎间盘关键点数量
    def num_disc_points(self):
        return self.backbone.num_disc_point

    @property
    # 获取识别到关键点参数
    def kp_parameters(self):
        return self.backbone.kp_parameters

    @property
    # 获取ResNet网络的输出通道数
    def resnet_out_channels(self):
        return self.backbone.resnet_out_channels

    @staticmethod
    def _gen_annotation(study: Study, vertebra_coords, vertebra_scores, disc_coords, disc_scores) -> dict:
        """
        生成标注数据
        :param study: 抽取的数据
        :param vertebra_coords: Nx2
        :param vertebra_scores: V
        :param disc_scores: Dx1
        :return:
        """
        # 根据T2矢状面的实例ID获取到index
        z_index = study.t2_sagittal.instance_uids[study.t2_sagittal_middle_frame.instance_uid]
        # 初始化关键点list
        point = []
        # 枚举每个脊椎的坐标以及关键点模型在该坐标的score
        for i, (coord, score) in enumerate(zip(vertebra_coords, vertebra_scores)):
            # 获取score最后一维上的最大值
            vertebra = int(torch.argmax(score, dim=-1).cpu())
            # 向关键点list中添加一个字典，记录了坐标、关键点名称、疾病名称以及index的信息
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': VERTEBRA_POINT_INT2STR[i],
                    'vertebra': VERTEBRA_DISEASE_INT2STR[vertebra]
                },
                'zIndex': z_index
            })
        # 枚举每个椎间盘的坐标以及关键点模型在该坐标的score
        for i, (coord, score) in enumerate(zip(disc_coords, disc_scores)):
            # 获取score最后一维上的最大值
            disc = int(torch.argmax(score, dim=-1).cpu())
            # 向关键点list中添加一个字典，记录了坐标、关键点名称、疾病名称以及index的信息
            point.append({
                'coord': coord.cpu().int().numpy().tolist(),
                'tag': {
                    'identification': DISC_POINT_INT2STR[i],
                    'disc': DISC_DISEASE_INT2STR[disc]
                },
                'zIndex': z_index
            })
        # 初始化当前检查对应的标注信息字典，字典内的信息有检查ID、实例ID、序列ID以及当前这张图像上所有关键点的标注信息
        annotation = {
            'studyUid': study.study_uid,
            'data': [
                {
                    'instanceUid': study.t2_sagittal_middle_frame.instance_uid,
                    'seriesUid': study.t2_sagittal_middle_frame.series_uid,
                    'annotation': [
                        {
                            'data': {
                                'point': point,
                            }
                        }
                    ]
                }
            ]
        }
        # 返回当前检查对应的标注信息字典
        return annotation

    def _train(self, sagittals, _, distmaps, v_labels, d_labels, v_masks, d_masks, t_masks) -> tuple:
        # 将脊椎和椎间盘的mask合并
        masks = torch.cat([v_masks, d_masks], dim=-1)
        # 返回关键点模型获取到的mask
        return self.backbone(sagittals, distmaps, masks)

    def _inference(self, study: Study, to_dict=False):
        # 获取当前检查的T2矢状面中间帧图像
        kp_frame = study.t2_sagittal_middle_frame
        # 将图片放缩到模型设定的大小
        sagittal = tf.resize(kp_frame.image, self.sagittal_size)
        # 将图像转为tensor类型
        sagittal = tf.to_tensor(sagittal).unsqueeze(0)

        # 通过关键点模型预测T2矢状面图像的脊椎坐标、椎间盘坐标以及特征
        v_coord, d_coord, _, feature_maps = self.backbone(sagittal, return_more=True)

        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / kp_frame.size[1]
        width_ratio = self.sagittal_size[1] / kp_frame.size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord.device)
        v_coord = (v_coord.float() / ratio).round()[0]
        d_coord = (d_coord.float() / ratio).round()[0]

        # 将脊椎score矩阵替换为列数与原来相同，行数为脊椎疾病数量的新矩阵
        v_score = torch.zeros(v_coord.shape[0], self.num_vertebra_diseases)
        # 将score矩阵第二列的元素设为1
        v_score[:, 1] = 1
        # 将椎间盘score矩阵替换为列数与原来相同，行数为椎间盘疾病数量的新矩阵
        d_score = torch.zeros(d_coord.shape[0], self.num_disc_diseases)
        # 将score矩阵第一列的元素设为1
        d_score[:, 0] = 1

        # 判断是否需要将数据转为字典形式
        if to_dict:
            # 调用函数将数据封装为字典并返回
            return self._gen_annotation(study, v_coord, v_score, d_coord, d_score)
        else:
            # 直接返回坐标与得分数据
            return v_coord, v_score, d_coord, d_score

    # 前向传播函数
    def forward(self, *args, **kwargs):
        # 判断是否要训练
        if self.training:
            # 返回关键点模型获取到的mask
            return self._train(*args, **kwargs)
        else:
            # 获取坐标与得分数据
            return self._inference(*args, **kwargs)

    # def crop(self, image, point):
    #     left, right = point[0] - self.crop_size, point[0] + self.crop_size
    #     top, bottom = point[1] - self.crop_size, point[1] + self.crop_size
    #     return image[:, top:bottom, left:right]
    #
    # def forward(self, sagittals, transverses, v_labels, d_labels, distmaps):
    #     v_patches = []
    #     for sagittal, v_label in zip(sagittals, v_labels):
    #         for point in v_label:
    #             patch = self.crop(sagittal, point)
    #             v_patches.append(patch)
    #
    #     d_patches = []
    #     for sagittal, d_label in zip(sagittals, d_labels):
    #         for point in d_label:
    #             patch = self.crop(sagittal, point)
    #             d_patches.append(patch)
    #     return v_patches, d_patches


# 疾病分类模型
class DiseaseModel(DiseaseModelBase):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 sagittal_shift: int = 0,
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 max_dist=6,
                 vertebra_score_scaler=None,
                 disc_score_scaler=None):
        # 初始化关键点识别模型，设置矢状面图像的大小
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size)
        # 判断是否共享backbone
        if share_backbone:
            self.kp_model = None
        else:
            # 为模型设置关键点模型
            self.kp_model = kp_model

        # 设置脊椎和椎间盘的线性层
        self.vertebra_head = torch.nn.Linear(self.out_channels, self.num_vertebra_diseases)
        self.disc_head = torch.nn.Linear(self.out_channels, self.num_disc_diseases)

        self.sagittal_shift = sagittal_shift

        self.use_kp_loss = use_kp_loss
        self.vertebra_loss = vertebra_loss
        self.disc_loss = disc_loss
        self.loss_scaler = loss_scaler
        self.max_dist = max_dist

        # 重新调整score以克服不平衡
        self.set_vetebra_score_scaler(vertebra_score_scaler)
        self.set_disc_score_scaler(disc_score_scaler)

        # 为了兼容性，实际上没有用
        self.k_nearest = 0
        self.transverse_size = self.sagittal_size

    # 调整脊椎score
    def set_vetebra_score_scaler(self, vertebra_score_scaler: list):
        if vertebra_score_scaler is None:
            vertebra_score_scaler = torch.ones(self.num_vertebra_diseases)
        else:
            vertebra_score_scaler = torch.tensor(vertebra_score_scaler)
        self.register_buffer('vertebra_score_scaler', vertebra_score_scaler)

    # 调整椎间盘score
    def set_disc_score_scaler(self, disc_score_scaler: list):
        if disc_score_scaler is None:
            # 没有score就直接初始化一个全为1的矩阵
            disc_score_scaler = torch.ones(self.num_disc_diseases)
        else:
            # 有score就将其转为tensor
            disc_score_scaler = torch.tensor(disc_score_scaler)
        # 向score的list添加一个缓冲区
        self.register_buffer('disc_score_scaler', disc_score_scaler)

    # 遍历脊椎和椎间盘的线性层并yield每个参数
    def disease_parameters(self, recurse=True):
        for p in self.vertebra_head.parameters(recurse):
            yield p
        for p in self.disc_head.parameters(recurse):
            yield p

    # 再次调整score
    def _rescale_score(self, vertebra_scores, disc_scores):
        # sigmoid激活函数
        vertebra_scores = vertebra_scores.sigmoid()
        disc_scores = disc_scores.sigmoid()
        # 调整score
        vertebra_scores = vertebra_scores * self.vertebra_score_scaler
        disc_scores = disc_scores * self.disc_score_scaler
        # 返回调整后的score
        return vertebra_scores, disc_scores

    # mask预测
    def _mask_pred(self, pred_coords, distmaps):
        # 获取预测到的宽、高数据，将其打平后，映射到[0, 最大距离-1]范围内
        width_indices = pred_coords[:, :, 0].flatten().clamp(0, distmaps.shape[-1]-1)
        height_indices = pred_coords[:, :, 1].flatten().clamp(0, distmaps.shape[-2]-1)

        # 获取图像和关键点的index
        image_indices = torch.arange(pred_coords.shape[0], device=pred_coords.device)
        image_indices = image_indices.unsqueeze(1).expand(-1, pred_coords.shape[1]).flatten()
        point_indices = torch.arange(pred_coords.shape[1], device=pred_coords.device).repeat(pred_coords.shape[0])

        # 过滤出满足条件的mask
        new_masks = distmaps[image_indices, point_indices, height_indices, width_indices] < self.max_dist
        # 变形
        new_masks = new_masks.reshape(pred_coords.shape[0], -1)
        # 返回新的mask
        return new_masks

    # 调整预测结果，也就是将mask反过来得到预测结果
    def _adjust_pred(self, pred_coords, distmaps, gt_coords):
        gt_coords = gt_coords.to(pred_coords.device)
        # 获取mask
        new_masks = self._mask_pred(pred_coords, distmaps)
        # 将mask所有值取反
        new_masks = torch.bitwise_not(new_masks)
        # 获取到预测的坐标
        pred_coords[new_masks] = gt_coords[new_masks]
        # 返回预测的坐标
        return pred_coords

    @staticmethod
    def _agg_features(d_point_feats, transverse, t_masks):
        """
        未来兼容，融合椎间盘矢状图和轴状图的特征
        :param d_point_feats:
        :param transverse:
        :param t_masks
        :return:
        """
        return d_point_feats

    # 训练
    def _train(self, sagittals, transverse, distmaps, v_labels, d_labels, v_masks, d_masks, t_masks) -> tuple:
        # 若使用关键点loss
        if self.use_kp_loss:
            # 获取mask
            masks = torch.cat([v_masks, d_masks], dim=-1)
            # 将distmap和mask传入backbone
            kp_loss, v_coords, d_coords, _, feature_maps = self.backbone(
                sagittals, distmaps, masks, return_more=True)
        else:
            # 只传入矢状面图像
            kp_loss, v_coords, d_coords, _, feature_maps = self.backbone(
                sagittals, None, None, return_more=True)

        # 返回关键点loss
        if self.loss_scaler <= 0:
            return kp_loss,

        # 决定使用哪种坐标训练
        if self.kp_model is not None:
            v_coords, d_coords = self.kp_model.eval()(sagittals)

        # 将错误的预测改为正确位置
        v_coords = self._adjust_pred(
            v_coords, distmaps[:, :self.num_vertebra_points], v_labels[:, :, :2]
        )
        d_coords = self._adjust_pred(
            d_coords, distmaps[:, self.num_vertebra_points:], d_coords[:, :, :2]
        )

        # 提取坐标点上的特征
        v_features = extract_point_feature(feature_maps, v_coords, *sagittals.shape[-2:])
        d_features = extract_point_feature(feature_maps, d_coords, *sagittals.shape[-2:])

        # 提取transverse特征
        d_features = self._agg_features(d_features, transverse, t_masks)

        # 计算scores
        v_scores = self.vertebra_head(v_features)
        d_scores = self.disc_head(d_features)

        # 计算损失
        v_loss = self.vertebra_loss(v_scores, v_labels[:, :, -1], v_masks)
        d_loss = self.disc_loss(d_scores, d_labels[:, :, -1], d_masks)

        # 堆叠损失并×loss_scaler
        loss = torch.stack([v_loss, d_loss]) * self.loss_scaler
        if kp_loss is None:
            # kp_loss为None，就只返回loss
            return loss,
        elif len(kp_loss.shape) > 0:
            # kp_loss形状>0，打平后与loss一并返回
            return torch.cat([kp_loss.flatten(), loss], dim=0),
        else:
            # kp_loss展开后与loss一并返回
            return torch.cat([kp_loss.unsqueeze(0), loss], dim=0),

    def _inference(self, study: Study, to_dict=False):
        # 获取中间帧参数
        middle_frame_size = study.t2_sagittal_middle_frame.size
        middle_frame_uid = study.t2_sagittal_middle_frame.instance_uid
        middle_frame_idx = study.t2_sagittal.instance_uids[middle_frame_uid]
        # 传入三张矢量图
        sagittal_dicoms = []
        for idx in range(middle_frame_idx - self.sagittal_shift, middle_frame_idx + self.sagittal_shift + 1):
            sagittal_dicoms.append(study.t2_sagittal[idx])

        sagittal_images = []
        # 遍历三张矢量图
        for dicom in sagittal_dicoms:
            # 将图片放缩到模型设定的大小
            image = tf.resize(dicom.image, self.sagittal_size)
            # 将图片转为tensor
            image = tf.to_tensor(image)
            # 放入list中
            sagittal_images.append(image)
        sagittal_images = torch.stack(sagittal_images, dim=0)

        if self.kp_model is not None:
            # 关键点模型存在，获取预测的坐标并计算得分
            v_coord, d_coord = self.kp_model(sagittal_images)
            _, feature_maps = self.backbone.cal_scores(sagittal_images)
        else:
            # 关键点模型不存在，调用backbone返回相应数据
            v_coord, d_coord, _, feature_maps = self.backbone(sagittal_images, return_more=True)

        # 修正坐标
        # 具体逻辑是先将多张矢状图上预测的点坐标都投影到中间帧上
        # 然后在中间帧上求中位数
        # 最后将中位数坐标分别投影回多张矢状图上，并以此提取点特征

        # 先将像素坐标转为人坐标系上的坐标
        v_coord_human = [dicom.pixel_coord2human_coord(coord) for dicom, coord in zip(sagittal_dicoms, v_coord)]
        d_coord_human = [dicom.pixel_coord2human_coord(coord) for dicom, coord in zip(sagittal_dicoms, d_coord)]
        # 将多张矢状图上预测的点坐标都投影到中间帧上
        v_coord = torch.stack([study.t2_sagittal_middle_frame.projection(coord) for coord in v_coord_human], dim=0)
        d_coord = torch.stack([study.t2_sagittal_middle_frame.projection(coord) for coord in d_coord_human], dim=0)
        # 在中间帧上求中位数
        v_coord_med = v_coord.median(dim=0)[0]
        d_coord_med = d_coord.median(dim=0)[0]
        # 获取到人坐标系上的中位数坐标
        v_coord_med_human = study.t2_sagittal_middle_frame.pixel_coord2human_coord(v_coord_med)
        d_coord_med_human = study.t2_sagittal_middle_frame.pixel_coord2human_coord(d_coord_med)
        # 将中位数坐标分别投影回多张矢状图上
        v_coord = torch.stack([dicom.projection(v_coord_med_human) for dicom in sagittal_dicoms], dim=0)
        d_coord = torch.stack([dicom.projection(d_coord_med_human) for dicom in sagittal_dicoms], dim=0)

        # 在三个feature_map上一起在提取点特征
        v_feature = extract_point_feature(feature_maps, v_coord, *self.sagittal_size)
        d_feature = extract_point_feature(feature_maps, d_coord, *self.sagittal_size)

        # 提取transverse特征
        transverse, t_mask = study.t2_transverse_k_nearest(
            d_coord_med.cpu(), k=self.k_nearest, size=self.transverse_size, max_dist=self.max_dist
        )
        # 矢状面图像的数量，这里 2 * 1 + 1 = 3
        num_sagittals = 2 * self.sagittal_shift + 1
        # 聚合特征
        d_feature = self._agg_features(
            d_feature,
            transverse.unsqueeze(0).expand(num_sagittals, -1, -1, 1, -1, -1),
            t_mask.unsqueeze(0).expand(num_sagittals, -1, -1)
        )

        # 计算分数，并取中位数
        v_score = self.vertebra_head(v_feature).median(dim=0)[0]
        d_score = self.disc_head(d_feature).median(dim=0)[0]
        v_score, d_score = self._rescale_score(v_score, d_score)

        # 将预测的坐标调整到原来的大小，注意要在extract_point_feature之后变换
        height_ratio = self.sagittal_size[0] / middle_frame_size[1]
        width_ratio = self.sagittal_size[1] / middle_frame_size[0]
        ratio = torch.tensor([width_ratio, height_ratio], device=v_coord_med.device)

        # 将坐标变回原来的大小
        v_coord_med = (v_coord_med.float() / ratio).round()
        d_coord_med = (d_coord_med.float() / ratio).round()

        if to_dict:
            # 转为字典后返回数据
            return self._gen_annotation(study, v_coord_med, v_score, d_coord_med, d_score)
        else:
            # 直接返回数据
            return v_coord_med, v_score, d_coord_med, d_score

# 疾病分类网络头
class DiseaseHead(torch.nn.Module):
    def __init__(self, in_channels, num_points, num_diseases):
        super().__init__()
        # 设置偏置和权重，并使用kaiming归一化
        self.bias = torch.nn.Parameter(torch.empty(num_points, in_channels))
        self.weights = torch.nn.Parameter(torch.empty(num_points, in_channels, in_channels))
        torch.nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        # ReLU激活后，归一化，最后一个线性层输出疾病分类
        self.relu = torch.nn.ReLU(inplace=True)
        self.norm = torch.nn.LayerNorm(in_channels)
        self.linear = torch.nn.Linear(in_channels, num_diseases)

    def forward(self, features):
        # 计算关键点数据
        point_wise = (features.unsqueeze(-1) * self.weights).sum(-1) + self.bias
        point_wise = self.norm(point_wise)
        features = features + point_wise
        features = self.relu(features)
        return self.linear(features)


# 疾病分类模型v2.0
class DiseaseModelV2(DiseaseModel):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 sagittal_shift: int = 0,
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 max_dist=6,
                 vertebra_score_scaler=None,
                 disc_score_scaler=None):
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size, sagittal_shift=sagittal_shift,
                         share_backbone=share_backbone, vertebra_loss=vertebra_loss, disc_loss=disc_loss,
                         loss_scaler=loss_scaler, use_kp_loss=use_kp_loss, max_dist=max_dist,
                         vertebra_score_scaler=vertebra_score_scaler, disc_score_scaler=disc_score_scaler)
        # 加上椎间盘和脊椎疾病分类网络头
        self.disc_head = DiseaseHead(self.out_channels, self.num_disc_points, self.num_disc_diseases)
        self.vertebra_head = DiseaseHead(self.out_channels, self.num_vertebra_points, self.num_vertebra_diseases)


# 疾病分类模型v3.0
class DiseaseModelV3(DiseaseModelV2):
    def __init__(self,
                 kp_model: KeyPointModel,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 sagittal_shift: int = 0,
                 share_backbone=False,
                 vertebra_loss=DisLoss([2.2727, 0.6410]),
                 disc_loss=DisLoss([0.4327, 0.7930, 0.8257, 6.4286, 16.3636]),
                 loss_scaler=1,
                 use_kp_loss=False,
                 max_dist=6,
                 vertebra_score_scaler=None,
                 disc_score_scaler=None,
                 k_nearest=0,
                 nhead=8,
                 transverse_only=False):
        super().__init__(kp_model=kp_model, sagittal_size=sagittal_size, sagittal_shift=sagittal_shift,
                         share_backbone=share_backbone, vertebra_loss=vertebra_loss, disc_loss=disc_loss,
                         loss_scaler=loss_scaler, use_kp_loss=use_kp_loss, max_dist=max_dist,
                         vertebra_score_scaler=vertebra_score_scaler, disc_score_scaler=disc_score_scaler)
        # 设置transverse参数
        self.transverse_size = transverse_size
        self.k_nearest = k_nearest
        self.transverse_only = transverse_only

        # 加入关键点模型
        self.backbone2 = deepcopy(kp_model)
        # transverse模块：平均池化->1*1卷积->归一化->RelU
        self.transverse_block = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            torch.nn.Conv2d(self.resnet_out_channels, self.out_channels, kernel_size=(1, 1)),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(inplace=True)
        )
        self.aggregation = torch.nn.TransformerEncoderLayer(self.out_channels, nhead=nhead)

    # 获取疾病参数
    def disease_parameters(self, recurse=True):
        for p in super(DiseaseModelV2, self).disease_parameters(recurse):
            yield p
        for p in self.transverse_block.parameters(recurse):
            yield p
        for p in self.aggregation.parameters(recurse):
            yield p

    def _agg_features(self, d_point_feats, transverses, t_masks):
        """
        融合椎间盘的矢状图和轴状图的特征
        :param d_point_feats: 椎间盘点特征，(num_batch, num_points, out_channels)
        :param transverses: (num_batch, num_points, k_nearest, 1, height, width)
        :param t_masks: (num_batch, num_points, k_nearest)，轴状图为padding的地方是True
        :return: (num_batch, num_points, out_channels)
        """
        # T当k nearest为0时，功能退化为V1
        if self.k_nearest <= 0:
            return d_point_feats

        # 计算轴状图的特征
        t_features = self.backbone2.cal_backbone(transverses.flatten(end_dim=2))
        t_features = self.transverse_block(t_features).reshape(*transverses.shape[:3], -1)
        t_masks = t_masks.to(t_features.device)

        # 判断是否仅计算轴状位图像特征和mask
        if self.transverse_only:
            all_features = t_features
            all_masks = torch.zeros(*t_masks.shape[:2], 1, device=t_masks.device, dtype=t_masks.dtype)
            all_masks = torch.cat([all_masks, t_masks[:, :, 1:]], dim=2)
        else:
            # 融合椎间盘的矢状图和轴状图的特征
            all_features = torch.cat([d_point_feats.unsqueeze(2), t_features], dim=2)
            # 矢状图的特征是全部都要用上的，所以d_masks全为False
            d_masks = torch.zeros(*t_masks.shape[:2], 1, device=t_masks.device, dtype=t_masks.dtype)
            all_masks = torch.cat([d_masks, t_masks], dim=2)

        # all_features: (batch_size, num_points, k_nearest, channels)
        # K = 1：
        if all_features.shape[2] == 1:
            final_features = all_features[:, :, 0]
        else:
            all_features = all_features.flatten(end_dim=1).permute(1, 0, 2)
            all_masks = all_masks.flatten(end_dim=1)
            print('haha')
            final_features = self.aggregation(all_features, src_key_padding_mask=all_masks)[0]
            final_features = final_features.reshape(*transverses.shape[:2], -1)
        return final_features


