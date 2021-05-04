from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from ..data_utils import gen_mask
from ..structure import Study


# 定义疾病数据集类
class DisDataSet(Dataset):
    def __init__(self,
                 studies: Dict[Any, Study],
                 annotations: Dict[Any, Tuple[torch.Tensor, torch.Tensor]],
                 prob_rotate: float,
                 max_angel: float,
                 num_rep: int,
                 sagittal_size: Tuple[int, int],
                 transverse_size: Tuple[int, int],
                 k_nearest: int,
                 max_dist: int,
                 sagittal_shift: int):
        # 设置Study
        self.studies = studies
        # 初始化标注数据
        self.annotations = []
        # 遍历标注数据
        for k, annotation in annotations.items():
            # 设置检查ID、序列ID和实例ID
            study_uid, series_uid, instance_uid = k
            # 判断当前检查ID是否在检查字典中，若不在则跳过该检查ID
            if study_uid not in self.studies:
                continue
            # 根据检查ID获取到当前一次检查
            study = self.studies[study_uid]
            # 判断当前序列ID否在当前一次检查中，若存在，继续判断当前实例ID否在当前序列中
            if series_uid in study and instance_uid in study[series_uid].instance_uids:
                # 若两个条件均满足，则将ID与对应的标注数据存入字典中
                self.annotations.append((k, annotation))
        # 设置旋转概率
        self.prob_rotate = prob_rotate
        # 设置最大旋转角度
        self.max_angel = max_angel
        # 设置重复次数
        self.num_rep = num_rep
        # 设置矢状面图像大小
        self.sagittal_size = sagittal_size
        # 设置轴状位图像大小
        self.transverse_size = transverse_size
        # 设置要选取的相邻图像数量
        self.k_nearest = k_nearest
        # 设置选取相邻图像的最大距离
        self.max_dist = max_dist
        # 设置矢状面偏移量
        self.sagittal_shift = sagittal_shift


    def __len__(self):
        # 返回标注数据条数 * 重复次数
        return len(self.annotations) * self.num_rep

    def __getitem__(self, item) -> (Study, Any, (torch.Tensor, torch.Tensor)):
        # 取模获得当前图像的index
        item = item % len(self.annotations)
        # 根据index获取到对应的标注数据（脊椎+椎间盘标注数据）
        key, (v_annotation, d_annotation) = self.annotations[item]
        # 返回获取到的标注数据
        return self.studies[key[0]], v_annotation, d_annotation

    # 整理数据
    def collate_fn(self, data: List[Tuple[Study, torch.Tensor, torch.Tensor]]) -> (Tuple[torch.Tensor], Tuple[None]):
        sagittal_images, transverse_images, vertebra_labels, disc_labels, distmaps = [], [], [], [], []
        v_masks, d_masks, t_masks = [], [], []
        # 遍历数据中的检查和脊椎+椎间盘标注数据
        for study, v_anno, d_anno in data:
            # 先构造mask
            v_mask = gen_mask(v_anno)
            d_mask = gen_mask(d_anno)
            # 将构造完毕的mask添加至对应的list中
            v_masks.append(v_mask)
            d_masks.append(d_mask)

            # 然后构造数据
            transverse_image, sagittal_image, distmap, pixel_coord, t_mask = study.transform(
                v_coords=v_anno[:, :2], d_coords=d_anno[:, :2], transverse_size=self.transverse_size,
                sagittal_size=self.sagittal_size, k_nearest=self.k_nearest, max_dist=self.max_dist,
                prob_rotate=self.prob_rotate, max_angel=self.max_angel, sagittal_shift=self.sagittal_shift
            )
            # 将构造完毕的数据添加至对应的list中
            sagittal_images.append(sagittal_image)
            distmaps.append(distmap)
            t_masks.append(t_mask)
            transverse_images.append(transverse_image)

            # 最后构造标签
            v_label = torch.cat([pixel_coord[:v_anno.shape[0]], v_anno[:, 2:]], dim=-1)
            d_label = torch.cat([pixel_coord[v_anno.shape[0]:], d_anno[:, 2:]], dim=-1)
            # 将构造完毕的标签添加至对应的list中
            vertebra_labels.append(v_label)
            disc_labels.append(d_label)

        # 将所有矢状面图像堆叠
        sagittal_images = torch.stack(sagittal_images, dim=0)
        # 将所有距离map堆叠
        distmaps = torch.stack(distmaps, dim=0)
        # 将所有轴状位图像堆叠
        transverse_images = torch.stack(transverse_images, dim=0)
        # 将所有脊椎标签堆叠
        vertebra_labels = torch.stack(vertebra_labels, dim=0)
        # 将所有椎间盘标签堆叠
        disc_labels = torch.stack(disc_labels, dim=0)
        # 将所有脊椎mask堆叠
        v_masks = torch.stack(v_masks, dim=0)
        # 将所有椎间盘mask堆叠
        d_masks = torch.stack(d_masks, dim=0)
        # 将所有轴状位图像mask堆叠
        t_masks = torch.stack(t_masks, dim=0)

        # 按照固定顺序构造数据
        data = (sagittal_images, transverse_images, distmaps, vertebra_labels, disc_labels, v_masks, d_masks, t_masks)
        label = (None, )
        # 返回构造好的数据以及标签
        return data, label

    # 生成样本
    def gen_sampler(self):
        # 初始化脊椎+椎间盘标注数据list
        v_annos, d_annos = [], []
        # 遍历所有标注数据中的key和对应的脊椎+椎间盘标注数据
        for key, (v_anno, d_anno) in self.annotations:
            # 将对应的数据添加到相应的list中
            v_annos.append(v_anno[:, -1])
            d_annos.append(d_anno[:, -1])
        # 将所有脊椎+椎间盘标注数据进行堆叠
        v_annos = torch.stack(v_annos, dim=0)
        d_annos = torch.stack(d_annos, dim=0)

        # 统计标注数据中每个独立元素的个数
        v_count = torch.unique(v_annos, return_counts=True)[1]
        d_count = torch.unique(d_annos, return_counts=True)[1]

        # 首先对独立元素个数的最后一维进行cumprod操作，获取到结果的最后一列数据后，将其除1，返回除法的浮点数结果而不作整数处理
        v_weights = torch.true_divide(1, torch.cumprod(v_count[v_annos], dim=-1)[:, -1])
        d_weights = torch.true_divide(1, torch.cumprod(d_count[d_annos], dim=-1)[:, -1])

        # 根据脊椎权重和椎间盘权重计算得到总权重
        weights = v_weights * d_weights

        # 使用计算好的权重进行加权随机采样，返回采样得到的数据
        return WeightedRandomSampler(weights=weights, num_samples=len(self), replacement=True)


# 重写了DataLoader
class DisDataLoader(DataLoader):
    def __init__(self, studies, annotations, batch_size, sagittal_size, transverse_size, k_nearest, prob_rotate=False,
                 max_angel=0, max_dist=8, sagittal_shift=0, num_workers=0,  num_rep=1, pin_memory=False,
                 sampling_strategy=None):
        # 断言采样策略为平衡采样或者None
        assert sampling_strategy in {'balance', None}
        # 获取疾病数据集对象
        dataset = DisDataSet(studies=studies, annotations=annotations, sagittal_size=sagittal_size,
                             transverse_size=transverse_size, k_nearest=k_nearest, prob_rotate=prob_rotate,
                             max_angel=max_angel, num_rep=num_rep, max_dist=max_dist, sagittal_shift=sagittal_shift)
        # 若为平衡采样
        if sampling_strategy == 'balance':
            # 生成样本
            sampler = dataset.gen_sampler()
            super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=dataset.collate_fn)
        # 若采样策略为None，则将数据随机打散
        else:
            super().__init__(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                             pin_memory=pin_memory, collate_fn=dataset.collate_fn)
