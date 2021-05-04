import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from code.core.disease import DisDataLoader, DiseaseModelBase, Evaluator
from code.core.key_point import KeyPointModel, NullLoss, SpinalModel, KeyPointBCELossV2
from code.core.structure import construct_studies

sys.path.append('core/nn_tools/')
from core.nn_tools import torch_utils


if __name__ == '__main__':
    # 获取开始训练时时间
    start_time = time.time()
    # 多进程加载训练数据、数据标注和数据计数
    train_studies, train_annotation, train_counter = construct_studies(
        '../data/lumbar_train150', '../data/lumbar_train150_annotation.json', multiprocessing=True)
    # 多进程加载测试数据、数据标注和数据计数
    valid_studies, valid_annotation, valid_counter = construct_studies(
        '../data/lumbar_train51/', '../data/lumbar_train51_annotation.json', multiprocessing=True)

    # 设定模型参数
    train_images = {}
    # 遍历训练数据中的检查ID和对应的检查
    for study_uid, study in train_studies.items():
        # 获取T2矢状面中间帧
        frame = study.t2_sagittal_middle_frame
        # 将该图像放入训练字典中对应的位置
        train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image

    # 设置网络骨架为ResNet34
    backbone = resnet_fpn_backbone('resnet34', True)
    # 建立脊柱模型
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    # 建立关键点模型
    kp_model = KeyPointModel(backbone, pixel_mean=0.5, pixel_std=1,
                             loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model)
    # 建立疾病分类模型
    dis_model = DiseaseModelBase(kp_model, sagittal_size=(512, 512))
    # 放置到GPU
    dis_model.cuda()
    # 打印模型参数
    print(dis_model)

    # 设定训练参数
    train_dataloader = DisDataLoader(
        train_studies, train_annotation, batch_size=8, num_workers=0, num_rep=20, prob_rotate=1, max_angel=180,
        sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.sagittal_size, k_nearest=0, max_dist=6,
        sagittal_shift=1, pin_memory=True
    )

    # 设定验证参数
    valid_evaluator = Evaluator(
        dis_model, valid_studies, '../data/lumbar_train51_annotation.json', num_rep=20, max_dist=6,
        metric='key point recall'
    )

    # 每个batch要训练的步骤
    step_per_batch = len(train_dataloader)
    # 设置Adam优化器，以及学习率
    optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
    # 设置最大训练轮次
    max_step = 30 * step_per_batch
    # 训练
    fit_result = torch_utils.fit(
        dis_model,
        train_data=train_dataloader,
        valid_data=None,
        optimizer=optimizer,
        max_step=max_step,
        loss=NullLoss(),
        metrics=[valid_evaluator.metric],
        is_higher_better=True,
        evaluate_per_steps=step_per_batch,
        evaluate_fn=valid_evaluator,
    )

    # 保存第一次训练模型
    torch.save(dis_model.backbone.cpu().state_dict(), '../models/pretrained_34.kp_model')
    # 输出训练时长
    print('task completed, {} seconds used'.format(time.time() - start_time))
