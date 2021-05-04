import sys
import time

import torch
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from code.core.disease import DisDataLoader, Evaluator, DiseaseModelV3, DisLoss
from code.core.key_point import SpinalModel, KeyPointModelV2, KeyPointBCELossV2, NullLoss, CascadeLossV2
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
        frame = study.t2_sagittal_middle_frame
        train_images[(study_uid, frame.series_uid, frame.instance_uid)] = frame.image

    # 设置网络骨架为ResNet50
    backbone = resnet_fpn_backbone('resnet50', True)
    # 建立脊柱模型
    spinal_model = SpinalModel(train_images, train_annotation,
                               num_candidates=128, num_selected_templates=8,
                               max_translation=0.05, scale_range=(0.9, 1.1), max_angel=10)
    # 建立关键点模型
    kp_model = KeyPointModelV2(backbone, pixel_mean=0.5, pixel_std=1,
                               loss=KeyPointBCELossV2(lamb=1), spinal_model=spinal_model,
                               cascade_loss=CascadeLossV2(1), loss_scaler=100, num_cascades=3)
    # kp_model.load_state_dict(torch.load('../models/2020070901.kp_model_v2'))
    # 加载第三次训练好的模型
    kp_model.load_state_dict(torch.load('../models/pretrained.dis_model_v3'))
    # 建立疾病分类模型
    dis_model = DiseaseModelV3(
        kp_model, sagittal_size=(512, 512), loss_scaler=0.01, use_kp_loss=True, share_backbone=True,
        transverse_size=(192, 192), sagittal_shift=1, k_nearest=1, transverse_only=True
    )

    # 放置到GPU
    dis_model.cuda(0)
    # 打印模型参数
    print(dis_model)

    # 设定训练参数
    train_dataloader = DisDataLoader(
        train_studies, train_annotation, batch_size=8, num_workers=0, num_rep=20, prob_rotate=1, max_angel=180,
        sagittal_size=dis_model.sagittal_size, transverse_size=dis_model.transverse_size, k_nearest=dis_model.k_nearest,
        sagittal_shift=dis_model.sagittal_shift, pin_memory=False, sampling_strategy=None
    )

    # 设定验证参数
    valid_evaluator = Evaluator(
        dis_model, valid_studies, '../data/lumbar_train51_annotation.json', num_rep=5, max_dist=6,
    )

    # 每个batch要训练的步骤
    step_per_batch = len(train_dataloader)
    # 设置Adam优化器，以及学习率
    optimizer = torch.optim.AdamW(dis_model.parameters(), lr=1e-5)
    # 设置最大训练轮次
    max_step = 40 * step_per_batch
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

    # 清除关键点网络
    dis_model.kp_model = None
    # 保存第三次训练模型
    torch.save(dis_model.cpu().state_dict(), '../models/pretrained.dis_model_v3')
    # 输出训练时长
    print('task completed, {} seconds used'.format(time.time() - start_time))
