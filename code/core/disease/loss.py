import torch


# 定义疾病分类诊断模型用到的损失函数
class DisLoss:
    def __init__(self, weight: list = None):
        # 判断权重是否为空
        if weight is not None:
            # 将权重转为张量
            weight = torch.tensor(weight)
        # 设置交叉熵损失函数
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, pred, target, mask):
        # 根据模型预测值计算loss
        self.loss.to(pred.device)
        # 将目标分类的device与预测值统一
        target = target.to(device=pred.device)
        # 获取到预测值的mask
        pred = pred[mask]
        # 获取到目标值的mask
        target = target[mask]
        # 返回计算好的loss
        return self.loss(pred, target)
