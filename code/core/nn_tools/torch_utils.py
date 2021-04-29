import math
import os
import pickle
import time
import traceback
from copy import deepcopy
from typing import List

import torch

from .utils import tqdm


# 迭代函数，一直迭代
def forever_iter(iterable):
    while True:
        for _ in iterable:
            yield _


# 对模型进行评估
def evaluate(module: torch.nn.Module, data, metrics: list):
    # 不计算梯度
    with torch.no_grad():
        # 进入评估模式，将本层及子层的training设定为False
        module.eval()
        # 获取loss
        loss_value = [[] for _ in metrics]
        # 遍历数据、标签，以进度条显示进度
        for data, label in tqdm(data, ascii=True):
            # 获取模型预测结果
            prediction = module(*data)
            if prediction is None:
                continue
            # 遍历loss和评价指标
            for a, b in zip(loss_value, metrics):
                a.append(b(*prediction, *label))
        # 将loss转为tensor
        loss_value = [torch.tensor([x for x in array if x is not None], device=data[0].device).mean()
                      for array in loss_value]
    # 返回评价指标及其loss
    return [(type(m).__name__, v) for m, v in zip(metrics, loss_value)]


# 训练函数
def fit(module: torch.nn.Module, train_data, valid_data, optimizer, max_step, loss, metrics: list, is_higher_better,
        evaluate_per_steps=None, early_stopping=-1, scheduler=None, init_metric_value=None, evaluate_fn=evaluate,
        checkpoint_dir=None):
    if checkpoint_dir is not None:
        # 设置检查点
        checkpoint_dir = os.path.join(checkpoint_dir, str(time.time()))
        os.mkdir(checkpoint_dir)
    # 状态变量
    print('using {} as training loss, using {}({} is better) as early stopping metric'.format(
        type(loss).__name__,
        metrics[0] if isinstance(metrics[0], str) else type(metrics[0]).__name__,
        'higher' if is_higher_better else 'lower'))
    # 每训练多少轮评估一次，更新或者到达最大轮次自动停止
    evaluate_per_steps = evaluate_per_steps or max_step

    # 最佳状态字典
    best_state_dict = deepcopy(module.state_dict())
    # 初始化最佳轮次
    best_step = -1
    # 初始化最佳评价指标
    best_metric_value = init_metric_value
    # 记录loss
    loss_record = []
    # 初始化训练轮次
    step = 0
    # 遍历训练数据
    generator = forever_iter(train_data)
    try:
        while step < max_step:
            # 休息
            time.sleep(0.5)
            # 训练！
            module.train(True)
            # 开始一次评估前的训练，通过进度条表示进度
            for _ in tqdm(range(evaluate_per_steps), ascii=True):
                # 训练轮次++
                step += 1
                # --------- 训练参数 ------------
                # 获取数据和标签
                data, label = next(generator)
                # 优化器梯度归零
                optimizer.zero_grad()
                # 获取模型预测结果
                prediction = module(*data)
                if prediction is None:
                    continue
                # 计算loss
                loss_value = loss(*prediction, *label)
                # 记录loss
                loss_record.append(float(loss_value.detach()))
                if loss_value is not None:
                    # 反向传播loss
                    loss_value.backward()
                    # 优化器
                    optimizer.step()
                if scheduler:
                    scheduler.step()
            if checkpoint_dir is not None:
                # 在检查点保存
                torch.save(module, os.path.join(checkpoint_dir, '{}.checkpoint'.format(step)))
            # ----- 计算校验集的loss和metric
            metrics_values = evaluate_fn(module, valid_data, metrics)
            metric_value = metrics_values[0][1]
            # 当有新的最佳指标出现
            if (best_metric_value is None
                    or metric_value == best_metric_value
                    or is_higher_better == (metric_value > best_metric_value)):
                # 更新最佳状态字典
                best_state_dict = deepcopy(module.state_dict())
                # 更新最佳状态对应的训练轮次
                best_step = step
                # 更新评价指标
                best_metric_value = metric_value

            with torch.no_grad():
                # 输出训练集上的loss
                print('step {} lumbar_train51 {}: {}'.format(
                    step, type(loss).__name__, torch.tensor(
                        [x for x in loss_record[-evaluate_per_steps:] if x is not None]).mean()))
            for a, b in metrics_values:
                # 输出验证集上的loss
                print('valid {}: {}'.format(a, b))
            # 提前停止的策略
            if step - best_step >= early_stopping > 0:
                break
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        # 将模型参数设置为最佳参数
        module.load_state_dict(best_state_dict)
    # 返回最佳评价指标以及loss的记录
    return best_metric_value, loss_record


class NumericEmbedding(torch.nn.Module):
    """
    参考torch.nn.Embedding文档，将数值型特征也变成相同的形状。
    实际上就是将输入的张量扩展一个为1的维度之后，加上一个没有常数项的全连接层
    """
    def __init__(self, input_dim: List[int], emb_dim):
        super(NumericEmbedding, self).__init__()
        # 将输入的维度进行扩张
        size = [1] * (len(input_dim) + 1) + [emb_dim]
        # 倒数第二次就是输入层的最后一层
        size[-2] = input_dim[-1]
        # 初始化权重矩阵
        self.weight = torch.nn.Parameter(torch.empty(size))
        # 使用kaiming分布，激活函数（LeakyReLU）的负斜率为根号5
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 在最后一层加一个维度
        output = torch.unsqueeze(inputs, -1)
        # 乘个权重，相当于全连接层
        output = output * self.weight
        # 返回全连接层输出
        return output
