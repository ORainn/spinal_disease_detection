import json
import math
from typing import Dict

import pandas as pd
from tqdm import tqdm

from .model import DiseaseModel
from ..data_utils import SPINAL_DISC_DISEASE_ID, SPINAL_VERTEBRA_DISEASE_ID
from ..structure import Study


# 求两个坐标的像素距离
def distance(coord0, coord1, pixel_spacing):
    x = (coord0[0] - coord1[0]) * pixel_spacing[0]
    y = (coord0[1] - coord1[1]) * pixel_spacing[1]
    output = math.sqrt(x ** 2 + y ** 2)
    return output


def format_annotation(annotations):
    """
    转换直接读取的annotation json文件的格式
    :param annotations: 直接读取的annotation json文件
    :return:
    """
    # 初始化字典
    output = {}
    # 遍历直接读取的annotation json文件
    for annotation in annotations:
        # 获取检查ID
        study_uid = annotation['studyUid']
        # 获取序列ID
        series_uid = annotation['data'][0]['seriesUid']
        # 获取实例ID
        instance_uid = annotation['data'][0]['instanceUid']
        # 临时存储标注信息的字典
        temp = {}
        # 遍历每个关键点
        for point in annotation['data'][0]['annotation'][0]['data']['point']:
            # 获取标识
            identification = point['tag']['identification']
            # 获取坐标
            coord = point['coord']
            # 若为椎间盘
            if 'disc' in point['tag']:
                # 设置对应的疾病
                disease = point['tag']['disc']
            # 若为脊椎
            else:
                # 设置对应的疾病
                disease = point['tag']['vertebra']
            # 若疾病为空
            if disease == '':
                disease = 'v1'
                # 将坐标和疾病传入
            temp[identification] = {
                'coord': coord,
                'disease': disease,
            }
        # 构造输出数据结构
        output[study_uid] = {
            'seriesUid': series_uid,
            'instanceUid': instance_uid,
            'annotation': temp
        }
    return output


# 评估类
class Evaluator:
    def __init__(self, module: DiseaseModel, studies: Dict[str, Study], annotation_path: str, metric='macro f1',
                 max_dist=6, epsilon=1e-5, num_rep=1):
        self.module = module
        self.studies = studies
        with open(annotation_path, 'r') as file:
            annotations = json.load(file)

        self.annotations = format_annotation(annotations)
        self.num_rep = num_rep
        self.metric = metric
        self.max_dist = max_dist
        # ε
        self.epsilon = epsilon

    def inference(self):
        # 停止模型训练
        self.module.eval()
        output = []
        # 遍历Study
        for study in self.studies.values():
            # 获取预测值
            pred = self.module(study, to_dict=True)
            output.append(pred)
        # 输出预测值
        return output

    def confusion_matrix(self, predictions) -> pd.DataFrame:
        """
        构建混淆矩阵
        :param predictions: 与提交格式完全相同
        :return:
        """
        # 构造椎间盘和脊椎疾病对应的列
        columns = ['disc_' + k for k in SPINAL_DISC_DISEASE_ID]
        columns += ['vertebra_' + k for k in SPINAL_VERTEBRA_DISEASE_ID]
        # 构建输出
        output = pd.DataFrame(self.epsilon, columns=columns, index=columns+['wrong', 'not_hit'])

        predictions = format_annotation(predictions)
        # 遍历标注数据
        for study_uid, annotation in self.annotations.items():
            study = self.studies[study_uid]
            # 获取中间帧像素间距
            pixel_spacing = study.t2_sagittal_middle_frame.pixel_spacing
            # 获取关键点标注数据
            pred_points = predictions[study_uid]['annotation']
            # 遍历标注数据
            for identification, gt_point in annotation['annotation'].items():
                # 坐标和疾病数据
                gt_coord = gt_point['coord']
                gt_disease = gt_point['disease']
                # 确定是椎间盘还是锥体
                if '-' in identification:
                    _type = 'disc_'
                else:
                    _type = 'vertebra_'
                # 遗漏的点记为fn
                if identification not in pred_points:
                    for d in gt_disease.split(','):
                        output.loc['not_hit', _type + d] += 1
                    continue
                # 根据距离判断tp还是fp
                pred_coord = pred_points[identification]['coord']
                pred_disease = pred_points[identification]['disease']
                if distance(gt_coord, pred_coord, pixel_spacing) >= self.max_dist:
                    for d in gt_disease.split(','):
                        output.loc['wrong', _type + d] += 1
                else:
                    for d in gt_disease.split(','):
                        output.loc[_type + pred_disease, _type + d] += 1
        # 返回混淆矩阵
        return output

    @staticmethod
    def cal_metrics(confusion_matrix: pd.DataFrame):
        # 计算关键点recall值
        key_point_recall = confusion_matrix.iloc[:-2].sum().sum() / confusion_matrix.sum().sum()
        precision = {col: confusion_matrix.loc[col, col] / confusion_matrix.loc[col].sum() for col in confusion_matrix}
        recall = {col: confusion_matrix.loc[col, col] / confusion_matrix[col].sum() for col in confusion_matrix}
        f1 = {col: 2 * precision[col] * recall[col] / (precision[col] + recall[col]) for col in confusion_matrix}
        macro_f1 = sum(f1.values()) / len(f1)

        # 只考虑预测正确的点
        columns = confusion_matrix.columns
        recall_true_point = {col: confusion_matrix.loc[col, col] / confusion_matrix.loc[columns, col].sum()
                             for col in confusion_matrix}
        f1_true_point = {col: 2 * precision[col] * recall_true_point[col] / (precision[col] + recall_true_point[col])
                         for col in confusion_matrix}
        macro_f1_true_point = sum(f1_true_point.values()) / len(f1)
        output = [('macro f1', macro_f1), ('key point recall', key_point_recall),
                  ('macro f1 (true point)', macro_f1_true_point)]
        output += sorted([(k+' f1 (true point)', v) for k, v in f1_true_point.items()], key=lambda x: x[0])
        # 返回预测正确的点的值
        return output

    def __call__(self, *args, **kwargs):
        confusion_matrix = None
        # 在设置的num_rep中训练，以进度条显示进度
        for _ in tqdm(range(self.num_rep), ascii=True):
            # 获取预测
            predictions = self.inference()
            if confusion_matrix is None:
                # 若混淆矩阵为None则生成混淆矩阵
                confusion_matrix = self.confusion_matrix(predictions)
            else:
                # 否则更新混淆矩阵
                confusion_matrix += self.confusion_matrix(predictions)
        # 计算混淆矩阵的值
        output = self.cal_metrics(confusion_matrix)

        i = 0
        # 找到'macro f1'列
        while i < len(output) and output[i][0] != self.metric:
            i += 1
        if i < len(output):
            # 将'macro f1'列放在最前面
            output = [output[i]] + output[:i] + output[i+1:]
        # 返回数据
        return output
