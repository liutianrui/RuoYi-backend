# -*- coding: utf-8 -*-
"""
@Author  : gpf
@License : (C) Copyright 2023
@Time    : 2023/04/18 13:06

"""
# %% 导入相关包
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

import importlib
from matplotlib import pyplot as plt

from manage import model_number

importlib.reload(plt)
from sklearn import tree
import pydotplus


def importData(data):
    Fault_diagnosis_data = pd.read_csv(data,
                                       usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                20,
                                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                                39,
                                                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                                                58,
                                                59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                                                77,
                                                78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                                                96,
                                                97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108])
    return Fault_diagnosis_data


def data_process(Fault_diagnosis_data, feature_mean, feature_median, feature_mode):
    length1 = len(feature_mean)
    i = 0
    while i < length1:
        Fault_diagnosis_data[feature_mean[i]].fillna(Fault_diagnosis_data[feature_mean[i]].mean(),
                                                     inplace=True)  # 使用该列平均值进行空缺值替换
        i += 1

    length2 = len(feature_median)
    j = 0
    while j < length2:
        Fault_diagnosis_data[feature_median[j]].fillna(Fault_diagnosis_data[feature_median[j]].median(), inplace=True)
        j += 1

    length3 = len(feature_mode)
    k = 0
    while k < length3:
        Fault_diagnosis_data[feature_mode[k]].fillna(Fault_diagnosis_data[feature_mode[k]].mode().iloc[0], inplace=True)
        k += 1


def load_data(Fault_diagnosis_data):
    # 载入特征和标签集
    X_test = Fault_diagnosis_data[
        ['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
         'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16',
         'feature17',
         'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25',
         'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33',
         'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41',
         'feature42', 'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49',
         'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57',
         'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65',
         'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73',
         'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature81',
         'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89',
         'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97',
         'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105',
         'feature106']]
    y_test = Fault_diagnosis_data['label']

    return X_test, y_test


def load_feature_data():
    feature_mean = ['feature0', 'feature2', 'feature4', 'feature7', 'feature8', 'feature11', 'feature15', 'feature22',
                    'feature23',
                    'feature27', 'feature28', 'feature30', 'feature38', 'feature41', 'feature42', 'feature43',
                    'feature48',
                    'feature49',
                    'feature51', 'feature56', 'feature58', 'feature63', 'feature70', 'feature71', 'feature75',
                    'feature79',
                    'feature81',
                    'feature82', 'feature84', 'feature85', 'feature86', 'feature95', 'feature96', 'feature99',
                    'feature102',
                    'feature106']
    feature_median = ['feature3', 'feature10', 'feature12', 'feature17', 'feature18', 'feature21', 'feature24',
                      'feature25',
                      'feature26',
                      'feature29', 'feature34', 'feature37', 'feature40', 'feature45', 'feature47', 'feature50',
                      'feature52', 'feature53',
                      'feature55', 'feature62', 'feature68', 'feature69', 'feature73', 'feature74', 'feature83',
                      'feature90', 'feature93',
                      'feature98', 'feature103', 'feature104']
    feature_mode = ['feature1', 'feature20', 'feature32', 'feature54', 'feature60', 'feature64', 'feature65',
                    'feature78',
                    'feature80',
                    'feature88', 'feature92']

    return feature_mean, feature_median, feature_mode


# 随机森林可视化
def tree_graph(rfc):
    target_names = ["0", "1", "2", "3", "4", "5"]
    feature_name = ['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7',
                    'feature8', 'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14',
                    'feature15', 'feature16', 'feature17', 'feature18', 'feature19', 'feature20', 'feature21',
                    'feature22', 'feature23', 'feature24', 'feature25', 'feature26', 'feature27', 'feature28',
                    'feature29', 'feature30', 'feature31', 'feature32', 'feature33', 'feature34', 'feature35',
                    'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41', 'feature42',
                    'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49',
                    'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56',
                    'feature57', 'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63',
                    'feature64', 'feature65', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70',
                    'feature71', 'feature72', 'feature73', 'feature74', 'feature75', 'feature76', 'feature77',
                    'feature78', 'feature79', 'feature80', 'feature81', 'feature82', 'feature83', 'feature84',
                    'feature85', 'feature86', 'feature87', 'feature88', 'feature89', 'feature90', 'feature91',
                    'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97', 'feature98',
                    'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105',
                    'feature106']
    Estimators = rfc.estimators_
    savePath = './media/treepdf/'
    for index, model in enumerate(Estimators):
        filename = 'tree_estimators_' + str(index) + '.pdf'
        dot_data = tree.export_graphviz(model, out_file=None,
                                        feature_names=feature_name,
                                        class_names=target_names,
                                        filled=True, rounded=True,
                                        special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        graph.write_pdf(savePath + filename)
    print('模型pdf生成完毕，生成路径为' + savePath)


def classify():
    data_train = "./preprocess_train.csv"  # 数据导入
    # data_train = "D:SoftCup\preprocess_train.csv"  # 数据导入
    # data_train = file  # 数据导入(可对前端上传的文件导入，功能暂未实现)

    # 数据读取
    Fault_diagnosis_data_train = importData(data_train)
    # Fault_diagnosis_data_test = importData(data_test)

    # 数据处理
    feature_mean, feature_median, feature_mode = load_feature_data()
    data_process(Fault_diagnosis_data_train, feature_mean, feature_median, feature_mode)

    # 载入特征和标签集
    X_rf_train, y_rf_train = load_data(Fault_diagnosis_data_train)
    # X_test,y_test = load_data(Fault_diagnosis_data_test)

    # 建立随机森林模型
    rfc = RandomForestClassifier(n_estimators=model_number, criterion='gini', max_depth=18, max_features=107,
                                 random_state=100)
    rfc.fit(X_rf_train, y_rf_train)  # 模型训练
    joblib.dump(rfc, "train_model.m")  # 存储模型
    # y_rf_pred = rfc.predict(X_rf_test)  # 使用随机森林（281个决策树分类器）多数投票产生的预测label列表

    print('模型训练完毕')

    # 随机森林可视化
    # tree_graph(rfc)

# nolabel部分（用于只生成json，不进行数据预测百分比分析）
def importData_nolabel(data):
    Fault_diagnosis_data = pd.read_csv(data,
                                       usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                20,
                                                21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                                                39,
                                                40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                                                58,
                                                59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76,
                                                77,
                                                78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
                                                96,
                                                97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107])
    return Fault_diagnosis_data

def load_data_nolabel(Fault_diagnosis_data):
    # 载入特征和标签集
    X_test = Fault_diagnosis_data[
        ['feature0', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', 'feature8',
         'feature9', 'feature10', 'feature11', 'feature12', 'feature13', 'feature14', 'feature15', 'feature16',
         'feature17',
         'feature18', 'feature19', 'feature20', 'feature21', 'feature22', 'feature23', 'feature24', 'feature25',
         'feature26', 'feature27', 'feature28', 'feature29', 'feature30', 'feature31', 'feature32', 'feature33',
         'feature34', 'feature35', 'feature36', 'feature37', 'feature38', 'feature39', 'feature40', 'feature41',
         'feature42', 'feature43', 'feature44', 'feature45', 'feature46', 'feature47', 'feature48', 'feature49',
         'feature50', 'feature51', 'feature52', 'feature53', 'feature54', 'feature55', 'feature56', 'feature57',
         'feature58', 'feature59', 'feature60', 'feature61', 'feature62', 'feature63', 'feature64', 'feature65',
         'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature71', 'feature72', 'feature73',
         'feature74', 'feature75', 'feature76', 'feature77', 'feature78', 'feature79', 'feature80', 'feature81',
         'feature82', 'feature83', 'feature84', 'feature85', 'feature86', 'feature87', 'feature88', 'feature89',
         'feature90', 'feature91', 'feature92', 'feature93', 'feature94', 'feature95', 'feature96', 'feature97',
         'feature98', 'feature99', 'feature100', 'feature101', 'feature102', 'feature103', 'feature104', 'feature105',
         'feature106']]
    # y_test = Fault_diagnosis_data['label']

    return X_test