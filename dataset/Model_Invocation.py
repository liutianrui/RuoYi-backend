# 模型调用 调用之前保存的训练好的随机森林模型。
import json

import joblib
import numpy as np

from dataset.Rf_Classifier_Plus import importData, load_feature_data, importData_nolabel, load_data_nolabel
from dataset.Rf_Classifier_Plus import data_process
from dataset.Rf_Classifier_Plus import load_data


def test_set(filepath):
    data_test = filepath  # 测试数据导入
    fault_diagnosis_data_test = importData(data_test)  # 测试数据读取
    feature_mean, feature_median, feature_mode = load_feature_data()
    data_process(fault_diagnosis_data_test, feature_mean, feature_median, feature_mode)  # 测试数据预处理
    x_rf_test, y_rf_test = load_data(fault_diagnosis_data_test)  # 测试数据载入特征和标签集
    rfc = joblib.load("train_model.m")  # 调用此前训练好的模型
    y_rf_pred = rfc.predict(x_rf_test)  # 模型用于测试集

    length_test = len(y_rf_test)  # 500

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    item = 0

    a = 0
    # 创建列表存放每一个类别的预测准确率和召回率
    # accuracy_i = []
    precision_i = []
    recall_i = []
    # 循环向列表中写入数据
    while a < 6:
        while item < length_test:  # 500
            if a == y_rf_pred[item]:
                if a == y_rf_test.values[item]:
                    TP += 1
                else:
                    FP += 1
            else:
                if a == y_rf_test.values[item]:
                    FN += 1
                else:
                    TN += 1
            item += 1
        print("Label为%d的指标：TP:%d,TN:%d,FP:%d,FN:%d" % (a, TP, TN, FP, FN))
        # macro_accuracy.append((TP + TN) / (TP + TN + FP + FN))    # 准确率
        precision_i.append(TP / (TP + FP))  # 精确率
        recall_i.append(TP / (TP + FN))  # 召回率
        print("precision_i:", precision_i)
        print("recall_i:", recall_i)
        a += 1
        TP, FP, TN, FN, item = 0, 0, 0, 0, 0

    b = 0
    jtem = 0
    LabelSum_i = [0, 0, 0, 0, 0, 0]

    LABEL0 = 0
    LABEL1 = 0
    LABEL2 = 0
    LABEL3 = 0
    LABEL4 = 0
    LABEL5 = 0

    while jtem < length_test:
        if 0 == y_rf_test.values[jtem]:
            LABEL0 += 1
        elif 1 == y_rf_test.values[jtem]:
            LABEL1 += 1
        elif 2 == y_rf_test.values[jtem]:
            LABEL2 += 1
        elif 3 == y_rf_test.values[jtem]:
            LABEL3 += 1
        elif 4 == y_rf_test.values[jtem]:
            LABEL4 += 1
        else:
            LABEL5 += 1
        jtem += 1
    # print(LabelSum_i[b])  # 测试集中每一类的样本数量
    # b += 1

    # %%
    # 模型评价指标
    sum = 0
    sun = 0
    for p in precision_i:
        sum += p
    macro_P = sum / len(precision_i)
    print("macro_P:", macro_P)

    for r in recall_i:
        sun += r
    macro_R = sun / len(recall_i)
    print("macro_R:", macro_R)
    macro_F1 = (2 * macro_P * macro_R) / (macro_P + macro_R)
    print("macro_F1:", macro_F1)
    print("排行得分：", 100 * (2 * macro_P * macro_R) / (macro_P + macro_R))

    # %%模型评估
    print("Test set predictions: \n {}".format(y_rf_pred))
    print("Test_Set Score: {:.15f}".format(np.mean(y_rf_pred == y_rf_test)))  # np.mean函数输出两个矩阵/数组的相似程度

    # %%
    # LABEL0 = LabelSum_i[0]
    # LABEL1 = LabelSum_i[1]
    # LABEL2 = LabelSum_i[2]
    # LABEL3 = LabelSum_i[3]
    # LABEL4 = LabelSum_i[4]
    # LABEL5 = LabelSum_i[5]

    # 测试结果json文件输出
    dictionary = {}
    keys = []
    values = []

    for i in range(len(y_rf_test)):
        keys.append((str(y_rf_test.index[i])))
        values.append(int(y_rf_pred[i]))

    for key, value in zip(keys, values):
        dictionary[key] = value

    filename = f"./media/data/ClassifyResults.json"
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)

    return macro_P, macro_R, macro_F1, LABEL0, LABEL1, LABEL2, LABEL3, LABEL4, LABEL5, precision_i, recall_i, dictionary

def test_set_nolabel(filepath):
    data_test = filepath  # 测试数据导入
    fault_diagnosis_data_test = importData_nolabel(data_test)  # 测试数据读取
    feature_mean, feature_median, feature_mode = load_feature_data()
    data_process(fault_diagnosis_data_test, feature_mean, feature_median, feature_mode)  # 测试数据预处理
    x_rf_test = load_data_nolabel(fault_diagnosis_data_test)  # 测试数据载入特征和标签集
    rfc = joblib.load("train_model.m")  # 调用此前训练好的模型
    y_rf_pred = rfc.predict(x_rf_test)  # 模型用于测试集

    length_test = len(y_rf_pred)  # 500
    #
    # TP = 0
    # FP = 0
    # TN = 0
    # FN = 0
    # item = 0
    #
    # a = 0
    # 创建列表存放每一个类别的预测准确率和召回率
    # accuracy_i = []
    precision_i = []
    recall_i = []
    # 循环向列表中写入数据
    # while a < 6:
    #     while item < length_test:  # 500
    #         if a == y_rf_pred[item]:
    #             if a == y_rf_pred.values[item]:
    #                 TP += 1
    #             else:
    #                 FP += 1
    #         else:
    #             if a == y_rf_pred.values[item]:
    #                 FN += 1
    #             else:
    #                 TN += 1
    #         item += 1
    #     print("Label为%d的指标：TP:%d,TN:%d,FP:%d,FN:%d" % (a, TP, TN, FP, FN))
    #     # macro_accuracy.append((TP + TN) / (TP + TN + FP + FN))    # 准确率
    #     precision_i.append(TP / (TP + FP))  # 精确率
    #     recall_i.append(TP / (TP + FN))  # 召回率
    #     print("precision_i:", precision_i)
    #     print("recall_i:", recall_i)
    #     a += 1
    #     TP, FP, TN, FN, item = 0, 0, 0, 0, 0

    # b = 0
    # jtem = 0
    # LabelSum_i = [0, 0, 0, 0, 0, 0]
    #
    # LABEL0 = 0
    # LABEL1 = 0
    # LABEL2 = 0
    # LABEL3 = 0
    # LABEL4 = 0
    # LABEL5 = 0

    # while jtem < length_test:
    #     if 0 == y_rf_pred.values[jtem]:
    #         LABEL0 += 1
    #     elif 1 == y_rf_pred.values[jtem]:
    #         LABEL1 += 1
    #     elif 2 == y_rf_pred.values[jtem]:
    #         LABEL2 += 1
    #     elif 3 == y_rf_pred.values[jtem]:
    #         LABEL3 += 1
    #     elif 4 == y_rf_pred.values[jtem]:
    #         LABEL4 += 1
    #     else:
    #         LABEL5 += 1
    #     jtem += 1
    # print(LabelSum_i[b])  # 测试集中每一类的样本数量
    # b += 1

    # %%
    # 模型评价指标
    # sum = 0
    # sun = 0
    # for p in precision_i:
    #     sum += p
    # macro_P = sum / len(precision_i)
    # print("macro_P:", macro_P)
    #
    # for r in recall_i:
    #     sun += r
    # macro_R = sun / len(recall_i)
    # print("macro_R:", macro_R)
    # macro_F1 = (2 * macro_P * macro_R) / (macro_P + macro_R)
    # print("macro_F1:", macro_F1)
    # print("排行得分：", 100 * (2 * macro_P * macro_R) / (macro_P + macro_R))
    #
    # # %%模型评估
    # print("Test set predictions: \n {}".format(y_rf_pred))
    # print("Test_Set Score: {:.15f}".format(np.mean(y_rf_pred == y_rf_pred)))  # np.mean函数输出两个矩阵/数组的相似程度

    # %%
    # LABEL0 = LabelSum_i[0]
    # LABEL1 = LabelSum_i[1]
    # LABEL2 = LabelSum_i[2]
    # LABEL3 = LabelSum_i[3]
    # LABEL4 = LabelSum_i[4]
    # LABEL5 = LabelSum_i[5]

    # 测试结果json文件输出
    dictionary = {}
    keys = []
    values = []

    for i in range(len(y_rf_pred)):
        keys.append((str(i)))
        values.append(int(y_rf_pred[i]))

    for key, value in zip(keys, values):
        dictionary[key] = value

    filename = f"./media/data/ClassifyResults.json"
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file, indent=4)

    return 99, 99, 9, 1, 1, 1, 1, 1, 1, precision_i, recall_i, dictionary