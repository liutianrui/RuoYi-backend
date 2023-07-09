print("正在处理中，请稍候...")
import json
import sys
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from tqdm import tqdm
import os
import pandas as pd
import warnings
pause_duration = 1 # 停顿时间（单位：秒）
print("该可执行文件采用数据集均为官方提供的:preprocess_train.csv、validate_1000.csv、test_2000_x.csv")
time.sleep(pause_duration)
print("获取脚本文件的所在路径，并根据该路径构建CSV文件的完整路径...")
time.sleep(pause_duration)
sys.stdout.flush()

base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "resources", "preprocess_train.csv")
dataset_vali_path = os.path.join(base_path, "resources", "validate_1000.csv")
dataset_test_path = os.path.join(base_path, "resources", "test_2000_x.csv")

warnings.filterwarnings("ignore")

print("读取CSV格式的数据文件...")
time.sleep(pause_duration)

sys.stdout.flush()

data = pd.read_csv(dataset_path)
data_vali = pd.read_csv(dataset_vali_path)
data_test = pd.read_csv(dataset_test_path)



feat_mean = ['feature0', 'feature2', 'feature4', 'feature7', 'feature8', 'feature11', 'feature15', 'feature22',
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
feat_median = ['feature3', 'feature10', 'feature12', 'feature17', 'feature18', 'feature21', 'feature24',
                      'feature25',
                      'feature26',
                      'feature29', 'feature34', 'feature37', 'feature40', 'feature45', 'feature47', 'feature50',
                      'feature52', 'feature53',
                      'feature55', 'feature62', 'feature68', 'feature69', 'feature73', 'feature74', 'feature83',
                      'feature90', 'feature93',
                      'feature98', 'feature103', 'feature104']
feat_mode = ['feature1', 'feature20', 'feature32', 'feature54', 'feature60', 'feature64', 'feature65',
                    'feature78',
                    'feature80',
                    'feature88', 'feature92']

# 缺失值填充函数
def data_process(data,feat_mean,feat_median,feat_mode):
    i,j,k = 0,0,0
    while i < len(feat_mean):
        data[feat_mean[i]].fillna(data[feat_mean[i]].mean(),inplace=True)
        i += 1
    while j < len(feat_median):
        data[feat_median[j]].fillna(data[feat_median[j]].median(), inplace=True)
        j += 1
    while k < len(feat_mode):
        data[feat_mode[k]].fillna(data[feat_mode[k]].mode().iloc[0], inplace=True)
        k += 1


# 是否进行数据预处理
train_input = input("是否进行数据集预处理？(y/n): ")
if train_input.lower() == 'y':
    # 执行数据预处理代码
    data_process(data, feat_mean, feat_median, feat_mode)
    data_process(data_vali, feat_mean, feat_median, feat_mode)
    data_process(data_test,feat_mean,feat_median,feat_mode)

    # 数据清洗  将data_vali中给定的特征（fm）中非众数值的样本数据都从数据集中删除，最终得到处理后的测试集（data_vali）
    fm = [2, 21, 33, 55, 61, 65, 66, 79, 81, 89, 93]
    for f in fm:
        feature_f = "feature" + str(f - 1)
        feature_f_mode = data_vali[feature_f].mode().values[0]
        data_vali = data_vali[data_vali[feature_f] == feature_f_mode]

    # 载入特征和标签集
    feat_list = []
    for i in range(107):
        if (i == 57) | (i == 77) | (i == 100):  # 不载入全为0的特征
            continue
        else:
            feat_list.append('feature' + str(i))

    X = data[feat_list]
    y = data['label']

    X_vali = data_vali[feat_list]
    y_vali = data_vali['label']

    X_test = data_test[feat_list]

    time.sleep(pause_duration)
    print("数据集预处理完毕！")
elif train_input.lower() == 'n':
    # 退出程序
    print("程序已退出。")
    exit()


# 是否进行模型训练
train_input = input("是否进行RandomForest模型训练？(y/n): ")
if train_input.lower() == 'y':
    print("RF模型训练中，请稍候...", flush=True)
    X_train, X_vali, y_train, y_vali = X, X_vali, y, y_vali
    # 随机森林分类器
    rfc = RandomForestClassifier(n_estimators=281, criterion='gini', max_depth=None, min_samples_split=2,
                                 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto',
                                 max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0,
                                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                                 warm_start=False, class_weight=None)
    rfc.fit(X_train, y_train) #训练模型

    # 显示进度条
    # show_progress = True
    # if show_progress:
    #     with tqdm(total=X_train.shape[0]) as pbar:
    #         for _ in range(10):
    #             rfc.estimators_ = []  # 清空已经训练好的树
    #             rfc.fit(X_train, y_train)  # 重新拟合
    #             pbar.update(X_train.shape[0] // 10)

    y_pred = rfc.predict(X_vali)
    print("RF模型训练完毕！")
elif train_input.lower() == 'n':
    # 退出程序
    print("程序已退出。")
    exit()

# 是否进行模型检验
evaluate_input = input("是否进行模型检验，获取验证集的性能评价指标？(y/n): ")
if evaluate_input.lower() == 'y':
    # 执行模型检验代码，打印准确率、召回率和F-score等指标
    labels = ["0", "1", "2", "3", "4", "5"]
    time.sleep(pause_duration)
    print(classification_report(y_vali, y_pred, target_names=labels))
    time.sleep(pause_duration)

    length_test = len(X_vali)
    TP, FP, TN, FN, item, a = 0, 0, 0, 0, 0, 0
    precision_i = []
    recall_i = []
    while a < len(labels):
        while item < length_test:
            if a == y_pred[item]:
                if a == y_vali.values[item]:
                    TP += 1
                else:
                    FP += 1
            else:
                if a == y_vali.values[item]:
                    FN += 1
                else:
                    TN += 1
            item += 1
        print("label_%d：TP:%d,TN:%d,FP:%d,FN:%d" % (a, TP, TN, FP, FN))
        time.sleep(pause_duration)

        precision_i.append((TP+TN) / (TP+FP+TN+FN))  # 精确率
        recall_i.append(TP / (TP + FN))  # 召回率
        a += 1
        TP, FP, TN, FN, item = 0, 0, 0, 0, 0
    print("每一类的预测准确率:", precision_i)
    time.sleep(pause_duration)

    print("每一类的召回率:", recall_i)
    time.sleep(pause_duration)

    # 模型评价指标
    sum, sun = 0, 0
    for p in precision_i:
        sum += p
    macro_P = sum / len(precision_i)
    for r in recall_i:
        sun += r
    macro_R = sun / len(recall_i)
    print("平均预测准确率macro_P:", macro_P)
    time.sleep(pause_duration)

    print("平均召回率macro_R:", macro_R)
    time.sleep(pause_duration)

    print("macro_F1:", (2 * macro_P * macro_R) / (macro_P + macro_R))
    time.sleep(pause_duration)

    print("F-score：", 100 * (2 * macro_P * macro_R) / (macro_P + macro_R))
    time.sleep(pause_duration)

    print("模型检验完毕！")
elif evaluate_input.lower() == 'n':
    # 退出程序
    print("程序已退出。")
    exit()

# 是否进行类别预测
predict_input = input("是否对测试集进行故障类别预测？(y/n): ")
if predict_input.lower() == 'y':
    # 执行类别预测代码
    # 获取可执行文件所在目录的路径
    exe_dir = os.path.dirname(sys.executable)

    # 创建保存结果的文件夹（如果不存在的话）
    data_dir = os.path.join(exe_dir, 'Submit')
    os.makedirs(data_dir, exist_ok=True)

    # 构造结果文件的完整路径
    filename = os.path.join(data_dir, 'submit.json')

    y_test = rfc.predict(X_test)

    print("PREDICTING...")
    time.sleep(1)

    # 测试结果json文件输出
    dictionary = {}
    keys = []
    values = []
    sample_id = data_test['sample_id']
    for i in range(len(sample_id)):
        # keys.append((str(y_rf_test.index[i])))
        keys.append(str(sample_id.index[i]))
        values.append(int(y_test[i]))

    for key, value in zip(keys, values):
        dictionary[key] = value

    # filename = f"./media/data/ClassifyResults.json"
    with open(filename, 'w') as json_file:
        json.dump(dictionary, json_file)

    # 读取JSON文件
    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    # 打印JSON内容（格式化输出）
    formatted_data = json.dumps(data)
    print(formatted_data)

    print("测试集故障类别预测完毕！！ 预测结果JSON文件已生成，请查看当前目录下的Submit文件夹下的submit.json文件。")
    input("按任意键关闭窗口...")
elif predict_input.lower() == 'n':
    # 退出程序
    print("程序已退出。")
    exit()