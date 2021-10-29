import numpy as np

from config import config
from XGBoost import XGBoost
from utils.dataFunctions import sigmod

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report


if __name__ == "__main__":

    # Load data
    cancer = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2)

    # 加载模型
    model = XGBoost()

    # 开始训练模型
    print("Begin training ...\n")
    model.fit(x_train, y_train)
    print("Training end.\n")

    # 测试模型
    y_pred = model.predict(x_test)

    if config["train"]["task_label"] == 1:
        y_pred = sigmod(y_pred)
        y_pred = np.round(y_pred)

    # 输出模型预测的结果
    print("\nXGBoost predict result: ")

    auc_score = roc_auc_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("auc_score: ", auc_score)
    print(report)
