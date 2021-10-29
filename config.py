
config = {}

config["train"] = {
    
    "learning_rate": 0.5,  # Step size for weight update.

    "n_estimators": 1,  # 单个类需要训练的树的数目
    "task_label": 1,  # 任务区分; 0表示回归任务，1表示二分类任务，2表示多分类任务
    "n_class": 1,  # 多分类任务中类别数目，默认为1
    
    "min_samples_split": 2,  # 树节点中分裂所需最小样本数
    "min_gain_split": 1e-7,  # 树节点分裂所需最小收益
    "max_depth": 3,  # 树的最大深度

    "eps": 0.3,  # 用于分桶近似的超参数，eps越小近似的精度最高；训练时间越久

    # Regularization parameter.
    "lambta": 0.001,
    "gamma": 0.001,

    'processNum': 6  # 多进程优化的参数，最大为电脑核心数目
}
