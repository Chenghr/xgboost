import numpy as np
import multiprocessing as mp

from config import config

def findBestSplit_mp(X_gh, block, sample_tag):
    """基于预排序以及分桶近似的思想，在拥有的sample中选择出近似的分裂点

    Returns:
        result: a dict, contains:
                max_gain: 划分产生的收益;
                feature_select: 选中的特征下标;
                threshold_index: block中行下标，不是直接的sample下标
    """

    processNum = config['train']['processNum']

    length = (X_gh.shape[1] - 2) // processNum
    remainder = (X_gh.shape[1] - 2) % processNum

    pool = mp.Pool(processNum)
    result_mp = []

    for idx in range(processNum):
        start = idx * length
        end = (idx + 1) * length

        if idx == processNum - 1:
            end += remainder
        
        # 这里的result不是立即返回
        result = pool.apply_async(func=findSplit,
                                    args=(X_gh, block, sample_tag, start, end))
        
        result_mp.append(result)
    
    pool.close()
    pool.join()

    result_list = [result.get() for result in result_mp]
    
    max_gain = result_list[0]["max_gain"]
    index = 0

    for i in range(1, len(result_list)):
        if max_gain < result_list[i]["max_gain"]:
            max_gain = result_list[i]["max_gain"]
            index = i 
    
    return result_list[index]


def findSplit(X_gh, block, sample_tag, start, end):
    """在指定特征构成的桶之间寻找最佳分裂点
    """

    max_gain, feature_select, threshold_index = 0, None, None

    # 对特征进行逐一查找
    for feature_i in range(start, end):

        # 对单个特征进行建桶
        buckets = get_buckets(X_gh, block, sample_tag, feature_i)

        # 在所有桶中选择
        bucketsNum = buckets.shape[0]

        for split_index in range(bucketsNum):

            if split_index == 0 or split_index == bucketsNum:
                # 考虑边界情况,这种情况下等于不分裂结点
                continue

            # 根据split_index进行一个划分尝试
            buckets_left, buckets_right = np.split(buckets, [split_index], axis=0)

            # Calculate gain
            current_gain = gain_calculate(buckets, buckets_left, buckets_right)

            # 如果当前划分能够得到一个更优的gain，则保存
            if current_gain > max_gain:
                max_gain = current_gain
                feature_select = feature_i
                threshold_index = int(buckets[split_index][-1])

    # 保存一个进程的运行结果，存储为字典
    result = {
        "max_gain": max_gain,
        "feature_select": feature_select,
        "threshold_index": threshold_index}

    # result_q.put(result)
    return result


def get_buckets(X_gh, block, sample_tag, feature_i):
    """Create buckets by hessian

    Returns:
        buckers: buckertsNum * 3 ; [G、H、block_i_index]; np.array
    """
    eps = config["train"]["eps"]
    gradient, hessian = [], []

    # 遍历单个特征
    for i in range(block.shape[0]):

        # 找到 X_gh 中对应的sample下标
        sample_index = int(block[i][feature_i])

        if sample_tag[sample_index] == 1:
            # 如果该sample位于当前节点中
            gradient.append(X_gh[sample_index][-2])
            hessian.append(X_gh[sample_index][-1])

    gradient = np.array(gradient)
    hessian = np.array(hessian)

    H_threshold = eps * hessian.sum()

    # 创建桶
    G_sk, H_sk, index_sk = [], [], []
    G_temp, H_temp = 0, 0

    for i in range(hessian.shape[0]):
        if H_temp < H_threshold:
            G_temp += gradient[i]
            H_temp += hessian[i]

        else:
            G_sk.append(G_temp)
            H_sk.append(H_temp)
            # 存储的是上个桶最后的block_i中第几行的下标
            index_sk.append(i - 1)

            G_temp = gradient[i]
            H_temp = hessian[i]

    # 将最后几个sample组成一个桶
    G_sk.append(G_temp)
    H_sk.append(H_temp)
    index_sk.append(hessian.shape[0] - 1)

    # 转换为n*1的矩阵
    G_sk = np.array(G_sk).reshape(len(G_sk), 1)
    H_sk = np.array(H_sk).reshape(len(H_sk), 1)
    index_sk = np.array(index_sk).reshape(len(index_sk), 1)

    buckets = np.concatenate((G_sk, H_sk, index_sk), axis=1)

    return buckets


def structure_score(buckets):
    """ Calculate the structure score of the current node 
    """

    lambta = config["train"]["lambta"]

    gradient = buckets[:, :1]
    hessian = buckets[:, 1:2]

    G_2 = np.power(gradient.sum(), 2)
    H = hessian.sum()

    return 0.5 * (G_2 / (H + lambta))


def gain_calculate(buckets, buckets_left, buckets_right):
    """ Calculates the payoff of the current node split 
    """

    gamma = config["train"]["gamma"]

    left_score = structure_score(buckets_left)
    right_score = structure_score(buckets_right)
    current_score = structure_score(buckets)

    gain = 0.5 * (left_score + right_score - current_score) - gamma

    return gain


def leaf_weight_calculation(X_gh, sample_tag):
    """calculate leaf weights
    """

    gradient, hessian = [], []
    lambta = config["train"]["lambta"]
    sampleNum = sample_tag.shape[0]

    for i in range(sampleNum):
        if sample_tag[i] == 1:
            # 如果该sample位于当前节点中
            gradient.append(X_gh[i][-2])
            hessian.append(X_gh[i][-1])

    gradient = np.array(gradient)
    hessian = np.array(hessian)

    leaf_weight = - (gradient.sum() / (hessian.sum() + lambta))

    return leaf_weight

