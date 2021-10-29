import numpy as np
import multiprocessing as mp

from config import config
from utils.dataFunctions import timecost
from utils.trainingFunctions import leaf_weight_calculation, findBestSplit_mp

class DecisionNode(object):
    """Class that represents a decision node or leaf in the decision tree
    """

    def __init__(self, feature_i=None, threshold=None, value=None,
                 left_branch=None, right_branch=None):
        self.feature_i = feature_i  # Index for the feature that is tested
        self.threshold = threshold  # Threshold value for feature
        self.value = value  # Value if the node is a leaf in the tree
        self.left_branch = left_branch  # 'Left' subtree
        self.right_branch = right_branch  # 'Right' subtree


class DecisionTree(object):
    def __init__(self, loss):
        self.root = None  # root node in decision tree
        self.loss = loss  # loss function

    @timecost("Training a tree time cost: ")
    def fit(self, x_train, gh, block):
        """Build a decision tree

        Args:
            X: len(samples) * len(features); 二维数组
            gh: len(samples) * 2; 二维数组，存储sample对应的一阶导和二阶导
            block: len(samples) * len(features); 二维数组，存储预排序信息
        """

        sample_tag = np.ones(x_train.shape[0], int)  # 标记sample是否存在当前节点

        # Add gh as last column of X
        X_gh = np.concatenate((x_train, gh), axis=1)

        self.root = self._build_tree(X_gh, block, sample_tag, current_depth=0)

    def predict(self, x_test):
        """Classify samples one by one and return the set of predict values

        Returns:
            y_pred: len(sample) * 1; np.array
        """
        y_pred = []

        for sample in x_test:
            pred = self._predict_sample(sample, self.root)
            y_pred.append(pred)

        return np.array(y_pred)

    def _build_tree(self, X_gh, block, sample_tag, current_depth=0):
        """Recursively construct the decision tree
        """

        min_samples_split = config['train']['min_samples_split']
        min_gain_split = config['train']['min_gain_split']
        max_depth = config['train']['max_depth']

        n_samples = sample_tag.sum()
        # 判断是否满足分裂的条件
        if n_samples >= min_samples_split and current_depth < max_depth:
            
            # 采用多进程以及分桶的近似算法获取近似最优分裂点
            result = findBestSplit_mp(X_gh, block, sample_tag)
            
            max_gain = result["max_gain"]
            feature_select = result["feature_select"]
            threshold_index = result["threshold_index"]

            if max_gain > min_gain_split:
                # 满足子树划分的所有条件
                
                # Update sample_tag
                sample_tag_left = sample_tag.copy()

                for i in range(sample_tag.shape[0] - threshold_index):

                    sample = int(block[i + threshold_index][feature_select])

                    if sample_tag_left[sample] == 1:
                        sample_tag_left[sample] = 0

                sample_tag_right = sample_tag - sample_tag_left
                
                # Build subtrees for the right and left branches
                left_branch = self._build_tree(X_gh, block, sample_tag_left, current_depth=current_depth + 1)
                right_branch = self._build_tree(X_gh, block, sample_tag_right, current_depth=current_depth + 1)

                # 返回一个中间节点
                return DecisionNode(feature_i=feature_select,
                                    threshold=X_gh[int(block[threshold_index][feature_select])][feature_select],
                                    left_branch=left_branch,
                                    right_branch=right_branch)

        # 不满足划分子树的条件，则当前节点为叶节点
        leaf_value = leaf_weight_calculation(X_gh, sample_tag)
        return DecisionNode(value=leaf_value)

    def _predict_sample(self, sample, treeNode: DecisionNode):
        """Do a recursive search down the tree and make a prediction of a data sample
        """
        # 查找出口
        if treeNode.value is not None:
            return treeNode.value

        # Choose the feature that we will test
        feature_value = sample[treeNode.feature_i]

        # Determine if we will follow left or right branch
        if feature_value >= treeNode.threshold:
            next_branch = treeNode.right_branch
        else:
            next_branch = treeNode.left_branch

        # Get the predicted value from the subtree
        return self._predict_sample(sample, next_branch)