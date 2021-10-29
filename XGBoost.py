import numpy as np

from config import config
from DecisionTree import DecisionTree
from utils.lossFunctions import SquareLoss, BinaryLoss
from utils.dataFunctions import preSort, sigmod, timecost


class XGBoost(object):

    def __init__(self):

        # Initialization parameters
        self.n_estimators = config["train"]["n_estimators"]  # Number of trees
        self.learning_rate = config["train"]["learning_rate"]  # Step size for weight update
        self.task_label = config["train"]["task_label"]  # The type of task
        self.n_class = config["train"]["n_class"]  # Number of categories in multi-classification tasks

        # choose loss function for tree
        if self.task_label == 0:
            self.loss = SquareLoss()
        else:
            self.loss = BinaryLoss()

        # Initialize decision trees
        self.trees = []

        for i in range(self.n_estimators * self.n_class):
            # 多任务中一个直观的思路是转换为 n 个二分类任务
            tree = DecisionTree(loss=self.loss)
            self.trees.append(tree)

    @timecost("Training model time cost: ")
    def fit(self, X, y_label):
        """Training tree model
        """

        n_samples = X.shape[0]  # get the sample number
        y_label = np.reshape(y_label, (n_samples, -1))

        # Presort according to features and store them as block structures
        block = preSort(X)

        if self.task_label == 0 or self.task_label == 1:

            if self.task_label == 0:
                # regression task, the predicted value is initialized to 0
                y_pred = np.zeros(np.shape(y_label))
            else:
                # binary classification task, the predicted value is initialized to 0.5, represent the probability
                y_pred = np.zeros(np.shape(y_label)) + 0.5

            for i in range(self.n_estimators):

                tree = self.trees[i]

                # Calculate the first and second derivatives of the current y_label and y_pred
                gradient = self.loss.gradient(y_label, y_pred)
                hessian = self.loss.hessian(y_label, y_pred)
                gh = np.concatenate((gradient, hessian), axis=1)

                # Train a tree with the first and second derivatives and block
                tree.fit(X, gh, block)

                # Update y_pred
                update_pred = tree.predict(X)
                update_pred = np.reshape(update_pred, (n_samples, -1))
                y_pred += self.learning_rate * update_pred

                # 在分类问题里面，需要将y_pred归一化到0-1之间
                if self.task_label == 1:
                    y_pred = sigmod(y_pred)

        else:
            # Multi-classification task
            pass
    
    @timecost("Prediction time cost: ")
    def predict(self, X):
        """Make predictions
        """

        y_pred = np.zeros(X.shape[0])

        for tree in self.trees:

            pred_temp = tree.predict(X)
            y_pred += pred_temp

        return y_pred
