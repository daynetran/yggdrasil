# You may want to install "gprof2dot"
from collections import Counter

import math
import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
import sklearn.tree
import matplotlib.pyplot as plt
from rcviz import callgraph, viz
import pydot

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


# Vectorized function for hashing for np efficiency
def w(x):
    return np.int(hash(x)) % 1000

h = np.vectorize(w)

class Node:
    def __init__(self, surprise, num_samples, num_samples_per_class, predicted_class):
        self.surprise = surprise
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.best_feat_label = None

class DecisionTree:
    def __init__(self, max_depth=3, feature_labels=None):
        # TODO implement __init__ function
        self.max_depth = max_depth
        self.features = feature_labels
        self.depth = 0
        self.root_node = None

    @staticmethod
    def information_gain(X, y, thresh):
        # TODO implement information gain function
        def entropy(y):
            # TODO implement entropy (or optionally gini impurity) function
            classes = np.unique(y)
            num_per_class = [0] * len(classes)
            index = 0
            for c in classes:
                for label in y:
                    if label == c:
                        num_per_class[index] += 1
                index += 1
            entropy = -sum(num / len(y) * math.log(num / len(y),2) for num in num_per_class)          
            return entropy

        current_entropy = entropy(y)
        left_i = np.where(X < thresh)[0]
        right_i = np.where(X >= thresh)[0]
        left_entropy = entropy(y[left_i])
        right_entropy = entropy(y[right_i])

        weighted_avg_entropy_after = (len(left_i) * left_entropy + len(right_i) * right_entropy) / (len(left_i) + len(right_i))
        
        return current_entropy - weighted_avg_entropy_after

    @staticmethod
    def entropy(y):
        # TODO implement entropy (or optionally gini impurity) function
        classes = np.unique(y)
        num_per_class = [0] * len(classes)
        index = 0
        for c in classes:
            for label in y:
                if label == c:
                    num_per_class[index] += 1
            index += 1
        entropy = -sum(num / len(y) * math.log(num / len(y),2) for num in num_per_class)          
        return entropy

    def split(self, X, y, idx, thresh):
        # TODO implement split function
        '''
        
        '''
        left = X[:,idx] < thresh
        X_left, y_left = X[left], y[left]
        X_right, y_right = X[~left], y[~left]
        return X_left, y_left, X_right, y_right

    @viz
    def fit(self, X, y):
        # TODO implement fit function
        '''
        Iterate through features
        '''
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            surprise=self.entropy(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if self.depth == 0:
            self.root_node = node

        if self.depth < self.max_depth:
            best_infogain, best_feat, best_thresh, best_feat_label =  0,0,0, None
            for k in range(X.shape[1]):
                x = X[:,k]
                unique_feat_vals = np.unique(x)
                thresholds = unique_feat_vals[:-1] + np.diff(unique_feat_vals)/2
                for t in thresholds:
                    gain = self.information_gain(x, y, t)
                    if gain > best_infogain:
                        best_infogain = gain
                        best_feat = k
                        best_thresh = t
                        best_feat_label = self.features[k]
            splits = self.split(X, y, best_feat, best_thresh)
            node.feature_index = best_feat
            node.threshold = best_thresh
            node.best_feat_label = best_feat_label
            # print(best_feat_label, best_thresh)
            self.depth += 1
            node.left = self.fit(splits[0], splits[1])
            node.right = self.fit(splits[2], splits[3])
        return node

    def predict(self, X):
        # TODO implement predict function
        def predict_pt(inputs):
            """Predict class for a single sample."""
            node = self.root_node
            while node.left:
                if inputs[node.feature_index] < node.threshold:
                    node = node.left
                else:
                    node = node.right
            return node.predicted_class

        return [predict_pt(X[i, :]) for i in range(X.shape[0])]



class BaggedTrees(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [
            sklearn.tree.DecisionTreeClassifier(random_state=i, **self.params)
            for i in range(self.n)
        ]

    def fit(self, X, y):
        # TODO implement function
        pass

    def predict(self, X):
        # TODO implement function
        pass


class RandomForest(BaggedTrees):
    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        # TODO implement function
        pass

# You do not have to implement the following boost part, though it might help with Kaggle.
class BoostedRandomForest(RandomForest):
    def fit(self, X, y):
        self.w = np.ones(X.shape[0]) / X.shape[0]  # Weights on data
        self.a = np.zeros(self.n)  # Weights on decision trees
        # TODO implement function
        return self

    def predict(self, X):
        # TODO implement function
        pass



def evaluate(clf):
    print("Cross validation:")
    cv_results = cross_validate(clf, X, y, cv=5, return_train_score=True)
    train_results = cv_results['train_score']
    test_results = cv_results['test_score']
    avg_train_accuracy = sum(train_results) / len(train_results)
    avg_test_accuracy = sum(test_results) / len(test_results)

    print('averaged train accuracy:', avg_train_accuracy)
    print('averaged validation accuracy:', avg_test_accuracy)
    if hasattr(clf, "decision_trees"):
        counter = Counter([t.tree_.feature[0] for t in clf.decision_trees])
        first_splits = [
            (features[term[0]], term[1]) for term in counter.most_common()
        ]
        print("First splits", first_splits)

    return avg_train_accuracy, avg_test_accuracy



if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"
    params = {
        "max_depth": 5,
        # "random_state": 6,
        "min_samples_leaf": 10,
    }
    N = 100

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        
        # TODO: preprocess titanic dataset
        # Notes: 
        # 1. Some data points are missing their labels
        # 2. Some features are not numerical but categorical
        # 3. Some values are missing for some features

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = './datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        spam_X = data['training_data']
        spam_y = data['training_labels']
        spam_Z = data['test_data']
        class_names = ["Ham", "Spam"]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)

    # 80/20 training/validation split
    print(spam_X.shape, spam_y.shape)
    spam_tuples = np.append(spam_X, spam_y, axis=1)
    spam_tuples_shuffled = np.random.permutation(spam_tuples)
    spam_tuples_shuffled = np.random.permutation(spam_tuples)

    # divide 80% of data into training data, then assign remaining data as validation data
    eighty_perc_cutoff = int(spam_tuples_shuffled.shape[0] * 0.8)
    spam_training_data = spam_tuples_shuffled[0:eighty_perc_cutoff, 0:-1]
    spam_training_labels = spam_tuples_shuffled[0:eighty_perc_cutoff, -1]
    spam_validation_data = spam_tuples_shuffled[eighty_perc_cutoff:, 0:-1]
    spam_validation_labels = spam_tuples_shuffled[eighty_perc_cutoff:, -1]

    # Basic decision tree for SPAM
    print('==================================================')
    print("\n\n3.4: Performance Evaluation")
    spam_dt = DecisionTree(max_depth=5, feature_labels=features)
    spam_dt.fit(this, X=spam_training_data, y=spam_training_labels)
    callgraph.render("sort.png")
    spam_train_predicts = spam_dt.predict(spam_training_data)
    print("Spam Training Accuracy", accuracy_score(spam_training_labels, spam_train_predicts))
    spam_val_predicts = spam_dt.predict(spam_validation_data)
    print("Spam Validation Accuracy", accuracy_score(spam_validation_labels, spam_val_predicts))

    print('==================================================')
    print("\n\n3.5: Writeup Requirements for the Spam Dataset")
    tree_depth = [1,3,5,10,20,30]
    val_scores = [0] * len(tree_depth)
    c = 0
    for i in tree_depth:
        dt = DecisionTree(max_depth=i, feature_labels=features)
        dt.fit(spam_training_data, spam_training_labels)
        spam_val_predicts = dt.predict(spam_validation_data)
        val_scores[c] = accuracy_score(spam_validation_labels, spam_val_predicts)
        print(f"Spam Validation Accuracy for depth{i}: ", val_scores[c])
        c += 1
    plt.plot(tree_depth, val_scores)
    plt.xlabel("Tree Depth")
    plt.ylabel("Validation Score")
    plt.savefig("Depth_Scores.png")
    print("Plot of Tree Depth vs. Validation Score has been saved to directory.")
    # print("Predictions", dt.predict(Z)[:100])
    print("Tree structure", dt.__repr__())

    # TODO implement and evaluate remaining parts


