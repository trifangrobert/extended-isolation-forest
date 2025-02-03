import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

# https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# https://en.wikipedia.org/wiki/Isolation_forest
def c(m: int) -> float:
    # average path length of unsuccessful search in a binary search tree
    if m > 2:
        return 2 * (np.log(m - 1) + 0.5772156649) - 2 * (m - 1) / m
    elif m == 2:
        return 1
    else:
        return 0

class SplitNode:
    def __init__(self, feature: int, threshold: float, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        
class ExtendedNode:
    def __init__(self, normal_vector: np.ndarray, intercept_point: np.ndarray, left=None, right=None):
        self.normal_vector = normal_vector
        self.intercept_point = intercept_point
        self.left = left
        self.right = right
        
class LeafNode:
    def __init__(self, size: int, data: np.ndarray):
        self.size = size
        self.data = data

class IsolationTree:
    def __init__(self, current_height: int, height_limit: int):
        self.current_height = current_height
        self.height_limit = height_limit
        self.node = None
        
    def fit(self, X: np.ndarray):
        if self.current_height >= self.height_limit or X.shape[0] <= 1 or np.all(X == X[0]):
            self.node = LeafNode(X.shape[0], X)
            return
        
        split_feature = np.random.randint(X.shape[1])
        mn_feature_value = X[:, split_feature].min()
        mx_feature_value = X[:, split_feature].max()
        split_threshold = np.random.uniform(mn_feature_value, mx_feature_value)
        
        left_indices = X[:, split_feature] < split_threshold
        right_indices = ~left_indices
        
        X_left = X[left_indices]
        X_right = X[right_indices]
        
        left_tree = IsolationTree(self.current_height + 1, self.height_limit)
        right_tree = IsolationTree(self.current_height + 1, self.height_limit)
        
        left_tree.fit(X_left)
        right_tree.fit(X_right)
        
        self.node = SplitNode(split_feature, split_threshold, left_tree, right_tree)
        
    def path_length(self, x: np.ndarray) -> float:
        if isinstance(self.node, LeafNode):
            return self.current_height + c(self.node.size)
        
        if x[self.node.feature] < self.node.threshold:
            return self.node.left.path_length(x)
        else:
            return self.node.right.path_length(x)
       
    def print(self):
        print(f"Height: {self.current_height}")        
        if isinstance(self.node, LeafNode):
            print(f"Leaf: {self.node.size}")
        else:
            print(f"Split: {self.node.feature}, {self.node.threshold}")
            self.node.left.print()
            self.node.right.print()
            
class RotationTree:
    def __init__(self, current_height: int, height_limit: int, origin=(0, 0)):
        self.current_height = current_height
        self.height_limit = height_limit
        self.node = None
        
        self.origin = origin
        self.angle = None
        
    def fit(self, X: np.ndarray):
        if self.current_height == 0:
            self.angle = np.random.uniform(0, 2 * np.pi)
            X = np.array([rotate(self.origin, (x, y), self.angle) for x, y in X])
        
        if self.current_height >= self.height_limit or X.shape[0] <= 1 or np.all(X == X[0]):
            self.node = LeafNode(X.shape[0], X)
            return
        
        split_feature = np.random.randint(X.shape[1])
        mn_feature_value = X[:, split_feature].min()
        mx_feature_value = X[:, split_feature].max()
        split_threshold = np.random.uniform(mn_feature_value, mx_feature_value)
        
        left_indices = X[:, split_feature] < split_threshold
        right_indices = ~left_indices
        
        X_left = X[left_indices]
        X_right = X[right_indices]
        
        left_tree = IsolationTree(self.current_height + 1, self.height_limit)
        right_tree = IsolationTree(self.current_height + 1, self.height_limit)
        
        left_tree.fit(X_left)
        right_tree.fit(X_right)
        
        self.node = SplitNode(split_feature, split_threshold, left_tree, right_tree)
        
    def path_length(self, x: np.ndarray) -> float:
        if self.current_height == 0:
            x = rotate(self.origin, x, self.angle)
        
        if isinstance(self.node, LeafNode):
            return self.current_height + c(self.node.size)
        
        if x[self.node.feature] < self.node.threshold:
            return self.node.left.path_length(x)
        else:
            return self.node.right.path_length(x)
        
class ExtendedTree:
    def __init__(self, current_height: int, height_limit: int, extension_level: int = 0):
        self.current_height = current_height
        self.height_limit = height_limit
        self.node = None
        
        self.extension_level = extension_level
        
    def fit(self, X: np.ndarray):
        if self.current_height >= self.height_limit or X.shape[0] <= 1 or np.all(X == X[0]):
            self.node = LeafNode(X.shape[0], X)
            return
        
        # normal_vector = np.random.normal(0, 1, X.shape[1])
        normal_vector = np.random.uniform(-1, 1, X.shape[1])
        normal_vector /= np.linalg.norm(normal_vector)
        
        mn_feature_value = np.min(X, axis=0)
        mx_feature_value = np.max(X, axis=0)
        intercept_point = np.random.uniform(mn_feature_value, mx_feature_value)

        extension_indices = np.random.choice(X.shape[1], X.shape[1] - self.extension_level, replace=False)
        normal_vector[extension_indices] = 0
            
        branch = (X - intercept_point).dot(normal_vector)
            
        left_indices = branch < 0
        right_indices = ~left_indices
        
        X_left = X[left_indices]
        X_right = X[right_indices]
        
        left_tree = ExtendedTree(self.current_height + 1, self.height_limit, self.extension_level)
        right_tree = ExtendedTree(self.current_height + 1, self.height_limit, self.extension_level)
        
        left_tree.fit(X_left)
        right_tree.fit(X_right)
        
        self.node = ExtendedNode(normal_vector, intercept_point, left_tree, right_tree)
        
    def path_length(self, x: np.ndarray) -> float:
        if isinstance(self.node, LeafNode):
            return self.current_height + c(self.node.size)
        
        branch = (x - self.node.intercept_point).dot(self.node.normal_vector)
        if branch < 0:
            return self.node.left.path_length(x)
        else:
            return self.node.right.path_length(x)
            
            
class IsolationForest:
    def __init__(self, n_trees: int = 100, sample_size: int = 256, tree_type="isolation", tree_args={}):
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.trees = []

        if tree_type == "isolation":
            self.tree = IsolationTree
        elif tree_type == "rotation":
            self.tree = RotationTree
        elif tree_type == "extended":
            self.tree = ExtendedTree
        self.tree_args = tree_args
        
    def fit(self, X: np.ndarray):
        self.sample_size = min(self.sample_size, X.shape[0])
        
        height_limit = np.ceil(np.log2(self.sample_size))
        for i in tqdm(range(self.n_trees), desc="Fitting Isolation Forest"):
            indices = np.random.choice(X.shape[0], self.sample_size, replace=False)
            sample = X[indices]
            tree = self.tree(0, height_limit, **self.tree_args)
            tree.fit(sample)
            self.trees.append(tree)
            
    def path_length(self, X: np.ndarray):
        paths = []
        for x in tqdm(X, desc="Computing path lengths"):
            path_lengths = [tree.path_length(x) for tree in self.trees]
            paths.append(path_lengths)
        paths = np.array(paths)
        avg_path_lengths = np.mean(paths, axis=1)
        return avg_path_lengths
    
    def anomaly_score(self, X: np.ndarray):
        avg_path_lengths = self.path_length(X)
        scores = [2 ** (-avg_length / c(self.sample_size)) for avg_length in avg_path_lengths]
        return scores
    
        
        
# Example
if __name__ == "__main__":
    # generate two clusters of data with center at (0, 0) and (10, 10)
    X1 = np.random.randn(500, 2)
    X2 = np.random.randn(500, 2) + 10
    X = np.vstack([X1, X2])
    
    # generate only one cluster of data at (0, 0)
    X = np.random.randn(1000, 2)
    
    # generate sinusoidal data
    x = np.random.rand(1000) * 8 * np.pi
    y = np.sin(x) + np.random.randn(1000) / 4
    X = np.c_[x, y]
    
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    
    model = IsolationForest(n_trees=100, sample_size=500, tree_type="extended", tree_args={"extension_level": 2})
    # model = IsolationForest(n_trees=100, sample_size=500, tree_type="rotation", tree_args={"origin": (0, 0)})
    model.fit(X)
    
    scores = model.anomaly_score(X)
    plt.scatter(X[:, 0], X[:, 1], c=scores, cmap="coolwarm")
    plt.colorbar()
    plt.show()
    
    # one cluster
    # x = np.linspace(-5, 5, 100)
    # y = np.linspace(-5, 5, 100)
    
    # two clusters
    x = np.linspace(-5, 15, 100)
    y = np.linspace(-5, 15, 100)
    
    # sinusoidal data
    # x = np.linspace(-5, 15, 100)
    # y = np.linspace(-3, 13, 100)
    
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    scores = model.anomaly_score(grid)
    
    plt.scatter(grid[:, 0], grid[:, 1], c=scores, cmap=plt.cm.YlOrRd)
    plt.colorbar()
    plt.show()
