import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter
#help us perform majority voting by selecting the most common 
#class label when assigning a prediction toa lef node
from sklearn.model_selection import train_test_split

data = pd.read_csv("Iris.csv")

X = data.drop(columns=["id", "Species"]).values
Y = data["Species"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

class Node:
  def __init__(self, feature_idx=None, threshold=None, info_gain=None, left=None,right=None, value=None):
    #Decision Node
    
    self.feature_idx = feature_idx
    self.threshold = threshold
    self.info_gain = info_gain
    self.left = left
    self.right = right
    
    #Leaf Node
    #it stores the final majority value for the predicted class value
    self.value = value

class DeciosionTree:
  def __init__(self, min_samples_split=2, max_depth=2):
    self.min_samples_split = min_samples_split
    self.max_depth = max_depth
    # stopping conditions so the tree doesnt split forever

  def build_tree(self, dataset,  curr_depth=0):
    X, y = dataset[:, :-1], dataset[:, -1]
    #take all rows : except the last one :-1
    n_samples, n_features = X.shape #we take the number of feauters and samples at the current level 
    if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
      best_split = self.best_split(dataset, n_features)

      if best_split["info_gain"] > 0:
        left_node = self.build_tree(best_split["left_dataset"], curr_depth + 1)
        right_node = self.build_tree(best_split["right_dataset"], curr_depth + 1)

        return Node(best_split["feature_idx"], best_split["threshold"], best_split["info_gain"], left_node, right_node)
    
    #if we have reached the max depth we return a leaf node
    leaf_value = Counter(y).most_common(1)[0][0]

    #getting the most frequent value
    return Node(value=leaf_value)
  
  def best_split(self, dataset, n_features):
    best_split = {
        'feature_idx': None,
        'threshold': None,
        'info_gain': -1,
        'left_datast': None,
        'right_dataset': None
    }
    #for each feature we want to get the best threshold [22,333,4]
    #and split by if data <= t it goes in the left side and if is bigger than the 
    #threshold t they go on the right
    for feature_idx in range(n_features):
      feature_values = dataset[:, feature_idx]
      thresholds = np.unique(feature_values)
      for threshold in thresholds:
        left_dataset, right_dataset = self.split(dataset, feature_idx, threshold)
        #split method where data[feature_idx]<=threshold for left part
        #and analogically for the right part
        if len(left_dataset) and len(right_dataset):
          parent_y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
           #get all the rows in the last column

          info_gain = self.infromation_gain(parent_y, left_y, right_y)

          if info_gain > best_split['info_gain']:
            best_split['feature_idx'] = feature_idx
            best_split['threshold'] = threshold
            best_split['info_gain'] = info_gain
            best_split['left_dataset'] = left_dataset
            best_split['right_dataset'] = right_dataset

    return best_split
  def split(self, dataset, feature_idx, threshold):
      left_dataset = np.array([row for row in dataset if row[feature_idx] <= threshold])     
      right_dataset = np.array([row for row in dataset if row[feature_idx] > threshold])
      #separates the dataset into two halves based on a threshold
      return left_dataset, right_dataset

  def information_gain(self, parent_y, left_y, right_y):
      left_weight = len(left_y) / len(parent_y)
      right_weight = len(right_y) / len(parent_y)
      #IG formula -> entropy[parent] - w_left.entropy[left] - w_right.entropy[right]
      information_gain = self.entropy(parent_y) - (left_weight*self.entropy(left_y) + right_weight * self.entropy(right_y))
      return information_gain

  def entropy(self, y):
      entropy = 0

      class_labels = np.unique(y)
      for class_label in class_labels:
          p = len(y[y == class_label]) / len(y)
          #p - probability of class i
          entropy -= -p * np.log2(p)
      return entropy

  #Functions for fitting and predicting

  def fit(self, X, y):
    dataset = np.concatanate([X, y.reshape(-1, 1)], axis=1)
    #y - is now a column - combining the two datasets into one
    self.root = self.build_tree(dataset)

  def predict(self, X):
      predictions = [self.predict_class(row, self.root) for row in X]
      return predictions

  def predict_class(self, row, node):
    #if it is a leaf node return the predicted value
      if node.value is not None:
          return node.value
      feature_val = row[node.feature_idx]
      #if we are on a parent node -> two scenarios 
      #if <= t we go on th left
      if feature_val <= node.threshold:
          return self.predict_class(row, node.left)
      else:
          return self.predict_class(row, node.right)

dt = DecisionTree(min_samples_split=2, max_depth=2)
dt.fit(X_train, y_train)
predictions = dt.predict(X_test)
accuracy = np.mean(predictions == y_test)*100 
#prints the accuracy if we have [0,1,0] and [0,0,0] - we have two of three matching pairs so 66% accuracy

print(accuracy)

