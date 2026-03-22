import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split

delta = pd.read_csv("data.csv")
X = data.drop(columns=['id', 'diagnosis']).values
y = data['diagnosis'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
plt.scatter(X_train[y_train == 'B', 0], X_train[y_train == 'B', 1], color='tab:green', label='Bening')
plt.scatter(X_train[y_train == 'M', 0], X_train[y_train == 'M', 1], color='tab:red', label='Bening')
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.plot()

def euclidean_distance(a,b):
  return np.sqrt(sum([(a[i] - b[i])**2 for i in range(len(a))]))

class KNN:
  def __init__(self, k):
    self.k = k

  def fit(self, X, y):
    self.X_train = X
    self.y_train = y
  
  def predict(self, new_points):
    predictions = [self.predict_class(new_point) for new_point in new_points]
    return np.array(predictions)

  def predict_class(self, new_point):
    distances = [euclidean_distance(point, new_point) for point in self.X_train]
  
    k_nearest_indices = np.argsort(distances)[:self.k] 
    #sorts by distance-returns k indexes of the smallest distances points
    k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

    most_common = Counter(k_nearest_labels).most_common(1)[0][0]
    #get the most common class label

    return most_common

knn = KNN(5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
accuracy = np.mean(predictions == y_test) * 100
print(accuracy)


