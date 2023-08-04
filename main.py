import pandas as pd
import numpy as np
np.random.seed(1)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def pca(X, n_components):
    cov_matrix = np.cov(X.T)            #Covariance
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)       #

    # Sort the eigenvalues in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    selected_eigenvectors = sorted_eigenvectors[:, :n_components]
    # Project the data onto the selected eigenvectors
    transformed_X = np.dot(X, selected_eigenvectors)
    return transformed_X


def calculate_euclidean_distances(X, centroids):
    absolute_diff = np.abs(X[:, np.newaxis] - centroids)
    distances = np.sum(absolute_diff, axis=-1)
    return distances


class KMeans:
    def __init__(self, n_clusters=2, max_iter=200):
        self.labels = None
        self.centroids = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(range(len(X)), size=self.n_clusters, replace=False)]
        self.labels = None

        for i in range(self.max_iter):
            self.labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X)

            if np.allclose(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
            plot_clusters(X, new_centroids, Centroids.predict(X), i+1)

    def _assign_labels(self, X):
        distances = calculate_euclidean_distances(X, self.centroids)
        return np.argmin(distances, axis=-1)

    def _update_centroids(self, X):
        new_centroids = []
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                new_centroids.append(np.mean(cluster_points, axis=0))
            else:
                new_centroids.append(self.centroids[i])
        return np.array(new_centroids)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def predict(self, X):

        return self._assign_labels(X)


def test_split(X, y, test_ratio=0.2):
    shuffle_idx = np.random.permutation(len(X))
    test = int(len(X) * test_ratio)
    test_idxs = shuffle_idx[:test]
    train_idx = shuffle_idx[test:]
    x_train = X[train_idx]
    y_train = y[train_idx]
    x_test = X[test_idxs]
    y_test = y[test_idxs]
    return x_train, y_train, x_test, y_test


def plot_clusters(data, centers, assignments, iteration=1):
    # Assign different colors to each cluster
    colors = ['yellow' , 'cyan']

    # Check the dimensionality of the data
    if data.shape[1] == 2:
        # Plot 2D data
        for i in range(len(data)):
            plt.scatter(data[i, 0], data[i, 1], color=colors[assignments[i]])

        # Plot cluster centers
        plt.scatter(centers[:, 0], centers[:, 1], color='black', marker='x', s=100)

        plt.xlabel('X')
        plt.ylabel('Y')

    elif data.shape[1] == 3:
        # Plot 3D data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for i in range(len(data)):
            ax.scatter(data[i, 0], data[i, 1], data[i, 2], color=colors[assignments[i]])

        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], color='black', marker='x', s=100)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.xlabel('K-Means Clustering')
    plt.title(f'iteration: {iteration}')
    plt.show()


Heart_Data = pd.read_csv('heart.csv')
X = Heart_Data.iloc[:, :-1].values
y = Heart_Data.iloc[:, -1].values
# X=(X-np.mean(X))/np.std(X)
scaler = StandardScaler()
X = scaler.fit_transform(X)
highest_components = 2
X = pca(X, highest_components)



x_train, y_train, x_test, y_test = test_split(X, y, test_ratio=0.2)
Centroids = KMeans(n_clusters=2, max_iter=1000)
Centroids.fit(x_train)

train_labels = Centroids.predict(x_train)
test_labels = Centroids.predict(x_test)
train_acc = Centroids.accuracy(train_labels, y_train) * 100
test_acc = Centroids.accuracy(test_labels, y_test) * 100
print(f'Train Accuracy: {train_acc} %')
print(f'Test Accuracy : {test_acc} %')
plot_clusters(X, Centroids.centroids, Centroids.predict(X), 000)













P = Centroids.predict(X)
KMean_Accuracy = Centroids.accuracy(y , P)
"""------------------------------------------>\\ *RandomForest* //<---------------------------------------------"""
RandomForestModel = RandomForestClassifier()
RandomForestModel.fit(x_train , y_train)
RandomForest_Accuracy = RandomForestModel.score(x_test , y_test)
print(f'Random Forest Accuracy: {RandomForest_Accuracy * 100}\n')
"""------------------------------------------>\\ *DecisionTree* //<---------------------------------------------"""
DecisionTreeModel = DecisionTreeClassifier()
DecisionTreeModel.fit(x_train , y_train)
DecisionTree_Accuracy = DecisionTreeModel.score(x_test , y_test)
print(f'Decision Tree Accuracy: {DecisionTree_Accuracy * 100}\n')
"""--------------------------------------->\\ *LogisticRegression* //<---------------------------------------------"""
LogisticRegressionModel = LogisticRegression()
LogisticRegressionModel.fit(x_train , y_train)
LogisticTestPrediction = LogisticRegressionModel.predict(x_test)
Logistic_Accuracy = accuracy_score(y_test , LogisticTestPrediction)







Accuracy_Models = {
    'K-Mean' : KMean_Accuracy*100 ,
    'Random Forest': RandomForest_Accuracy*100 ,
    'Decision Tree': DecisionTree_Accuracy*100 ,
    'Logistic Regression': Logistic_Accuracy*100
}

plt.figure(figsize=(10,10))
plt.bar(*zip(*Accuracy_Models.items()), color=['Red' , 'Cyan' , 'Yellow' , 'Green'])
plt.title('Models')
plt.ylabel('Accuracy')
plt.show()
