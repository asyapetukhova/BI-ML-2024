import numpy as np


class KNNClassifier:
    """
    K-neariest-neighbor classifier using L1 loss
    """
    
    def __init__(self, k=1):
        self.k = k
    

    def fit(self, X, y):
        self.train_X = X
        self.train_y = y


    def predict(self, X, n_loops=0):
        """
        Uses the KNN model to predict clases for the data samples provided
        
        Arguments:
        X, np array (num_samples, num_features) - samples to run
           through the model
        num_loops, int - which implementation to use

        Returns:
        predictions, np array of ints (num_samples) - predicted class
           for each sample
        """
        
        if n_loops == 0:
            distances = self.compute_distances_no_loops(X)
        elif n_loops == 1:
            distances = self.compute_distances_one_loops(X)
        else:
            distances = self.compute_distances_two_loops(X)
        
        if len(np.unique(self.train_y)) == 2:
            return self.predict_labels_binary(distances)
        else:
            return self.predict_labels_multiclass(distances)


    def compute_distances_two_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Uses simplest implementation with 2 Python loops

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                # Расстояние L1 между i-м образцом тестового набора и j-м образцом обучающего набора
                dists[i, j] = np.sum(np.abs(X[i] - self.train_X[j]))

        return dists



    def compute_distances_one_loop(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Vectorizes some of the calculations, so only 1 loop is used

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            dists[i] = np.sum(np.abs(X[i] - self.train_X), axis=1)

        return dists
        
        
    def compute_distances_no_loops(self, X):
        """
        Computes L1 distance from every sample of X to every training sample
        Fully vectorizes the calculations using numpy

        Arguments:
        X, np array (num_test_samples, num_features) - samples to run
        
        Returns:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        """

        
        num_test = X.shape[0]
        num_train = self.train_X.shape[0]
        dists = np.sum(np.abs(X[:, np.newaxis] - self.train_X), axis=2)

        return dists


    def predict_labels_binary(self, distances):
        """
        Returns model predictions for binary classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of bool (num_test_samples) - binary predictions 
           for every test sample
        """

        n_train = distances.shape[1]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test)
        
        # Находим индекс ближайшего обучающего примера к i-му тестовому примеру
        closest_idx = np.argmin(distances, axis=1)
    
        # Предсказываем метку ближайшего обучающего примера
        prediction = self.train_y[closest_idx]

        return prediction


    def predict_labels_multiclass(self, distances):
        """
        Returns model predictions for multi-class classification case
        
        Arguments:
        distances, np array (num_test_samples, num_train_samples) - array
           with distances between each test and each train sample
        Returns:
        pred, np array of int (num_test_samples) - predicted class index 
           for every test sample
        """

        n_train = distances.shape[0]
        n_test = distances.shape[0]
        prediction = np.zeros(n_test, dtype=int) # Здесь был np.int, пришлось исправить код, поскольку в новых версиях Numpy это устарело

        
        for i in range(n_test):
        # Находим индексы k ближайших соседей для текущего тестового образца
            closest_indices = np.argsort(distances[i])[:self.k]
        # Получаем классы ближайших соседей
            closest_classes = self.train_y[closest_indices]
        # Выбираем наиболее часто встречающийся класс среди ближайших соседей
            prediction[i] = np.argmax(np.bincount(closest_classes))
    
        return prediction