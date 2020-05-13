from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import numpy
from matplotlib import pyplot as plt
import os


class PreClustering:

    def __init__(self, data):
        self.data = data

    def plot_distances_neighbors(self):
        try:
            neigh = NearestNeighbors(n_neighbors=2)
            neigh.fit(self.data)
            distances, indices = neigh.kneighbors(self.data, return_distance=True)
            distances = numpy.sort(distances, axis=0)
            distances = distances[:, 1]
            plt.clf()
            plt.ylabel('Distances')
            plt.xlabel('N points')
            plt.title('Nearest neighbors distances per point\n choose epsilon (maximum curvature)')
            plt.plot(distances)
            plt.show()
        except Exception as e:
            print('\nProcedure  PreClustering.plot_distances_neighbors did not work', e.__repr__())

    def epsilon(self):
        try:
            self.plot_distances_neighbors()
            epsilon = input('Enter epsilon: ')
            #epsilon = 0.27
            return epsilon
        except Exception as e:
            print('\nMethod PreClustering.epsilon did not work', e.__repr__())

    def running(self):
        try:
            print('\nStarting dbscan...')
            epsilon = float(self.epsilon())
            clusters = DBSCAN(eps=epsilon, min_samples=2,
                              metric='euclidean', n_jobs=(os.cpu_count() - 1)).fit_predict(self.data)
            return clusters
        except Exception as e:
            print('\nMethod PreClustering.running did not work', e.__repr__())

    def display_clusters_outliers(self, outliers, data, labels_clusters):
        try:
            if outliers:
                data['cluster'] = labels_clusters
                print('\nOutliers rows:\n', data.iloc[outliers])
        except Exception as e:
            print('\nMethod PreClustering.display_clusters_outliers did not work', e.__repr__())







