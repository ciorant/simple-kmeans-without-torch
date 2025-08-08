import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
import sys


class KMeans:
    def __init__(self, k, max_iter=160, convergence=1e-5, init_method='kmeans++'):
        self.k = k
        self.max_iter = max_iter
        self.convergence = convergence
        self.centroids = None
        self.labels = None
        self.init_method = init_method

    def _kmeans_plus_plus_init(self,data):
        # K-means++ initialization
        m,n = data.shape
        centroids = np.zeros((self.k,n))
        centroids[0] = data[np.random.choice(m)]

        for i in range (1,self.k):
            distance_squared = np.array([
                min([np.linalg.norm(point-centroid)**2 for centroid in centroids[:1]])
                for point in data
            ])
            probabilities = distance_squared / distance_squared.sum()
            cumulative_prob = probabilities.cumsum()
            random_val = np.random.rand()
            chosen_idx = np.where(cumulative_prob >= random_val)[0][0]
            centroids[i] = data[chosen_idx]

        return centroids

    def fit(self, data):
        m, n = data.shape

        if self.init_method == 'kmeans++':
            self_centroids = self._kmeans_plus_plus_init(data)
        else:
            indices = np.random.choice(m, self.k, replace=False)
            self.centroids = data[indices]

        self.labels = np.zeros(m, dtype=int)

        for iteration in range(self.max_iter):
            old_centroids = self.centroids.copy()

            for i in range(m):
                self.labels[i] = np.argmin(np.linalg.norm(data[i] - self.centroids, axis=1))

            for j in range(self.k):
                points = data[self.labels == j]
                if len(points) > 0:
                    self.centroids[j] = np.mean(points, axis=0)
                else:
                    self.centroids[j] = data[np.random.choice(m)]

            if np.linalg.norm(self.centroids - old_centroids) < self.convergence:
                break

        return self.centroids, self.labels

    def predict(self, data):
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")

        predictions = np.zeros(len(data), dtype=int)
        for i in range(len(data)):
            distances = np.linalg.norm(data[i] - self.centroids, axis=1)
            predictions[i] = np.argmin(distances)
        return predictions

    def compress_image(self, image_data):
        predictions = self.predict(image_data)
        compressed_data = self.centroids[predictions]
        return compressed_data


class ImageCompressor:
    def __init__(self, k_clusters=16,init_method='kmeans++'):
        self.k_clusters = k_clusters
        self.init_method = init_method
        self.kmeans = KMeans(k_clusters,init_method=init_method)
        self.original_shape = None

    def load_image(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Image file not found: {filepath}")

        image = imread(filepath).astype(float) / 255.0
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be RGB format")

        return image

    def train(self, training_image_path):
        training_image = self.load_image(training_image_path)
        training_pixels = training_image.reshape(-1, 3)
        self.kmeans.fit(training_pixels)
        return training_image

    def compress(self, target_image_path):
        target_image = self.load_image(target_image_path)
        self.original_shape = target_image.shape
        target_pixels = target_image.reshape(-1, 3)

        compressed_pixels = self.kmeans.compress_image(target_pixels)
        compressed_image = compressed_pixels.reshape(self.original_shape)

        return target_image, compressed_image

    def visualize_results(self, original, compressed, title="Image Compression Results"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.imshow(original)
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2.imshow(compressed)
        ax2.set_title(f"Compressed ({self.k_clusters} colors)")
        ax2.axis('off')

        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    def save_compressed(self, compressed_image, output_path):
        plt.imsave(output_path, np.clip(compressed_image, 0, 1))


def main():
    try:
        compressor = ImageCompressor(k_clusters=16)

        training_image = compressor.train('data/peppers-small.tiff') # Training image that I'm using
        original, compressed = compressor.compress('data/peppers-large.tiff') # Again training image -
        # fill out with your own!
        compressor.visualize_results(original, compressed)

        compression_ratio = (original.size * 8) / (
                    compressor.k_clusters * 3 * 8 + original.size * np.log2(compressor.k_clusters))
        print(f"Approximate compression ratio: {compression_ratio:.2f}:1")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the image files exist in the 'data' directory")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()