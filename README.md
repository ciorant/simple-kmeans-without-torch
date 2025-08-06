# Image Compression using K-Means Clustering

A Python implementation of image compression using K-means clustering algorithm using mostly NumPy (without Torch). The technique reduces the color palette of images by clustering similar colors together.

## Features

- K-means clustering implementation from scratch
- Image compression with customizable color palette size
- Support for RGB images
- Visualization of original vs compressed results
- Modular design for easy integration

## How it Works

- Training Phase: K-means algorithm learns color clusters from a training image
- Compression Phase: Each pixel in the target image is assigned to the nearest cluster centroid
- Reconstruction: Original pixels are replaced with their cluster centroids

