# Clustering Algorithms for Constellation Structure

#### Code can be found in clustering_code.md

As part of an advanced machine learning group project, we applied various clustering algorithms to Hipparcos and Tycho-2 datasets, which contain information about the brightest stars in the sky. Hipparcos contains 36 features of approximately 120,000 rows of data. Tycho-2 is the larger of the two datasets with roughly 2.5 million rows and 37 features. Important features that would most pertinent to the project were extracted for analysis such as galactic longitude and latitude, right ascension, declination, and luminosity.

We performed a supervised and unsupervised portion of the project, which was then combined for a discussion of overall results (althought they can be discussed independently as well). I've only included the unsupervised portion, since it contains my main contributions (which included production of visuals like the one below). The visual below shows a small example of a clustering algorithm ran on a small dataset.

![alt tag](https://github.com/ksiegler1/ConstellationClustering/blob/master/visualexample.png)

We approached the unsupervised problem two ways - 1. given the current constellation structure, is there a clustering algorithm that can provide a good approximation? and 2. ignoring all constellation labeling, is there a better reorganization of constellations if we define good constellations to be compact and separated clusters of stars? The Silhouette Index (SI) was used to evalute cluster performance in this setting, which provides a measure of compactness and separate-ness for clusters. SI values closer to 1 are optimal.

For comparison of clusters between models, which is appropriate for addressing the first problem, we used the Adjusted Rand Index (ARI). The ARI is used to compare constellation labels to external criteria (the ground truth), in our case the current constellation labels.

The clustering algorithms employed included spherical k-means++, affinity propagation, mean shift, and heirarchical clustering. Of these, affinity propagation performed best for both metrics, with ARI = 0.393 and SI = 0.474.

