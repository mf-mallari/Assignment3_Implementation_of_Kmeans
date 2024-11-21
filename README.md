# Assignment 3: Implementation of Kmeans from Scratch


## Assignment Overview: 

In this assignment, you will be implementing the K-Means algorithm from scratch. K-Means is one of the fundamental unsupervised learning algorithms that partition data into K-distinct clusters based on distance metrics. 

You will be working with the classic Iris dataset to train your implementation and an extended iris dataset to test your implementation. 

Additionally, you will be responsible for finding out how many clusters there actually are (no googling the answer of course ]: )!

For this assignment, we will be:

1. Implementing KMeans clustering from scratch
2. Using the algorithm to cluster the classic Iris dataset
3. Creating visualizations to understand cluster performance
4. Using the elbow method to determine optimal cluster numbers

## Iris Dataset Context

The original Iris dataset contains 4 recorded features of the iris flower: 

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width


## Assignment Context

### KMeans Clustering
KMeans clustering works by:

1. Randomly initializing K centroids
2. Assigning points to the nearest centroid
3. Updating centroid positions based on the mean of assigned points
4. Repeating steps n-iterations until convergence

The algorithm uses distance metrics (typically Euclidean) to measure the similarity between points and centroids. 

### Plotting Predictions
Once you've successfully created your KMeans algorithm, initialize your KMeans algorithm, fit and predict your model on the extended-iris dataset, choose a scoring method to use and plot it!

Note: Make sure you import the scoring method you chose.

### Finding K 
One method you may remember from class is the elbow technique which helps you determine the optimal number of clusters (K) for KMeans clustering. It works by: 
1. Running KMeans with different values of K
2. Calculating the inertia for each K
3. Plotting K vs. inertia 
4. Finding the elbow point where increasing K yields diminishing returns 

### Resources to help you get started

Creating KMeans: 
* [KMeans Algorithm Explanation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
* [Randomizing my centroid position](https://numpy.org/doc/2.1/reference/random/generated/numpy.random.choice.html)
* [Assigning data to centroids](https://numpy.org/doc/stable/reference/generated/numpy.argmin.html)
* [What is cdist?](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html)


Plotting your data and Finding K
* [Understanding the Elbow Method](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/)
* [Scoring methods](https://scikit-learn.org/stable/api/sklearn.metrics.html#module-sklearn.metrics.cluster)



Note: You do not need to modify anything in visualization.py.

### As a reminder, try not to use ChatGPT to generate code, but have it suggest tools that may be helpful


# Grading (70 points total): 

### KMeans Class Implementation (45 points)
* Init method (5 points):
    * Correctly initializes all parameters (5)

* Fit method (25 points): 
    * Correct random centroid initialization (5)
    * Correct implementation of distance (5)
    * Correct cluster assignment (5)
    * Correct centroid update mechanism (5)
    * Correct convergence checking (5)

* Predict method (5 points):
    * Correct assignment of new points to clusters (5)

* Helper Methods (10 points):
    * get_error implementation (5)
    * get_centroid implementation (5)

### Analysis and Visualization (25 points)
* Evaluation (1):
    * Model predicts centroid on new dataset (1)

* Visualization (20):
    * Picked the proper scoring method to evaluate the KMeans model (5)
    * Utilized plot_3d_cluster to view clusters (3)
    * Generated Elbow Plot (12)

* Analysis (4):
    * Correct K prediction (1):
    * Valid K prediction reasoning (3): 

