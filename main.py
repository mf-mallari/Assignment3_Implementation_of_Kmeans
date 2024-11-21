import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris 
# from sklearn.metrics import  
from sklearn.metrics import davies_bouldin_score
from cluster import (KMeans)
from cluster.visualization import plot_3d_clusters


def main(): 
    # Set random seed for reproducibility
    np.random.seed(42)
    # Using sklearn iris dataset to train model
    og_iris = np.array(load_iris().data)
    
    # Initialize your KMeans algorithm
    KMeans_model = KMeans(k = 5, metric= "euclidean", max_iter = 1000, tol = 1e-6)
    
    # Fit model
    KMeans_model.fit(og_iris)

    # Load new dataset
    df = np.array(pd.read_csv('data/iris_extended.csv', 
                usecols = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']))

    # Predict based on this new dataset
    prediction = KMeans_model.predict(df)
    
    # You can choose which scoring method you'd like to use here:
    db_score = davies_bouldin_score(df, prediction)
    print(f"Davies-Bouldin Score: {db_score}")
    
    # Plot your data using plot_3d_clusters in visualization.py
    plot_3d_clusters(df, prediction, KMeans_model.get_centroids(), db_score)
    
    # Try different numbers of clusters (loop)
    inertia_values = []
    db_scores = []
    k_values = range(2, 19)  # Testing k from 2 to 9
    for k in k_values:
        temp_kmeans = KMeans(k=k, metric="euclidean", max_iter=500, tol=1e-6)
        temp_kmeans.fit(og_iris)
        temp_prediction = temp_kmeans.predict(og_iris)
        inertia_values.append(temp_kmeans.get_error())
        db_scores.append(davies_bouldin_score(og_iris, temp_prediction))
    
    # Plot the elbow plot
    plt.figure()
    plt.plot(k_values, inertia_values, marker='o', label='Inertia')
    plt.title("Elbow Plot")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.legend()
    plt.grid()
    plt.show()
    
    
    # Question: 
    # Please answer in the following docstring how many species of flowers (K) you think there are.
    # Provide a reasoning based on what you have done above: 
    
    """
    How many species of flowers are there: 5
    
    Reasoning: at 5 clusters there is a significant slowing of the inertia decrease, forming an "elbow". Additionally 
    by looking at the 3d plot divided into 5 clusters, you can make out 5 distinct planes of separation. For 4 of the species
    the features are relatively close, which may explain why beyond 5 clusters there is some waivering in the inertia. 

    A note about the Elbow plot: depending on the seed value, or the location of the first centroids, affects the initial inertia.
    At 42 there is an inital increase in the interita before a steep drop off. However, when choosing another random seed value, say 10,
    then the elbow plot looks as it should. Perhaps signifying that the "random" point fell close to the correct clusters/
       
    I hypothesize the initial spike in interita is due to the relative proximity of all of the data clusters. These intial clusters may 
    be pulled towards a different cluster due to the close proximity, once there are a few adjustments to the centroid position, and the 
    datapoints belong to their respective clusters, that is when we will see inertia drop.
    
    
    
    """

    
if __name__ == "__main__":
    main()