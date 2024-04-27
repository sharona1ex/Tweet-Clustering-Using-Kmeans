# Tweet-Clustering-Using-Kmeans

## Objective
Implement the tweet clustering function using the Jaccard Distance metric and K-means clustering algorithm to cluster redundant/repeated tweets into the same cluster. The K-means algorithm is to be developed from scratch.

## Motivation
Twitter provides a service for posting short messages. In practice, many of the tweets are very 
similar to each other and can be clustered together. By clustering similar tweets together, we can 
generate a more concise and organized representation of the raw tweets, which will be very 
useful for many Twitter-based applications (e.g., truth discovery, trend analysis, search ranking, 
etc.)

## Challenges
K-means algorithm uses numerical values to compute distances and new centroids. However, tweets being texts needs additional processing to make it easier for K-means algorithm to perform clustering.

## Dataset
We are going to use the following dataset for this exercise: [Link](https://archive.ics.uci.edu/ml/datasets/Health+News+in+Twitter)

There about 16 datasets within this tweet collection. We have chosen `bbchealth.txt` for this project.

## How to run this code?
1. Install the repository using this command: `git clone https://github.com/StarRider/Tweet-Clustering-Using-Kmeans.git`
2. Go inside `Tweet-Clustering-Using-Kmeans` folder.
3. Using Python 3.11.5, run the following: `python src/model/kmeans.py`

Upon running the code, you would see the following output:
```
K:  3
SSE:  3431.8514306948496
Size of each cluster:
0: 3333 tweets
1: 15 tweets
2: 581 tweets
K:  5
SSE:  3263.8516658057274
Size of each cluster:
0: 2560 tweets
1: 54 tweets
2: 343 tweets
3: 680 tweets
...
```

## Results of K-means

<img src="https://github.com/StarRider/Tweet-Clustering-Using-Kmeans/assets/30108439/61781b49-8753-4cde-91ea-c58b89a896f8" width="750">
