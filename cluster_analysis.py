# This notebook deals with cluster analysis using both K Means Clsuter and
# Cosine Similarity

import pandas as pd
from sklearn.cluster import KMeans

wine = pd.read_csv("new_wine.csv")

# Gettting only wine data and excluding tatser name
# This way we will be able to make groups on wines only

wine_only = wine.iloc[:,1:671]



