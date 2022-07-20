from google.colab import files
data_to_load=files.upload()

import pandas as pd
import csv
import plotly.express as px
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
df=pd.read_csv("petals_sepals.csv")
print(df.head())

graph= px.scatter(df,x="petal_size",y="sepal_size")
graph.show()

from sklearn.cluster import KMeans
X=df.iloc[:,[0,1]].values
print(X)

WCSS=[]
for i in range(1,11):
  kMeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kMeans.fit(X)
  WCSS.append(kMeans.interia_)
