# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load data, handle missing values, and extract relevant features for clustering.
2. Determine Clusters: Use the Elbow Method to find optimal number of clusters based on WCSS.
3. K-Means Clustering: Fit the K-Means model with optimal clusters, predict cluster labels for data.
4.Visualization: Plot data points with distinct colors for each cluster to visualize clustering results. 

## Program and Output:
```
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: HARIHARAN J
RegisterNumber: 212223240047
import pandas as pd 
import matplotlib.pyplot as plt 
data=pd.read_csv("Mall_Customers.csv")
data.head()
```
![image](https://github.com/user-attachments/assets/5a5445d9-8ac1-4a71-8d34-55ea7618d686)
```
data.info()
```
![image](https://github.com/user-attachments/assets/737c1c09-f215-426a-8947-88b15fcdcc85)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/34865351-f35b-4bc3-b920-1abaa3b9728f)

```
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No.of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
```
![image](https://github.com/user-attachments/assets/85dcfd46-bc7f-40f7-89e2-3f03d3e8723a)
```
km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
KMeans(n_clusters=5)
y_pred=km.predict(data.iloc[:,3:])
y_pred
```
![image](https://github.com/user-attachments/assets/d5026ffe-c7da-47fe-8af8-27fd617deb3b)
```
data["cluster"]=y_pred
df0 = data[data["cluster"]==0]
df1 = data[data["cluster"]==1]
df2 = data[data["cluster"]==2]
df3 = data[data["cluster"]==3]
df4 = data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"], color = "gold")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"], color = "pink")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"], color = "green")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"], color = "blue")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"], color = "red")
plt.show()
```
![image](https://github.com/user-attachments/assets/5eca2f62-9ecf-4226-bbd3-0b621edfdb2f)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
