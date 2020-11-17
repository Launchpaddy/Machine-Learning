
library(datasets)
data = state.x77


################################################
#Agglomerative Hierarchical Clustering
################################################
distance = dist(as.matrix(data))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)





################################################
#Using k-means Clustering
###############################################

# Cluster into k=5 clusters:
myClusters = kmeans(data, 3)

# Summary of the clusters
summary(myClusters)

# Centers (mean values) of the clusters
myClusters$centers

# Cluster assignments
myClusters$cluster

# Within-cluster sum of squares and total sum of squares across clusters
myClusters$withinss
myClusters$tot.withinss


# Plotting a visual representation of k-means clusters
library(cluster)
clusplot(data, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


################################################
#Dropping "AREA"
###############################################


#data1 <- filter(data=="Frost")
library(datasets)
data = state.x77


newData <- subset(data, select = -c(8))

################################################
#Scaling data (normalizing)
################################################
data_scaled = scale(newData)

################################################
#Agglomerative Hierarchical Clustering
################################################
distance = dist(as.matrix(data_scaled))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)









################################################
#Dropping "all but FOST
###############################################
frostData <- subset(data_scaled, select = -c(1:6))

################################################
#Agglomerative Hierarchical Clustering for Frost
################################################
distance = dist(as.matrix(frostData))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)








################################################
#Scaling data (normalizing)
################################################
data_scaled = scale(newData)


################################################
#Agglomerative Hierarchical Clustering
################################################
distance = dist(as.matrix(data_scaled))

# now perform the clustering
hc = hclust(distance)

# finally, plot the dendrogram
plot(hc)











#data1 <- filter(data=="Frost")
library(datasets)
data = state.x77


################################################
#Scaling data (normalizing)
################################################
data_scaled = scale(data)


################################################
#Using k-means Clustering
###############################################

# Cluster into k=5 clusters:
myClusters = kmeans(data_scaled, 3)

# Summary of the clusters
summary(myClusters)

# Centers (mean values) of the clusters
myClusters$centers



# Within-cluster sum of squares and total sum of squares across clusters
myClusters$withinss
myClusters$tot.withinss


# Plotting a visual representation of k-means clusters
library(cluster)
clusplot(data, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)


table = NULL;
for (i in 1:25) {
  data = kmeans(data_scaled, i)
  table[i] = data$tot.withinss
}
plot(table)


# Cluster into k=4 clusters:
myClusters = kmeans(data_scaled, 4)

# Cluster assignments
myClusters$cluster

library(cluster)
clusplot(data, myClusters$cluster, color=TRUE, shade=TRUE, labels=2, lines=0)

myClusters$centers
