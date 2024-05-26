# imports

import os
import pandas as pd
import numpy as np
import warnings

from os import path
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from itertools import product
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

NEW_LINE = "\n"
SEPERATOR = "\n-----------------------------------\n"
warnings.filterwarnings('ignore') 

'''

DATA READ

'''

cur_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(cur_dir,"falldetection_dataset.csv")
df = pd.read_csv(path, header=None)

#print(df.head())
#print(SEPERATOR)


X_df = df.drop([0,1], axis=1)
y_df = df.iloc[:,1]
#print(X_df.head())
#print(NEW_LINE)
#print(y_df.head())
#print(SEPERATOR)

X = X_df.to_numpy() # X is 2D numpy array, each item is row / motor action
y = y_df.to_numpy() # Y is a 1D numpy array, each item is the label (F/NF)



''' PART A Functions '''

def plot_clusters(pcs, y_kmeans, num_clusters=2, figsize=(8, 6)):

    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, num_clusters))  # Generate a color map for the clusters

    for i in range(num_clusters):
        plt.scatter(pcs[y_kmeans == i, 0], pcs[y_kmeans == i, 1], s=50, c=[colors[i]], label=f'Cluster {i+1}')
    
    plt.xlabel('Data Projected to First PC')
    plt.ylabel('Data Projected to Second PC')
    plt.title('K-means Clustering on Projected Data')
    plt.legend()
    plt.show()

def calculate_majority_vote_accuracy(y, y_kmeans, num_clusters=2):
    label_map = {'F': 0, 'NF': 1}
    y_numeric = np.array([label_map[label] for label in y])

    # map cluster labels to the most frequent true label in each cluster
    cluster_labels = np.zeros_like(y_kmeans)
    for cluster in range(num_clusters):
        # find all items in this cluster
        mask = (y_kmeans == cluster)
        if np.sum(mask) == 0:  # If no points in the cluster, continue
            continue
        
        # count the occurrences of each class label in this cluster
        # return counts of unique elements sorted by element value
        labels, counts = np.unique(y_numeric[mask], return_counts=True)
        majority_label = labels[np.argmax(counts)]  # The label with the highest count
        
        # assign this majority label to all members of this cluster
        cluster_labels[mask] = majority_label

    # calculate and return the accuracy
    return accuracy_score(y_numeric, cluster_labels)


'''

PART A

'''

print("\n\n----------- PART A -----------\n")

# use PCA and extract the top 2 PCs
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
#print(pcs)

explained_variance_ratio = pca.explained_variance_ratio_
eig_vals = pca.explained_variance_
eig_vals_sorted = sorted(eig_vals, reverse=True)
cumulative_variance = np.cumsum(explained_variance_ratio)*100
print("Cumulative variance explained before any preprocessing")
print(cumulative_variance) # first PCA explains %75.3, first and second %83.82

plt.scatter(pcs[:, 0], pcs[:, 1])
plt.xlabel('Data Projected to First PC')
plt.ylabel('Data Projected to Second PC')
plt.title('Transformed Data')
plt.show() # not interpretable

# let's try to run K-mean with this setting

# Run K-means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(pcs)

accuracy = calculate_majority_vote_accuracy(y, y_kmeans)
print(f'Accuracy of the k-means before preprocessing (2 clusters): {accuracy:.4f}')
print(NEW_LINE)

# Plot the clusters
plot_clusters(pcs, y_kmeans)

# remove outliers
out_1= np.argmax(pcs[:,0])
out_2= np.argmax(pcs[:,1])

X_new = np.delete(X, out_1, axis=0)
X_new = np.delete(X_new, out_2, axis=0)
y_new = np.delete(y, out_1, axis=0)
y_new = np.delete(y_new, out_2, axis=0)

# scale the new data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_new)

pca_new = PCA(n_components=2)
pcs_new = pca_new.fit_transform(X_scaled)

explained_variance_ratio = pca_new.explained_variance_ratio_
eig_vals = pca_new.explained_variance_
eig_vals_sorted = sorted(eig_vals, reverse=True)
cumulative_var_ratio = np.cumsum(explained_variance_ratio)*100
print("Cumulative variance explained after preprocessing")
print(cumulative_var_ratio)

plt.scatter(pcs_new[:, 0], pcs_new[:, 1])
plt.xlabel('Data Projected to First PC')
plt.ylabel('Data Projected to Second PC')
plt.title('Transformed Data')
plt.show()

# Now re-run K-means clustering
num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(pcs_new)

accuracy = calculate_majority_vote_accuracy(y_new, y_kmeans)
print(f'Accuracy of the k-means after preprocessing (2 clusters): {accuracy:.4f}')

# plot the clusters
plot_clusters(pcs_new, y_kmeans)

# now we are ready to test with different k-values
for i in range(3,11):
    print(NEW_LINE)
    print(f"K-Means with {i} Clusters")
    kmeans = KMeans(n_clusters=i, random_state=123)
    y_kmeans = kmeans.fit_predict(pcs_new)
    accuracy = calculate_majority_vote_accuracy(y_new, y_kmeans, i)
    print(f'Accuracy of the k-means with {i} Clusters: {accuracy:.4f}')
    plot_clusters(pcs_new, y_kmeans, num_clusters=i)



print("\n\n----------- PART B -----------\n")

''' PART B Functions '''

# implements cross validation with Grid Search
def cross_validator(estimator, params):
    clf = GridSearchCV(estimator, params, cv=3)
    clf.fit(X_train, y_train)
    return clf.cv_results_

# print results 
def process_grid_search_results(report, results, mlp=False):
    display = []
    mlp_best = []
    idx = -1
    for i in results:
        idx += 1
        acc = report['mean_test_score'][i]
        params = report['params'][i]
        data = [f"{acc:.8f}"] 
        for key in sorted(params.keys()):
            data.append(f"{key}: {params[key]}")
            if mlp and idx == 0:
                mlp_best.append(params[key])
        display.append(' | '.join(data))
    if mlp:
        return display, mlp_best
    return display

'''

PART B

'''

print("\n------- B.1) SVM -------\n")

# first apply PCA to reduce the number of features that are to be used
pca = PCA(n_components=19)
pcs = pca.fit_transform(X_scaled)
print('Varince explained with first 19 PCs: %' + str(np.sum(pca.explained_variance_ratio_)*100)) 
print(NEW_LINE)

# map labels to binary nd perform the split
y_binary = np.where(y_new == 'F', 1, 0)
X_train, X_test, y_train, y_test = train_test_split(pcs, y_binary, test_size=0.3, random_state=123)

svm = SVC()
svm_poly = SVC()

params = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], 'kernel': ('linear', 'sigmoid', 'rbf'), 'gamma': ('scale', 'auto'), 'degree': [0]}
poly_params = {'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1], 'kernel': ['poly'],  'gamma':('scale', 'auto'), 'degree': [1,2,3,4,5]}

# perform 3-fold cross validation using Grid Search
cv_report = cross_validator(svm, params)
results_descending = np.argsort(cv_report['mean_test_score'])[::-1]

poly_cv_report = cross_validator(svm_poly, poly_params)
poly_results_descending= np.argsort(poly_cv_report['mean_test_score'])[::-1]

results = process_grid_search_results(cv_report, results_descending)
results += process_grid_search_results(poly_cv_report, poly_results_descending)

# print results
column_names = ["Accuracy"] + [key for key in sorted(cv_report['params'][0].keys())]
header_line = column_names[0] + "   | " + column_names[1] + "     | " + column_names[2] 
header_line += "     | " + column_names[3] + "      | " + column_names[4]
print(header_line)
print('-' * len(header_line)) 

# print each row
for row in sorted(results, reverse=True, key=lambda x: float(x.split('|')[0].strip())):
    print(row)


print("\n---- SVM BEST MODEL ----\n")

svm_best = SVC(C=0.5,gamma='auto',kernel='linear',degree=0)
svm_best = svm_best.fit(X_train, y_train)
svm_result = svm_best.predict(X_test)
print("Best SVM configuration: C=0.5, degree=0, gamma='auto', kernel='linear'" + "\n")
print(f"Final Accuracy of the best SVM model on Test Dataset: {metrics.accuracy_score(y_test, svm_result)}")


print("\n------- B.2) MLP -------\n")

mlp = MLPClassifier(max_iter=100000)
mlp_params = {'hidden_layer_sizes': [(2,2),(4,4),(8,8),(16,16),(32,32),(64,64)], 'solver' : ('sgd', 'adam'), 'alpha':[0.001, 0.005, 0.01, 0.05, 0.1], 'activation' : ('logistic','relu')}

mlp_cv_report = cross_validator(mlp, mlp_params)
mlp_results_descending = np.argsort(mlp_cv_report['mean_test_score'])[::-1]

mlp_results, mlp_best_config = process_grid_search_results(mlp_cv_report, mlp_results_descending, True)

# print results
column_names = ["Accuracy"] + [key for key in sorted(mlp_cv_report['params'][0].keys())]
header_line = column_names[0] + "   | " + column_names[1] + "       | " + column_names[2] 
header_line += "     | " + column_names[3] + "       | " + column_names[4]
print(header_line)
print('-' * len(header_line)) 

# print each row
for row in sorted(mlp_results, reverse=True, key=lambda x: float(x.split('|')[0].strip())):
    print(row)


print("\n---- MLP BEST MODEL ----\n")
print("Best MLP configuration: activation=" + mlp_best_config[0] + ", alpha=" + str(mlp_best_config[1]) 
      + ", hidden_layer_sizes=" + str(mlp_best_config[2]) + ", solver=" + mlp_best_config[3] + "\n")

mlp_best = MLPClassifier(activation=mlp_best_config[0], alpha=mlp_best_config[1], hidden_layer_sizes=mlp_best_config[2], solver=mlp_best_config[3], max_iter=100000)
mlp_best = mlp_best.fit(X_train, y_train)
mlp_result = mlp_best.predict(X_test)
print(f"Final Accuracy of the best MLP model on Test Dataset: {metrics.accuracy_score(y_test, mlp_result)}")
print(NEW_LINE)