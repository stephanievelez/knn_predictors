from matplotlib.pyplot import axis
import numpy as np, random, scipy.stats as ss

def majority_vote_fast(votes):
    mode, count = ss.mstats.mode(votes)
    return mode

def distance(p1, p2): #this is used to determine k-nn nearest neighbors
    return np.sqrt(np.sum(np.power(p2 - p1, 2)))

def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0]) #shape[0] is the how many rows there are in this array
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances) #returns the indices of the array
    return ind[:k]

def knn_predict(p, points, outcomes, k=5):
    ind = find_nearest_neighbors(p, points, k)
    return majority_vote_fast(outcomes[ind])[0]


def find_nearest_neighbors(p, points, k=5):
    distances = np.zeros(points.shape[0])
    for i in range(len(distances)):
        distances[i] = distance(p, points[i])
    ind = np.argsort(distances)
    return ind[:k]

import pandas as pd

data_folder="data"
data = pd.read_csv(data_folder + "/data.csv")
data["is_red"] = (data["color"] == "red").astype(int)
numeric_data = data.drop("color", axis=1)
numeric_data
group_by = numeric_data.groupby('is_red')
group_by_counter = group_by.count()

import sklearn.preprocessing
scaled_data = sklearn.preprocessing.scale(numeric_data)
numeric_data = pd.DataFrame(scaled_data, columns = numeric_data.columns)

numeric_data

import sklearn.decomposition
pca = sklearn.decomposition.PCA(n_components=2) #Principal component analysis (PCA)
principal_components = pca.fit_transform(numeric_data)
#principal_components.shape
x = principal_components[:,0] #first column
y = principal_components[:,1]

def accuracy(predictions, outcomes):
    return 100*np.mean(predictions == outcomes)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5) #the number of neighbors that will vote for the class of the target point
knn.fit(numeric_data, data['high_quality']) #knn.fit takes numeric_data as a training point against data from the high quality column
# Enter your code here!

library_predictions=knn.predict(numeric_data)
print(accuracy(library_predictions,data["high_quality"]))
#print(x,y)
print(principal_components)

n_rows = data.shape[0]
# Enter your code here.
random.seed(123)
selection=random.sample(range(n_rows),10)

predictors = np.array(numeric_data)
training_indices = [i for i in range(len(predictors)) if i not in selection]
outcomes = np.array(data["high_quality"])
p=predictors[selection]
my_predictions = knn_predict(p,predictors[training_indices,:],outcomes[training_indices],k=5)
percentage = accuracy(my_predictions,data.high_quality.iloc[selection])

print(predictors[training_indices,:])
print(numeric_data)