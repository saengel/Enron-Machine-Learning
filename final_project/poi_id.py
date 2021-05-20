#!/usr/bin/python

import sys
import pickle
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'total_payments', 'bonus', 'expenses', "percent_to_email_poi", "percent_from_email_poi"] # You will need to use more features

### Load the dictionary containing the dataset
import pickle
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

### Task 2: Remove outliers
df = pd.DataFrame.from_dict(data_dict)
df = df.replace(to_replace='NaN', value=pd.NA)
df = df.dropna(axis=1, subset=['from_poi_to_this_person', 'from_this_person_to_poi','to_messages','from_messages'])
df = df.replace(to_replace=pd.NA, value='NaN')
print(df)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = df.to_dict()

for each in my_dataset:
    cur = my_dataset[each]
    if 'from_poi_to_this_person' in cur and 'from_this_person_to_poi' in cur:
        if cur['to_messages'] != 0 and cur['from_messages'] != 0:
            percent_to_email_poi = int(cur['from_poi_to_this_person']) / int(cur['to_messages'])
            percent_from_email_poi = int(cur['from_this_person_to_poi']) / int(cur['from_messages'])
            my_dataset[each]["percent_to_email_poi"] = percent_to_email_poi
            my_dataset[each]["percent_from_email_poi"] = percent_from_email_poi
    # print(cur)


# print(my_dataset['METTS MARK'])

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Metrics Imports
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

metrics = []

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf = make_pipeline(PCA(n_components=2), GaussianNB())
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

nb_accuracy = accuracy_score(pred, labels_test)
nb_f1 = f1_score(pred, labels_test, average=None)
metrics.append({"Algorithm": "Naive Bayes", "Accuracy": nb_accuracy, "F1 Score": nb_f1})

# K-Means
from sklearn.cluster import KMeans
clf = make_pipeline(PCA(n_components=2), KMeans(n_clusters=2, random_state=3))
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
kmeans_accuracy = accuracy_score(pred, labels_test)
kmeans_f1 = f1_score(pred, labels_test, average=None)
metrics.append({"Algorithm": "K-Means Clustering", "Accuracy": kmeans_accuracy, "F1 Score": kmeans_f1})

# Decision Tree
from sklearn import tree
clf = make_pipeline(PCA(n_components=2), tree.DecisionTreeClassifier())
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
dtree_accuracy = accuracy_score(pred, labels_test)
dtree_f1 = f1_score(pred, labels_test, average=None)
metrics.append({"Algorithm": "Decision Tree", "Accuracy": dtree_accuracy, "F1 Score": dtree_f1})

# SVM
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto'))
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
svm_accuracy = accuracy_score(pred, labels_test)
svm_f1 = f1_score(pred, labels_test, average=None)
metrics.append({"Algorithm": "SVM", "Accuracy": svm_accuracy, "F1 Score": svm_f1})

df = pd.DataFrame.from_dict(metrics)
print(df)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)