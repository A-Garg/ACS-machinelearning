from __future__ import print_function
"""
Created on 2017-09-16

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Trains an SVC model on a combined dataset containing AMI-Quebec 2003 and ACS data.
The SVC model has parameters that were previously obtained by a gridsearch.

Input:  The combined dataset in the file 'combined_AMI_ACS.csv'.
Output: A classification report and the AUROC are printed to screen. 
        The classifier is pickled in the file 'ml_classifier.pickle'.
"""



''' Imports '''


import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file
import time # to measure time of execution
import sys # to accept command-line arguments

# scikit-learn version 0.18.2: tools for machine learning
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics


''' Functions '''


def AUROC_bootstrap_CI(y_test, y_score, interval = 0.95, n_bootstraps = 10000):
    """
    Calculates the confidence interval for the 
        area under a receiver operating curve statistic.
    Code is adapted from:
        https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals
        
    Arguments
        y_test:       the true value of the response
        y_score:      the predicted score for the response, given by the classifier
        interval:     the size of the confidence interval, default 0.95 (95%)
        n_bootstraps: the number of times to resample, default 1000
        
    Returns a tuple: (confidence_lower, confidence_upper)
    """

    #print("\nCalculating {}% confidence interval.".format(interval*100))
    #print("Bootstrapping with {} random samples.\n".format(n_bootstraps))
    
    # Convert lists to numpy arrays to allow better indexing
    y_test, y_score = np.array(y_test), np.array(y_score)
    
    bootstrapped_scores = []
    rng = np.random.RandomState()
    
    for i in (range(n_bootstraps)):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_score) - 1, len(y_score))
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = metrics.roc_auc_score(y_test[indices], y_score[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(np.array(bootstrapped_scores))
    
    # Compute the lower and upper bound of the confidence interval
    low  = (1 - interval)/2
    high = 1 - low
    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    
    return (confidence_lower, confidence_upper)


''' Load data '''


# Enter the file name containing the dataset within the quotes
file_name = "combined_AMI_ACS.csv"

# Enter the features within the quotes, 
    # the way they are encoded in the first line of the dataset
    # The columns must be in order: age, peak creatinine, LVEF
feature_columns = ["age", "peakcreat", "lvef"]

# Enter the outcome variable, e.g. "death" or "death5yr" or "cvdeat"
    # (the way it is encoded in the first line of the dataset)
response_column = ["inhospdeath", "timetodeath"]

# Read the dataset to memory
data = pd.read_csv(file_name, 
                   usecols    = feature_columns + response_column,
                   low_memory = False,
                   na_values  = [""," ","ND", "UNKNOWN"])

data.dropna(thresh = 3, inplace = True)                   



''' Machine learning: reprocessing '''


# Create feature and response data frames
features = data[['age','peakcreat','lvef']]
response = data["inhospdeath"]

# Drop rows that are missing a death response
available_responses = response.apply(lambda x: x in [0,1])
features = features[available_responses]
response = response[available_responses]

# Convert data from pandas to numpy array
features = features.as_matrix()
response = response.as_matrix()

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = model_selection.train_test_split(features, response, test_size=0.3)

# Impute missing values
imputer_object = preprocessing.Imputer(strategy = "mean").fit(X_train)
X_train = imputer_object.transform(X_train)

# Standardize the data
standardizer_object = preprocessing.StandardScaler().fit(X_train)
X_train = standardizer_object.transform(X_train)



''' Machine learning: train the classifier '''


# Initialize type of classifier
# (In this case, support vector machine, C-support vector)
classifier = svm.SVC(C            = 0.05,
                     kernel       = 'rbf',
                     gamma        = 0.04,
                     class_weight = 'balanced',
                     max_iter     = 10000, 
                     probability  = True)

# Make predictions on training data
classifier.fit(X_train, y_train)

print(classifier)

''' Machine learning: make predictions on the test set '''


# Impute missing values and standardize the test set
X_test = imputer_object.transform(X_test)
X_test = standardizer_object.transform(X_test)

# Make predictions on test data
y_pred = classifier.predict(X_test)
y_prob = [x[1] for x in classifier.predict_proba(X_test)]

# Calculate and print metrics
report = metrics.classification_report(y_test, y_pred, digits = 3)
print("\n" + report + "\n")

AUROC = metrics.roc_auc_score(y_test, y_prob)
CI_low, CI_high = AUROC_bootstrap_CI(y_test, y_prob)
print("Machine learning AUROC: {:.3f} [{:0.3f} - {:0.3f}]".format(
      AUROC, CI_low, CI_high))

      
''' Store classifier for later use '''      


# Store the bare-minimum needed to use the classifier elsewhere
with open("ml_classifier.pickle", "wb") as f:
    pickle.dump(classifier, f)
    pickle.dump(imputer_object, f)
    pickle.dump(standardizer_object, f)
