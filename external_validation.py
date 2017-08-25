from __future__ import print_function
"""
Created on 2017-08-23

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Tests a machine learning model on an external dataset.
New users: edit to the section "Variables specific to dataset: make modifications here"

Input:  
    1. dataset in CSV format
    2. machine learning model in pickle format
    
Output: ROCs of mortality predicted by GRACE score vs. machine learning
"""


''' Imports '''


import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file

# scikit-learn version 0.18.2, tools for machine learning
from sklearn.metrics import roc_curve, roc_auc_score, classification_report

# matplotlib version 2.0.2: tools for plotting
import matplotlib.pyplot as plt



''' Variables specific to dataset: make modifications here '''


# Enter the file name containing the dataset within the quotes
file_name = "amiquebec2003.csv"

# Enter the features within the quotes, 
    # the way they are encoded in the first line of the dataset
    # The columns must be in order: age, LVEF, peak creatinine, PCI
feature_columns = ["AGE", "peakcreat", "EJECTIONFRACTION"]

# Enter the outcome variable, e.g. "death" or "death5yr"
    # (the way it is encoded in the first line of the dataset)
response_column = ["death"]

# Enter the column that contains the pre-computed GRACE score
GRACE_column = ["GRACE"]



''' Load data '''


# Read the dataset to memory
data = pd.read_csv(file_name, 
                   usecols    = feature_columns + response_column + GRACE_column,
                   low_memory = False,
                   na_values  = [""," ","ND", "UNKNOWN"])

# Load machine learning model
with open("ml_classifier.pickle", "rb") as f:
    classifier   = pickle.load(f)
    imputer      = pickle.load(f)
    standardizer = pickle.load(f)

# Specify the features                
features = data[feature_columns]
print("Features: {}".format(list(features)))

# Specify the response
response = data[str(response_column[0])]
print("Response: {}\n".format(response.name))



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
    
    bootstrapped_scores = []
    rng = np.random.RandomState()
    
    for i in (range(n_bootstraps)):
        # Bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_score) - 1, len(y_score))
        if len(np.unique(y_test[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        score = roc_auc_score(y_test[indices], y_score[indices])
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(np.array(bootstrapped_scores))
    
    # Compute the lower and upper bound of the confidence interval
    low  = (1 - interval)/2
    high = 1 - low
    confidence_lower = sorted_scores[int(low * len(sorted_scores))]
    confidence_upper = sorted_scores[int(high * len(sorted_scores))]
    
    return (confidence_lower, confidence_upper)


    
''' Process data and apply the classifier'''


# Drop rows that are missing a response
available_responses = response.apply(lambda x: x in [0,1])
features = features[available_responses]
response = response[available_responses]

# Impute missing values
features = imputer.transform(features)
# For categorical data, rounding will convert imputed means back to 1/0 categories
# Thus imputing categorical values as most common value
# Rounding the floats (age) to the nearest whole number will not significantly affect the results
features = features.round()

# Reshape response to feed to machine learning model
response = response.as_matrix()

# Standardize the data
features = standardizer.transform(features)



''' Test the machine learning model on the data '''


prediction_scores = classifier.decision_function(features)
predictions       = classifier.predict(features)

FPR_ML, TPR_ML, thresholds_ML = roc_curve(response, prediction_scores)
ML_AUROC = roc_auc_score(response, prediction_scores)
ML_CI_low, ML_CI_high = AUROC_bootstrap_CI(response, prediction_scores)

print(classification_report(response, predictions))



''' Plot the ROCs '''


plt.figure()
# Plot a dashed line with slope 1 to represent "Luck"
# Any good ROC curve should lie above this one
plt.plot([0, 1], [0, 1], 
         label = "Luck (area = {:.3f})".format(0.5),
         linestyle='--')

# Plot the machine learning ROC
plt.plot(FPR_ML, TPR_ML, 
         label = "Machine learning (area = {:.3f}, CI = [{:0.3f} - {:0.3f}])".format(
            ML_AUROC, ML_CI_low, ML_CI_high))
print("Machine learning AUROC: {:.3f} [{:0.3f} - {:0.3f}]".format(
      ML_AUROC, ML_CI_low, ML_CI_high))

# Plot the GRACE ROC
GRACE_AUROC = roc_auc_score(response, data["GRACE"])
FPR_GRACE, TPR_GRACE, thresholds_GRACE = roc_curve(response, data["GRACE"])
GRACE_CI_low, GRACE_CI_high = AUROC_bootstrap_CI(response, data["GRACE"])

plt.plot(FPR_GRACE, TPR_GRACE, 
         label = "GRACE (area = {:.3f}, CI = [{:0.3f} - {:0.3f}])".format(
            GRACE_AUROC, GRACE_CI_low, GRACE_CI_high))
print("GRACE AUROC: {:.3f} [{:0.3f} - {:0.3f}]".format(
      GRACE_AUROC, GRACE_CI_low, GRACE_CI_high))
            
# Set plot parameters
plt.axis('scaled') # force 1:1 aspect ratio
plt.title("ROC of GRACE vs. machine learning\nResponse = {}".format(response_column[0]))
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
# Place legend in bottom right corner (since most action occurs top left)    
plt.legend(loc = "lower right")   

plt.show()
