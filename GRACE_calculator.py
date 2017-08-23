from __future__ import print_function
"""
Created on 2017-08-22

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Calculates the GRACE score for rows in a dataset.
How the GRACE score is calculated: figure 4 of
http://jamanetwork.com/journals/jamainternalmedicine/fullarticle/216232


Input:  cleaned pandas dataframe in file 'modified_renamed_optima_data_cleaned.pickle'.
Output: receiver operating characteristic (ROC) for the GRACE score
"""


''' Imports '''

import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file
import re # regex

# scikit-learn version 0.18.2, for plotting an ROC
from sklearn.metrics import roc_curve, roc_auc_score
# to convert floats to classes
from sklearn.preprocessing import LabelBinarizer

# matplotlib version 2.0.2: tools for plotting
import matplotlib.pyplot as plt


''' Load data '''


with open("modified_renamed_optima_data_cleaned.pickle","rb") as f:
    data = pickle.load(f)
    

    
''' Create helper columns to assist with scoring '''


# Divide age into ranges of 10 and convert to score
age_bins  = (0,40,50,60,70,80,90,np.inf)
age_score = (0,18,36,55,73,91,100)
data["age_score"] = pd.cut(data["age"],
                           bins   = age_bins,
                           labels = age_score,
                           right  = False)
# Convert the categories into numbers
data["age_score"] = pd.to_numeric(data["age_score"])


# Divide creatinine into bins and convert to score
# Multiply by 88.4 to convert from mg/dl to micromol/L
creat_bins  = [88.4*x for x in (0,0.4,0.8,1.2,1.6,2,4,np.inf)]
creat_score = (1,3,5,7,9,15,20)
data["creat_score"] = pd.cut(data["baselinecreat"],
                             bins   = creat_bins,
                             labels = creat_score,
                             right  = False)
# Convert the categories into numbers
data["creat_score"] = pd.to_numeric(data["creat_score"])


# Score troponin, assuming abnormal values:
    # Troponin T > 0.01 ng/mL
    # Troponin I > 0.02 ng/mL
# Of troponin T and troponin I, if either is elevated, add 15 to the GRACE score
high_trop = (data["TnT"] > 0.01) | (data["TnI"] >= 0.02)
data["troponin_score"] = np.where(high_trop, 15, 0)


# If both TnT and TnI are NaN, replace the troponin score with NaN
trop_isnan = (np.isnan(data["TnT"])) & (np.isnan(data["TnI"]))
data["troponin_score"] = np.where(trop_isnan, np.nan, data["troponin_score"])


# Score presence of ST depression
    # Assume that NSTEMI is equivalent to ST segment depression
    # (since we don't have ST depression data)
data["STdepression_score"] = np.where(data["NSTEMI"] == 1, 11, 0)
# Convert missing data back to NaN
STdepression_isnan = np.isnan(data["NSTEMI"])
data["STdepression_score"] = np.where(STdepression_isnan, np.nan, data["STdepression_score"])


# Score percutaneous coronary intervention
data["PCI_score"] = np.where(data["angioplasty"] == 0, 14, 0)
# Convert missing data back to NaN
PCI_isnan = np.isnan(data["angioplasty"])
data["PCI_score"] = np.where(PCI_isnan, np.nan, data["PCI_score"])



''' Calculate the score '''


data["GRACE_score"] = data["age_score"]          \
                    + data["STdepression_score"] \
                    + data["creat_score"]        \
                    + data["troponin_score"]     \
                    + data["PCI_score"]
                    

                    
''' Generate an ROC '''


# Drop rows that are missing a response
available_responses = data["death"].apply(lambda x: x in [0,1])
kept_data = data[available_responses]
# Drop rows that are missing a GRACE score
available_scores = kept_data["GRACE_score"].apply(lambda x: not np.isnan(x))
kept_data = kept_data[available_scores]

# Convert death data to a format that sklearn likes
response = LabelBinarizer().fit_transform(kept_data["death"])


AUROC = roc_auc_score(response, kept_data["GRACE_score"])
fpr, tpr, thresholds = roc_curve(response, kept_data["GRACE_score"])
print(AUROC)



''' Plot ROC '''


plt.figure()
# Plot a dashed line with slope 1 to represent "Luck"
# Any good ROC curve should lie above this one
plt.plot([0, 1], [0, 1], 
         label = "Luck (area = {:.3f})".format(0.5),
         linestyle='--')

# Plot the ROC         
plt.plot(fpr, tpr, label = "GRACE (area = {:.3f})".format(AUROC))
plt.axis('scaled') # force 1:1 aspect ratio      
plt.title("ROC of GRACE vs. machine learning")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")  



''' Compare GRACE with two machine learning models '''


# Only compare two machine learning models for clarity
file_list = ["gridsearch_results_3_features.pickle", 
             "gridsearch_results_6_features.pickle"]

for file in file_list:
    with open(file, "rb") as f:
        classifier = pickle.load(f).best_estimator_
        report     = pickle.load(f)
        X_train, X_test, y_train, y_test = pickle.load(f)
        feature_list = pickle.load(f)
    
    # Calculate data for plotting ROC
    y_score = classifier.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    AUROC = roc_auc_score(y_test, y_score)
    
    # Get the number from the file name, and use it as the label
    label = "ML {} features (area = {:.3f})".format(re.search("\d+", file).group(0), AUROC)
    plt.plot(fpr, tpr, label = label)
 
# Place legend in bottom right corner (since most action occurs top left)    
plt.legend(loc = "lower right")   
plt.show()



