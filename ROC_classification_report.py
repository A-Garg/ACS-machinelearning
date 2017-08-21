from __future__ import print_function

"""
Created on 2017-08-16

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Tests the best classifier of many gridsearches on their unseen test sets.
Produces an ROC classifying the best classifier based on which features are used.
Also outputs a classification report, containing precision and recall statistics.
"""


''' Imports '''


import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file
import itertools # to cycle through line styles

# scikit-learn version 0.18.2: tools for machine learning
from sklearn.metrics import roc_curve, roc_auc_score

# matplotlib version 2.0.2: tools for plotting
import matplotlib.pyplot as plt



''' Load data '''


# Modify file_list and label_list as new gridsearch results are produced
# Both lists must be in the same order
file_list  = ["gridsearch_results_weight_only.pickle",
              "gridsearch_results_age_only.pickle",
              "gridsearch_results_2_features.pickle",
              "gridsearch_results_3_features.pickle",
              "gridsearch_results_5_features.pickle",
              "gridsearch_results_8_features.pickle",
              "gridsearch_results_10_features.pickle",
              "gridsearch_results_all_features.pickle"]
              
label_list = ["Weight only", "Age only", 
              "2 features", "3 features", 
              "5 features", "8 features", "10 features",
              "All 74 features"] 
              
            


print("Plotting results from files:")
print(file_list)

# Empty lists to store data from files
classifiers   = []
reports       = []
sets          = []
feature_lists = []

# Get serialized data from files
for i in range(len(file_list)):
    with open(file_list[i], "rb") as f:
        classifiers.append(pickle.load(f))
        reports.append(pickle.load(f))
        
        X_train, X_test, y_train, y_test = pickle.load(f)
        sets.append((X_train, X_test, y_train, y_test))
        
        feature_lists.append(pickle.load(f))
    

    
''' Plot ROC '''


plt.figure()
# Plot a dashed line with slope 1 to represent "Luck"
# Any good ROC curve should lie above this one
plt.plot([0, 1], [0, 1], 
         label = "Luck (area = {:.3f})".format(0.5),
         linestyle='--')
plt.axis('scaled') # force 1:1 aspect ratio      
plt.title("ROC of various machine learning models")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")


# # Plot what the hypothetical GRACE 2.0 score curve would look like
# Deprecated because the curve is not ideal--the first part is too steep, and the rest too flat.
# x = np.linspace(0,1,100)
# GRACE_AUROC = 0.83
# # Use a root function as a smooth curve
# root = 1/(1/(GRACE_AUROC) - 1)
# y = x ** (1/root)
# area = np.trapz(y, x)
# plt.plot(x,y, label = "Hypothetical GRACE 2.0 (area = {:.3f})".format(area))


# Plot each ROC curve, corresponding to the best estimator from each gridsearch

# Pick one of the following two lines for plotting (all solid, or alternating)
line_styles = itertools.cycle(("-"))
#line_styles = itertools.cycle(("-", "-."))

for i in range(len(file_list)):

    X_test = sets[i][1]
    y_test = sets[i][3]
    y_score = classifiers[i].best_estimator_.decision_function(X_test)

    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    AUROC = roc_auc_score(y_test, y_score)
    plt.plot(fpr, tpr, 
             label = "{} (area = {:.3f})".format(label_list[i], AUROC),
             linestyle = line_styles.next())

# Place legend in bottom right corner (since most action occurs top left)    
plt.legend(loc = "lower right")

    
plt.show()



''' Print classification reports '''


for i in range(len(file_list)):
    print("Classification report for {}:".format(label_list[i]))
    print("Which includes features: " + str(feature_lists[i]))
    print(reports[i])
