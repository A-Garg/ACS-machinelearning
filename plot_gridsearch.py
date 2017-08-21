from __future__ import print_function

"""
Created 2017-08-10

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Takes results of ml_v4_gridsearch.py and plots parameters (x-axis) vs. 
area under receiver operating characteristic curve (AUROC) 
that is obtained by cross-validation.

Input:  pickled results containing the classifier, and classification report
    The classifier is an sklearn.model_selection.GridSearchCV object.
    The classification report is an sklearn.metrics.classification_report object.
    
Output: plots of parameters vs. cross-validated AUROC.
"""


''' Imports '''


import matplotlib.pyplot as plt # version 2.0.2
import pandas as pd # version 0.20.2
import pickle
import math
import sys



''' Load data '''


try:
    file_name = sys.argv[1]
except IndexError:
    print("No filename provided. Defaulting to gridsearch_results.pickle")
    file_name = "gridsearch_results.pickle"
    
with open(file_name, "rb") as f:
    classifier = pickle.load(f)
    report     = pickle.load(f)
    
# Convert results to pandas data frame
results = pd.DataFrame(classifier.cv_results_)


   
''' Plot of all AUROCs using cross-validation '''


# Set threshold of results to view: [threshold, 1]
threshold = 0.5

# Count number of good results (over threshold)
# Condition returns series of boolean values, where True == 1
# Summing number of Trues gives count
over_threshold = sum(results.mean_test_score > threshold) 
N_obs  = results.shape[0]


# All cross-validation AUROCs
plt.figure(1)
plt.title("Histogram of cross-validated AUROC scores over " +
          "{} ({}/{} parameter combinations)".format(threshold, over_threshold, N_obs))
plt.xlabel("AUROC")
plt.ylabel("Count") 
plt.hist(results["mean_test_score"], range = (threshold, 1), bins = 50)



''' Plots of cross-validated AUROC vs. each individual parameter '''


# Share y-axis among subplots
# Create 4 subplots
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows = 2, ncols = 2, sharey = "row")
fig.suptitle("Mean cross-validated AUROC vs. various parameters (N = {})".format(N_obs)) # title for all four subplots
fig.subplots_adjust(hspace = 0.5, top = 0.85) # increase space between titles


# Plot kernel vs. test score
kernel_means  = results["mean_test_score"].groupby(results["param_kernel"]).mean()
kernel_names  = list(kernel_means.index)
kernel_values = [value for value in kernel_means[kernel_names]]
ax0.set_title("kernel")
ax0.bar(range(len(kernel_values)), kernel_values, tick_label = kernel_names) 
ax0.set_ylim(threshold,1) # restrict bar graph to higher values


# Plot class weight vs. test score
# Replace "None" with "unbalanced", since None confuses pandas
results.replace(to_replace = [None], value = "unbalanced", inplace = True)
weight_means  = results["mean_test_score"].groupby(results["param_class_weight"]).mean()
weight_names  = list(weight_means.index)
weight_values = [value for value in weight_means[weight_names]]
ax1.set_title("class weight")
ax1.bar(range(len(weight_values)), weight_values, tick_label = weight_names)


# Plot gamma vs. test score
gamma_means  = results["mean_test_score"].groupby(results["param_gamma"]).mean()
gamma_names  = list(gamma_means.index)
gamma_values = [value for value in gamma_means[gamma_names]]
ax2.set_title("gamma")
ax2.plot(gamma_names, gamma_values)
ax2.set_xscale("log")
ax2.set_ylim(threshold,1)


# Plot gamma vs. test score
C_means  = results["mean_test_score"].groupby(results["param_C"]).mean()
C_names  = list(C_means.index)
C_values = [value for value in C_means[C_names]]
ax3.set_title("C")
ax3.plot(C_names, C_values)
ax3.set_xscale("log")


plt.show()