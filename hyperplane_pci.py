#!/usr/bin/python
from __future__ import print_function
"""
Created on 2017-10-02

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Given a machine learning model with four features: age, peak creatinine, LVEF, and PCI
Returns two 3D subplots of survival vs. death over the three features, with or without PCI

Input:  ml_classifier.pickle
            The file that contains a pickled version of the machine learning model
Output: 3D plot
"""

import pickle # retrieve machine learning classifier
from tqdm import tqdm # create loading bars

# matplotlib, tools for plotting, version 2.0.2
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load machine learning model
with open("pci_classifier.pickle", "rb") as f:
    classifier   = pickle.load(f)
    standardizer = pickle.load(f)

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print("Generating and plotting points...")    
# Generate points to categorize
for age in tqdm(range(30,91,5)):
  for peakcreat in (range(20,601,20)):
    for lvef in range(0,81,5):
      
        # Case without PCI
        pci = 0
        # Machine learning: standardize inputs
        no_pci_features = standardizer.transform([[age, peakcreat, lvef, pci]])
        # Machine learning: make prediction and calculate probability
        no_pci_prediction = classifier.predict(no_pci_features)[0]
        
        
        # Case with PCI
        pci = 1
        # Machine learning: standardize inputs
        yes_pci_features = standardizer.transform([[age, peakcreat, lvef, pci]])
        # Machine learning: make prediction and calculate probability
        yes_pci_prediction = classifier.predict(yes_pci_features)[0]
        
        
        # Set colour based on if PCI improves things
        if   (no_pci_prediction == 1) and (yes_pci_prediction == 1): colour = 'yellow'
        elif (no_pci_prediction == 0) and (yes_pci_prediction == 0): colour = 'yellow'
        
        elif (no_pci_prediction == 1) and (yes_pci_prediction == 0): colour = 'red'
        elif (no_pci_prediction == 0) and (yes_pci_prediction == 1): colour = 'green'
        
        else: print("error")
        
        
        # Plot the point
        ax.scatter(age, peakcreat, lvef, c = colour)

# Set axis names
ax.set_title('PCI vs. no PCI change in survival')
ax.set_xlabel('Age')
ax.set_ylabel('Peak creatinine')
ax.set_zlabel('LVEF')

# Set legend labels
same       = mpatches.Patch(color = 'yellow', label = 'no change')
pci_worse  = mpatches.Patch(color = 'red',    label = 'PCI worse')
pci_better = mpatches.Patch(color = 'green',  label = 'PCI better')
ax.legend(handles = [same, pci_worse, pci_better])

print("\nShowing plot...")
plt.show()