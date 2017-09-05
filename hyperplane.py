#!/usr/bin/python
from __future__ import print_function
"""
Created on 2017-09-05

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Given a machine learning model with three features: age, peak creatinine, and LVEF
Returns a 3D plot of survival vs. death over the three features.

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
with open("ml_classifier.pickle", "rb") as f:
    classifier   = pickle.load(f)
    imputer      = pickle.load(f)
    standardizer = pickle.load(f)

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

print("Generating and plotting points...")    
# Generate points to categorize
for age in tqdm(range(30,101,10)):
  for peakcreat in tqdm(range(20,1500,50)):
    for lvef in range(0,81,10):

        # Machine learning: standardize inputs
        features = standardizer.transform([[age, peakcreat, lvef]])

        # Machine learning: make prediction and calculate probability
        prediction = classifier.predict(features)[0]
        
        # Set colour based on prediction
        if prediction == 1: colour = 'red' # death
        else: colour = 'blue' # survival
        
        # Plot the point
        ax.scatter(age, peakcreat, lvef, c = colour)

# Set axis names
ax.set_xlabel('Age')
ax.set_ylabel('Peak creatinine')
ax.set_zlabel('LVEF')

# Set legend labels
death    = mpatches.Patch(color = 'red',  label = 'death')
survival = mpatches.Patch(color = 'blue', label = 'survival')
ax.legend(handles = [death, survival])

print("\nShowing plot...")
plt.show()