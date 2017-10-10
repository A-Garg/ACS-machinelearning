#!/usr/bin/python
from __future__ import print_function
"""
Created on 2017-10-02

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Creates a website from which to use the PCI machine learning model.
    
Input:  pci_classifier.pickle
            The file that contains a pickled version of the machine learning model
        Age, Peak Creatinine, and LVEF entered through an HTML form
Output: Prediction of how PCI changes survival
"""

import cgi # website tools
import pickle # retrieve machine learning classifier

# Load machine learning model
with open("./pci_classifier.pickle", "rb") as f:
    classifier   = pickle.load(f)
    standardizer = pickle.load(f)

    
# Start of HTML
print("""Content-Type: html

<html>
<head>
    <title>PCI Mortality Predictor</title>

    <!-- This container keeps the forms lined up -->
    <style type="text/css">
    .container {
        width: 200px;
        clear: both;
    }
    .container input {
        width: 100%;
        clear: both;
    }
    </style>
</head>

<body>
<h2>PCI mortality predictor</h2>
<p>Below, enter age (years), peak creatinine level (micromol/L), and left ventricle ejection fraction (%).</p>
""")

# Code to get form data
form = cgi.FieldStorage()

age       = form.getvalue("age")
peakcreat = form.getvalue("peakcreat")
lvef      = form.getvalue("lvef")


try:
    # Get form data
    print("""
    <div class="container">

        <form action="/ml_website_pci.py" method="get">
        
        <!-- Default values are those that were previously entered -->
        Age:             <input type="number" name="age" value="{}"><br>
        Peak creatinine: <input type="number" name="peakcreat" value="{}"<br><br>
        LVEF:            <input type="number" name="lvef" value="{}" />

        <input type="submit" value="Submit" />
        </form>
    </div>
    """.format(age, peakcreat, lvef))

    # Print data that was collected
    print("""
    Machine learning prediction based on:<br>
    Age: {}<br>
    Peak creatinine: {} micromol/L<br>
    LVEF: {}%
    """.format(age, peakcreat, lvef))
    
    for PCI in (0,1):
        
        # Machine learning: standardize inputs
        features = standardizer.transform([[age, peakcreat, lvef, PCI]])

        # Machine learning: make prediction and calculate probability
        prediction = classifier.predict(features)
        probability_array    = classifier.predict_proba(features)[0]
        survival_probability = float(max(probability_array))

        # Convert 0/1 label to survival/death
        if prediction[0] == 0:
            prediction_text = "survival"
        else: prediction_text = "death"
        
        if PCI == 0: 
            PCI_prediction     = prediction_text
            PCI_probability    = survival_probability
        else: 
            no_PCI_prediction  = prediction_text
            no_PCI_probability = survival_probability

    # End of HTML
    print("""
    <p>
      In-hospital mortality predictions:<br>
      Without PCI: <b>{}</b>, certainty = {:.1f}%</b><br>      
      With PCI: <b>{}</b>, certainty = {:.1f}%<br><br>
      
      Certainty of prediction is a number between 50% and 100%, closer to 100% is better.
    </p>
    """.format(PCI_prediction, PCI_probability*100,
               no_PCI_prediction, no_PCI_probability*100))

# If there's a ValueError, all of the data was not entered.
# When the user presses the submit button with the values entered,
    # the script will reload and run through the try block instead.
except ValueError: pass

print("""
<p><a href="https://github.com/A-Garg/ACS-machinelearning">Source code</a></p>
</body></html>""")