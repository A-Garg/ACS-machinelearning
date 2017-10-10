#!/usr/bin/python
from __future__ import print_function
"""
Created on 2017-09-03

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Creates a website from which to use the machine learning model
    
Input:  ml_classifier.pickle
            The file that contains a pickled version of the machine learning model
        Age, Peak Creatinine, and LVEF entered through an HTML form
Output: Prediction of survival or death in hospital
"""

import cgi # website tools
import pickle # retrieve machine learning classifier

# Load machine learning model
with open("./ml_classifier.pickle", "rb") as f:
    classifier   = pickle.load(f)
    imputer      = pickle.load(f)
    standardizer = pickle.load(f)

    
# Start of HTML
print("""Content-Type: html

<html>
<head>
    <title>ACS Mortality Predictor</title>

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
<h2>ACS mortality predictor</h2>
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

        <form action="/ml_website.py" method="get">
        
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

    # Machine learning: standardize inputs
    features = standardizer.transform([[age, peakcreat, lvef]])

    # Machine learning: make prediction and calculate probability
    prediction = classifier.predict(features)
    probability_array    = classifier.predict_proba(features)[0]
    survival_probability = float(max(probability_array))

    # Convert 0/1 label to survival/death
    if prediction[0] == 0:
        prediction_text = "survival"
    else: prediction_text = "death"

    # End of HTML
    print("""
    <p>In-hospital mortality prediction: <b>{}</b><br>
    Certainty of prediction (between 50% and 100%, closer to 100% is better): {:.1f}%</p>
    """.format(prediction_text, survival_probability*100))

# If there's a ValueError, all of the data was not entered.
# When the user presses the submit button with the values entered,
    # the script will reload and run through the try block instead.
except ValueError: pass

print("""
<p><a href="https://github.com/A-Garg/ACS-machinelearning">Source code</a></p>
</body></html>""")