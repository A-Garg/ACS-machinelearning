# ACS-machinelearning
Create a machine learning model that is simpler and outperforms the GRACE score for acute coronary care.

## Files
### clean_data.py

This file takes a CSV of data from an ACS-1 database. It cleans the data and stores it in a Pandas dataframe, which is written to a pickled file. Cleaning includes:
* Renaming some columns
* Changing string encodings to numbers (e.g. YES/NO to 1/0)
* Removing some obviously mis-entered numbers
* One-hot encoding categorical variables
* Adding summary response columns (e.g. a column for "any complication" that combines columns such as "shock", "stroke", and others using a logical OR).
* Removing redundant data

#### Usage
At command line, type `python clean_data.py [name of CSV file to clean]`

#### Applying this file to another dataset
This script is specific to the optima ACS-1 dataset. In order to adapt it to another dataset, one must:
1. Create a list of desired column names, as they appear in the first row of the CSV. Store them in the variable `column_names`. For example: `column_names = ["age", "female", "weight", "ethnicity"]`

2. Create a dictionary of the pandas dtype of each column, except for floats. Store this as the variable `column_types`. For example: `column_types = {"female": "category", "ethnicity": "category"}`

3. Write the name of float types into the dictionary `float_converters`. `float_converters` is a dictionary of each variable as a key and the `float_error` function as the value. For example: `float_converters = {"age": float_error, "weight": float_error}`

4. Verify that cleaning step of the script is relevant to the particular dataset.

### feature_selection.py


### gridsearch.py


### plot_gridsearch.py


### ROC_classification_report.py

## Module Versions

These scripts are written in Python 2.7.13.

The following modules are used in the scripts:
* pandas 0.20.2
* numpy 1.13.1
* matplotlib 2.0.2
* sklearn 0.18.2
* tqdm 4.15.0

## Acknowledgements
* Philippe Minh Tri Nguyen, for providing the ideas and code on which this repository is based
* Dr. Thao Huynh, for supervising this project
