# ACS-machinelearning
Create a machine learning model that is simpler and outperforms the GRACE score for acute coronary care.

## Files
For each file, additional documentation is available within the file itself.

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
1. Create a list of desired column names, as they appear in the first row of the CSV. 
   
   Store them in the variable `column_names`. 
   
   For example: `column_names = ["age", "female", "weight", "ethnicity"]`

2. Create a dictionary of the pandas dtype of each column, except for floats. 
   
   Store this as the variable `column_types`. 
   
   For example: `column_types = {"female": "category", "ethnicity": "category"}`

3. Write the name of float types into the dictionary `float_converters`. 

   `float_converters` is a dictionary of each variable as a key and the `float_error` function as the value. 
   
   For example: `float_converters = {"age": float_error, "weight": float_error}`

4. Verify that cleaning step of the script is relevant to the particular dataset.

### feature_selection.py
This script contains various functions that assist in feature selection. It takes the cleaned dataset from `clean_data.py`, and outputs plots and text to the terminal.

#### Usage
One must select which feature selection methods to use within the script. At the bottom of the script, in the `if __name___ == "__main__:"` block, there are commented lines. Choose which feature selection methods to use by uncommenting them, i.e. removing the `#` at the beginning of the line.

The available methods are:
* Variance
* Recursive feature elimination with cross-validation
* L1 regularization
* L2 regularization
* Correlation
* Support vector machine model

Each method is described in further detail where it is defined in the script, or by printing `function.__doc__`

After selecting the desired methods, run `python feature_selection.py` at the command line.

Each function in this script can also be imported into other python scripts.

### gridsearch.py
Takes data that is outputted by `clean_data.py`, and finds the best parameters for classification.
The classification parameters apply to a support vector machine:
Each parameter combination is tested using 10-fold stratified cross-validation.

* C
* gamma
* kernel
* class weight

#### Usage
Select features to use by setting the variable `features` equal to a list of features from `data_frame`. For example: `features = data_frame[["age", "weight", "female"]]`

At command line, type `python gridsearch.py` followed by the name of an output file.

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
