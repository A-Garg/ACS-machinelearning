# ACS-machinelearning
Create a machine learning model that is simpler and outperforms the GRACE score for acute coronary care.

## Quick start: Make predictions on another dataset
The key to making predictions on another dataset is `external_validation.py`.

(To be continued)

## Files
For each file, additional documentation is available within the file itself.

### clean_data.py

This file takes a CSV of data from the ACS-1 database. It cleans the data and stores it in a Pandas dataframe, which is written to a pickled file. Cleaning includes:
* Renaming some columns
* Changing string encodings to numbers (e.g. YES/NO to 1/0)
* Removing some obviously mis-entered numbers
* One-hot encoding categorical variables
* Adding summary response columns (e.g. a column for "any complication" that combines columns such as "shock", "stroke", and others using a logical OR).
* Removing redundant data

#### Usage
At command line, type `python clean_data.py [name of CSV file to clean]`

#### Applying this file to learn from another dataset
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
Outputs a results file containing serialized objects.

The parameters tested are:
* C
* gamma
* kernel
* class weight

The serialized objects (pickled) are, in order:
1. classifier (sklearn.model_selection.GridSearchCV object)
2. classification report (string from sklearn.metrics.classification_report)
3. train and test data (tuple of X_train, X_test, y_train, y_test)
4. feature names (list)

#### Usage
Select features to use by setting the variable `features` equal to a list of features from `data_frame`. For example: `features = data_frame[["age", "weight", "female"]]`

At command line, type `python gridsearch.py [name of output file]`

### plot_gridsearch.py
Plot a histogram of areas under receiver operating characteristic (AUROC) curve, as well as mean AUROC vs. each parameter.

At command line, enter `python plot_gridsearch.py`

### ROC_classification_report.py

Tests the best classifier of each gridsearch on its unseen test sets.
Produces an ROC classifying the best classifier based on which features are used.
Also outputs a classification report, containing precision and recall statistics.

#### Usage
One must first modify `file_list` and `label_list`. `file_list` should contain a list of files to use. These should all be outputs of `gridsearch.py`. `label_list` should contain human-readable labels describing what results are in each file, for example "Age only".

Once `file_list` and `label_list` are completed, run this program at command line by entering `python ROC_classification_report.py`

### GRACE_calculator.py
This script calculates the GRACE score for items in the cleaned ACS-1 database. The GRACE score is calculated from various characteristics as detailed in [Figure 4](http://jamanetwork.com/data/Journals/INTEMED/5461/ioi21057f4.png) of the article [*Predictors of Hospital Mortality in the Global Registry of Acute Coronary Events*](http://jamanetwork.com/journals/jamainternalmedicine/fullarticle/216232).

Once the script calculates the GRACE score, it calculates and produces its receiver operating characteristic (ROC). It then plots this ROC against those of machine learning classifiers as a comparison.

Because the ACS-1 database does not contain all of the variables that contribute to the GRACE score, some simplifying assumptions are made.

Four variables are ignored:
* History of congestive heart failure
* History of myocardial infarction
* Resting heart rate
* Systolic blood pressure

The cutoffs for an abnormal troponin level are set at:
* 0.01 ng/mL for troponin-T
* 0.02 ng/mL for troponin-I

#### Usage
Ensure that the `.pickle` file from `clean_data.py` is available. In the section of the program labelled `Load data`, type the name of this file. For the comparisons with the machine learning classifiers, the results from `gridsearch.py` must also be present, and it will be necessary to update the list `file_list` accordingly. 

Once done, run this file at command line using `python GRACE_calculator.py`. 


## Module Versions

These scripts are written in Python 2.7.13.

The following modules are used in the scripts:
* [pandas 0.20.2](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [numpy 1.13.1](https://scipy.org/install.html)
* [matplotlib 2.0.2](https://matplotlib.org/users/installing.html)
* [sklearn 0.18.2](http://scikit-learn.org/stable/install.html)
* [tqdm 4.15.0](https://pypi.python.org/pypi/tqdm)

All of these can be downloaded in on

## Acknowledgements
* Philippe Minh Tri Nguyen, for providing the ideas and code on which this repository is based
* Dr. Thao Huynh, for supervising this project
