# ACS-machinelearning
Create a machine learning model that is simpler and outperforms the GRACE score for acute coronary care.

# Table of contents
* [Quick start: Make predictions on another dataset](#quick-start--make-predictions-on-another-dataset)
  + [Before you start](#before-you-start)
  + [Ensuring the dataset is ready](#ensuring-the-dataset-is-ready)
  + [Tailoring the validation script to the dataset](#tailoring-the-validation-script-to-the-dataset)
  + [Running the script](#running-the-script)
* [More documentation: details about each file](#more-documentation--details-about-each-file)
  + [clean_data.py](#clean_datapy)
  + [feature_selection.py](#feature_selectionpy)
  + [gridsearch.py](#gridsearchpy)
  + [plot_gridsearch.py](#plot_gridsearchpy)
  + [ROC_classification_report.py](#roc_classification_reportpy)
  + [GRACE_calculator.py](#grace_calculatorpy)
  + [hyperplane.py](#hyperplanepy)
  + [external_validation.py](#external_validationpy)
* [Module Versions](#module-versions)
* [Contributor](#contributor)
* [Acknowledgements](#acknowledgements)



## Quick start: Make predictions on another dataset


### Before you start
Be sure that python 2.7.x is installed. python 3.x may work but probably won't. You additionally need to have these modules installed:

* [numpy](https://scipy.org/install.html)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [sklearn](http://scikit-learn.org/stable/install.html)
* [matplotlib](https://matplotlib.org/users/installing.html)

The dataset must be in CSV (comma-separated values) format. From this GitHub repository, download [external_validation.py](external_validation.py) and [ml_classifier.pickle](ml_classifier.pickle) into the same folder as the dataset. 

*How to download a file*: click on the file, then click "Raw" in the top right corner. Save this file into your working directory (e.g. by using 'CRTL + S').

### Ensuring the dataset is ready
The dataset should have at columns corresponding to the following five variables, in the listed format.

| Name                                              | Format                   |
| ------------------------------------------------- | ------------------------ |
| Age                                               | Number                   |
| Left ventricle ejection fraction                  | Number between 0 and 100 |
| Peak creatinine (or other creatinine measurement) | Number (in µmol/L)       |
| Mortality outcome                                 | 1 or 0                   |
| Calculated GRACE score                            | Number or probability    |


### Tailoring the validation script to the dataset

Open [external_validation.py](external_validation.py) using a text editor (e.g. Notepad, Notepad++). The part to edit is the section under `''' Variables specific to dataset: make modifications here '''`. There are four modifications to make here:

1. Edit the file name: change `amiquebec2003.csv` to the file the dataset you are testing. Be sure to leave the quotation marks around the file.
2. Enter the column names for the features that are used to make the predictions. **Important: the features must be in the same order as below.** The features are:
   1. Age
   2. Left ventricle ejection fraction
   3. Peak creatinine (or other creatinine)
   
   Each column name should be obtained from the first line of the CSV file. Replace `["AGE", "EJECTIONFRACTION", "peakcreat"]` as necessary. Be sure that each name is surrounded by quotation marks, and that the names are separated by spaces (i.e. don't change the current format). And again, *it is critical that the order of the features not be changed*. Otherwise, the machine learning model will be using the wrong numbers to make its predictions.
3. Enter the name of the column containing the outcome variable, usually death. If, for example, death was encoded as `death5yr` in the dataset, change `death` to `death5yr`. Be sure the square brackets and the quotation marks remain around the outcome variable.
4. Enter the name of the column containing the pre-computed GRACE score. This works the same way as modifying the outcome variable.

### Running the script
Run the python script [external_validation.py](external_validation.py). This can be done at command line--in the same folder, run `python external validation.py`. Or, it may be possible to simply double click the script to run it.

## More documentation: details about each script
For each file, additional documentation is available within the file itself.


### [clean_data.py](clean_data.py)

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

4. Verify that the proceeding cleaning steps of the script are relevant to the particular dataset.


### [feature_selection.py](feature_selection.py)
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


### [gridsearch.py](gridsearch.py)
Takes data that is outputted by [clean_data.py](clean_data.py), and finds the best parameters for classification.
The classification parameters apply to a support vector machine:
Each parameter combination is tested using 10-fold stratified cross-validation.
Outputs two results files containing serialized objects. The first, which ends with `_fulldata.pickle`, contains all of the data to reproduce the output of this file. The second, which is the same file name but without the `_fulldata`, contains just enough information to apply the best classifier from the gridsearch to a new problem.

The parameters tested are:
* C
* kernel
* class weight
* gamma, if kernel is RBF (radial basis function)
* degree, if kernel is polynomial

When tested on the AMI-optima dataset, the imputed values were the mean of available values:
* Age: 68.3
* Peak creatinine: 120.3
* LVEF: 60.0
For categorical variables, the most common category was imputed.

For the `fulldata.pickle` file, the serialized (pickled) objects are, in order:
1. all gridsearch classifiers (sklearn.model_selection.GridSearchCV object)
2. imputer (sklearn.preprocessing.Imputer object)
3. standardizer (sklearn.preprocessing.StandardScalar object)
4. classification report (string from sklearn.metrics.classification_report)
5. train and test data (tuple of X_train, X_test, y_train, y_test)
6. feature names (list)

For the other `.pickle` file, the serialized (pickled) objects are:
1. the best gridsearch classifier (sklearn.svm.SVC object)
2. imputer (sklearn.preprocessing.Imputer object)
3. standardizer (sklearn.preprocessing.StandardScalar object)

Because the order of the first three pickled objects is the same in both `.pickle` files, the `fulldata.pickle` can substitute for the condensed file if needed.

#### Usage
Select features to use by setting the variable `features` equal to a list of features from `data_frame`. For example: `features = data_frame[["age", "weight", "female"]]`

At command line, type `python gridsearch.py [name of output file without file extension]`.


### [plot_gridsearch.py](plot_gridsearch.py)
Plot a histogram of areas under receiver operating characteristic (AUROC) curve, as well as mean AUROC vs. each parameter.

At command line, enter `python plot_gridsearch.py`


### [ROC_classification_report.py](ROC_classification_report.py)

Tests the best classifier of each gridsearch on its unseen test sets.
Produces an ROC classifying the best classifier based on which features are used.
Also outputs a classification report, containing precision and recall statistics.

#### Usage
One must first modify `file_list` and `label_list`. `file_list` should contain a list of files to use. These should all be outputs of [gridsearch.py](gridsearch.py). `label_list` should contain human-readable labels describing what results are in each file, for example "Age only".

Once `file_list` and `label_list` are completed, run this program at command line by entering `python ROC_classification_report.py`


### [GRACE_calculator.py](GRACE_calculator.py)
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
Ensure that the `.pickle` file from [clean_data.py](clean_data.py) is available. In the section of the program labelled `Load data`, type the name of this file. For the comparisons with the machine learning classifiers, the results from [gridsearch.py](gridsearch.py) must also be present, and it will be necessary to update the list `file_list` accordingly. 

Once done, run this file at command line using `python GRACE_calculator.py`. 


### [hyperplane.py](hyperplane.py)
This script uses a 3-feature machine learning model to generate a three-dimensional plot of classifications.

The density of dots can be adjusted by modifying the last argument of the [range](https://docs.python.org/2/library/functions.html#range) functions in the script. The lower the number, the higher the density of points along that axis.

The script can be run at command line using `python hyperplane.py`.


### [external_validation.py](external_validation.py)
This script applies the machine learning model in [ml_classifier.pickle](ml_classifier.pickle) to a new dataset.

It takes in two files, a CSV of the dataset, as well as the pickled machine learning classifier. The script the returns receiver operating characteristic (ROC) curves for both the machine learning model and the GRACE score. 

In order to calculate confidence intervals for the areas under the ROCs, it bootstraps using 10000 random variables.

#### Usage
See above, [within the quick start guide](#tailoring-the-validation-script-to-the-dataset).

## Module Versions

These scripts are written in Python 2.7.13.

The following modules are used in the scripts:
* [pandas 0.20.2](https://pandas.pydata.org/pandas-docs/stable/install.html)
* [numpy 1.13.1](https://scipy.org/install.html)
* [matplotlib 2.0.2](https://matplotlib.org/users/installing.html)
* [sklearn 0.18.2](http://scikit-learn.org/stable/install.html)
* [tqdm 4.15.0](https://pypi.python.org/pypi/tqdm)

All of these (plus other modules) can be downloaded at once through an [Anaconda distribution](https://www.continuum.io/downloads). If using Anaconda, download the Python 2.7 version instead of the Python 3.x version.

## Contributor
These scripts and documentation were written by Akhil Garg. Do not hesitate to email me with questions. My email address can be found near the top of [external_validation.py](external_validation.py). 

## Acknowledgements
* Philippe Minh Tri Nguyen, for providing the ideas and code on which this repository is based
* Dr. Thao Huynh, for supervising this project
