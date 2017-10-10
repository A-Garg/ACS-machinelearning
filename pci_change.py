from __future__ import print_function
"""
Created on 2017-10-02

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Trains an SVC model on a combined dataset containing AMI-Quebec 2003 and ACS data.
The model contains the features age, peak creatinine, LVEF, and PCI.

Input:  The combined dataset in the file 'combined_AMI_ACS.csv'.
Output: A classification report and the AUROC are printed to screen. 
        The classifier is pickled in the file 'pci_classifier.pickle'.
"""



''' Imports '''


import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file
import time # to measure time of execution
import sys # to accept command-line arguments

# scikit-learn version 0.18.2: tools for machine learning
from sklearn import pipeline
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics



# This construction allows two or more jobs to run simultaneously
# Good to have if doing a gridsearch with cross-validation
if __name__ == "__main__":
    
    
    ''' Measure time '''
    
    
    start_time = time.time()
    print("\n--- Start time of script: {} ---\n".format(time.ctime()))
    
    
    ''' Load data '''

    
    # Enter the file name containing the dataset within the quotes
    file_name = "combined_AMI_ACS.csv"

    # Enter the features within the quotes, 
        # the way they are encoded in the first line of the dataset
        # The columns must be in order: age, peak creatinine, LVEF
    feature_columns = ["age", "peakcreat", "lvef", "pci"]

    # Enter the outcome variable, e.g. "death" or "death5yr" or "cvdeat"
        # (the way it is encoded in the first line of the dataset)
    response_column = ["inhospdeath"]

    # Read the dataset to memory
    data = pd.read_csv(file_name, 
                       usecols    = feature_columns + response_column,
                       low_memory = False,
                       na_values  = [""," ","ND", "UNKNOWN"])

    # Drop columns with missing data                  
    data.dropna(inplace = True)                   



    ''' Machine learning: preprocessing '''


    # Create feature and response data frames
    features = data[['age','peakcreat','lvef','pci']]
    response = data["inhospdeath"]

    # Convert data from pandas to numpy array
    features = features.as_matrix()
    response = response.as_matrix()

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, response, test_size = 0.3)

    # Standardize the data
    standardizer_object = preprocessing.StandardScaler().fit(X_train)
    X_train = standardizer_object.transform(X_train)



    ''' Machine learning: train the classifier '''


    # Initialize type of classifier
    # (In this case, support vector machine, C-support vector)
    estimator = svm.SVC(C            = 0.1,
                        kernel       = 'rbf',
                        gamma        = 0.1,
                        class_weight = 'balanced',
                        max_iter     = 10000, 
                        probability  = True)
                         
    # # Grid of parameters to test
    # C_values     = np.logspace(-5,5,21)
    # gamma_values = np.logspace(-10,-1,10)

    # #parameters = [{'C': C_values, 'gamma': gamma_values, 'class_weight': ['balanced', None]}]
    
    # # After running a grid search, I found these parameters to be the best:
    # parameters = [{'C': [0.1], 'gamma': [0.1], 'class_weight':['balanced']}]

    # gridsearch = model_selection.GridSearchCV(estimator  = estimator, 
                                              # param_grid = parameters, # see above for these two 
                                              # scoring    = "roc_auc" , # want to maximize AUROC
                                              # cv         = model_selection.StratifiedKFold(5), # folds in stratified k-fold cross-validation
                                              # verbose    = 1, # how much to output
                                              # n_jobs     = 3 # do jobs in parallel
                                             # )

    # Make predictions on training data
    estimator.fit(X_train, y_train)

    # Store the best estimator as a separate variable
    #best_estimator = gridsearch.best_estimator_
    best_estimator = estimator
    
    print(best_estimator)


    
    ''' Machine learning: make predictions on the test set '''


    # Standardize the test set
    X_test = standardizer_object.transform(X_test)

    # Make predictions on test data
    y_pred = best_estimator.predict(X_test)
    y_prob = [x[1] for x in best_estimator.predict_proba(X_test)]

    # Calculate and print metrics
    report = metrics.classification_report(y_test, y_pred, digits = 3)
    #print("\n" + report + "\n")

    AUROC = metrics.roc_auc_score(y_test, y_prob)
    print("Machine learning AUROC: {:.3f}".format(AUROC))



    ''' Store classifier for later use '''      


    # Store the bare-minimum needed to use the classifier elsewhere
    with open("pci_classifier.pickle", "wb") as f:
        pickle.dump(best_estimator, f)
        pickle.dump(standardizer_object, f)
        
    
    
    ''' Finish measuring time '''
    
    
    elapsed_time = time.time() - start_time
    
    # Convert to hours, minutes, seconds
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print("\nElapsed time (h:m:s):\n--- {:.0f}:{:02.0f}:{:02.0f} ---".format(h, m, s))

