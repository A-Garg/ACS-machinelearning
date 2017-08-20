from __future__ import print_function
"""
Created on 2017-08-10

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Takes data, tests C-support vector SVMs with different parameters for classification using a grid search.
Finds the best parameters for C-SVM.

Input:  cleaned pandas dataframe in file 'modified_renamed_optima_data_cleaned.pickle'.
Output: results of the grid search in gridsearch_results.pickle. 

"""



''' Imports '''


import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file
import time # to measure time of execution
import sys # to accept command-line arguments

# scikit-learn version 0.18.2: tools for machine learning
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import svm
from sklearn import metrics


# This construction allows two jobs to run simultaneously
if __name__ == "__main__":
    
    
    ''' Verify that output file name has been given '''
    
    
    try: file_out = sys.argv[1]
    except IndexError:
        print("No output file name give. Defaulting to gridsearch_results.pickle.")
        file_out = "gridsearch_results.pickle"
    
    
    ''' Measure time '''
    
    
    start_time = time.time()
    print("--- Start time of script: {} ---\n".format(time.ctime()))

    
    
    ''' Load data '''


    with open("modified_renamed_optima_data_cleaned.pickle","rb") as f:
        data_frame = pickle.load(f)
         

         
    ''' Feature and response selection '''    


    # Features to use (pick one)

    # Features from Canadian ACS Risk Score 
    ### Original parameters: age >= 75, Killip > 1, systolic BP < 100, HR > 100 bpm
    ### See more: https://www.ncbi.nlm.nih.gov/pubmed/23816022
    #features = data_frame[["age"]] # We only have age, need others

    
    # All features
    # features = data_frame.drop(["shock", "stroke", "mechanicalventilation", "chf", 
                                # "cardiogenicshock", "ventriculararrythmia", "atrialfibrillation", 
                                # "bradyarrythmia", "arrrythmia", "cardiacarrest", "timibleed", 
                                # "gibleed", "infection", "death", "any_complication"],
                                # axis = 1)
                   

    # 10 best features selected based on F-score
    # Removed "clearanceofcreatinine", "RADIAL", and "angioplasty" leaving behind only 7
    #features = data_frame[['bb', 'dapt', 'stat', 'peakcreat', 'age', 'cath', 'nadirhemoglobin']]
    
    
    # 8 features selected based on linear SVM weights
    # Removed variables that could potentially affect the outcome
    # features = data_frame[['inhospenox', 'platelets', 'age', 'clearanceofcreatinine', 'priorcvatia',
                          # 'INHOSPCABG', 'angioplasty', 'lvef']]

                           
    # 10 features selected based on F-score, SVM, and L2 regularization
    features = data_frame[['peakcreat', 'age', 'angioplasty', 'lvef', 'weight', 
                           'inhospenox', 'STEMI', 'NSTEMI', 'unstable_angina', 
                           'INHOSPCABG']]
                           
    
    # 4 features
    #features = data_frame[['peakcreat', 'age', 'angioplasty', 'weight']]
    
    
    
    print("Selected features: {}\n".format(list(features)))
    
    # Response variable (pick one)
    response = data_frame["death"]
    #response = data_frame["any_complication"]



    ''' Parameters of the supervised learning algorithm '''


    # Initialize type of classifier
    # (In this case, support vector machine, C-support vector)
    estimator = svm.SVC(max_iter = 10000)



    # Grid of parameters to test
    # Not testing polynomial kernel
    parameters = [{'C':            np.logspace(-5,5,21),  
                   'gamma':        np.logspace(-8,-1,22),
                   'kernel':       ['rbf', 'linear', 'sigmoid'],
                   'class_weight': ['balanced', None]
                 }]
       


    # Number of folds to use in stratified cross-validation
    folds = 10;


        
    ''' Machine learning: Preprocessing '''


    # Get  a list of column names since we will lose this information later
    feature_names = list(features)


    # Drop rows that are missing a response
    available_responses = response.apply(lambda x: x in [0,1])
    features = features[available_responses]
    response = response[available_responses]



    # Impute missing values
    features = preprocessing.Imputer(strategy = "mean").fit_transform(features)
    # For categorical data, rounding will convert imputed means back to 1/0 categories
    # Thus imputing categorical values as most common value
    # Rounding the floats (age, weight) to the nearest whole number will not significantly affect the results
    features = features.round()



    # Reshape response to feed to sklearn
    response = response.as_matrix()
    # Uncomment if DeprecationWarning about 1-D data
    # Breaks other things below
    #response = response.values.reshape(-1, 1)



    # Standardize data for sklearn
    # Unfortunately, this also standardizes 1/0 columns
    features = preprocessing.StandardScaler().fit_transform(features)


            
    ''' Machine learning: main '''        


    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, response, test_size=0.3)


    # Feed the classifier and parameters to the grid
    classifier = model_selection.GridSearchCV(estimator  = estimator, 
                                              param_grid = parameters, # see above for these two 
                                              scoring    = "roc_auc" , # want to maximize AUROC
                                              cv         = model_selection.StratifiedKFold(folds), # folds in stratified k-fold cross-validation
                                              verbose    = 1, # how much to output
                                              n_jobs     = 1 # do jobs in parallel
                                             )

    # Test the model on training data
    classifier.fit(X_train, y_train)

    # Make predictions on test data
    y_true, y_pred = y_test, classifier.predict(X_test)

    
    
    ''' Output the results and write to file '''

    
    # Print best parameters and result  
    print("\n")
    print("Best parameters: " + str(classifier.best_params_))
    print("Best area under ROC: {:0.3f}".format(classifier.best_score_))
    print("\n")

    
        
    # Print quick report containing precision, recall, F1 score
    report = metrics.classification_report(y_true, y_pred)
    print(report)                                     

    

    # Store the pickled results in file
    with open(file_out,"wb") as f:
        pickle.dump(classifier, f)
        pickle.dump(report, f)
        pickle.dump((X_train, X_test, y_train, y_test), f)
        pickle.dump(feature_names, f)
       

    
    ''' Finish measuring time '''
    
    elapsed_time = time.time() - start_time
    
    # Convert to hours, minutes, seconds
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print("\nElapsed time (h:m:s):\n--- {:.0f}:{:02.0f}:{:02.0f} ---".format(h, m, s))

