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


# This construction allows two or more jobs to run simultaneously
if __name__ == "__main__":
    
    
    ''' Verify that output file name has been given '''
    
    
    try: file_out = sys.argv[1]
    except IndexError:
        print("No output file name given. " + 
              "Defaulting to gridsearch_classifier and " +
              "gridsearch_classifier_fulldata")
        file_out = "gridsearch_classifier"
    
    
    ''' Measure time '''
    
    
    start_time = time.time()
    print("\n--- Start time of script: {} ---\n".format(time.ctime()))

    
    
    ''' Load data '''


    with open("modified_renamed_optima_data_cleaned.pickle","rb") as f:
        data_frame = pickle.load(f)
         

         
    ''' Feature and response selection '''    


    # Features to use (pick one)
    
    # All features
    # features = data_frame.drop(["shock", "stroke", "mechanicalventilation", "chf", 
                                # "cardiogenicshock", "ventriculararrythmia", "atrialfibrillation", 
                                # "bradyarrythmia", "arrrythmia", "cardiacarrest", "timibleed", 
                                # "gibleed", "infection", "death", "any_complication"],
                                # axis = 1)

    
    # 6 features
    #features = data_frame[['peakcreat', 'age', 'angioplasty', 'STEMI', 'NSTEMI', 'unstable_angina']]
    
    # 3 features
    features = data_frame[['age', 'peakcreat', 'lvef']]
    
    
    print("Selected features: {}\n".format(list(features)))
    
    # Response variable (pick one)
    response = data_frame["death"]
    #response = data_frame["any_complication"]

    print("Selected response: {}\n".format(response.name))

    

    ''' Parameters of the supervised learning algorithm '''


    # Initialize type of classifier
    # (In this case, support vector machine, C-support vector)
    estimator = svm.SVC(max_iter = 10000)



    # Grid of parameters to test
    C_values     = np.logspace(-5,5,21)
    gamma_values = np.logspace(-8,-1,22)
    
    parameters = [{'kernel': ['rbf'],     'C': C_values, 'gamma': gamma_values, 'class_weight': ['balanced', None]},
                  {'kernel': ['linear'],  'C': C_values,                        'class_weight': ['balanced', None]},
                  {'kernel': ['sigmoid'], 'C': C_values,                        'class_weight': ['balanced', None]},
                  {'kernel': ['poly'],    'C': C_values, 'degree':(2,3,4),      'class_weight': ['balanced', None]}]
       


    # Number of folds to use in stratified cross-validation
    folds = 10;


        
    ''' Machine learning: Preprocessing '''


    # Get a list of column names since we will lose this information later
    feature_names = list(features)


    # Drop rows that are missing a response
    available_responses = response.apply(lambda x: x in [0,1])
    features = features[available_responses]
    response = response[available_responses]



    # Impute missing values
    imputer_object = preprocessing.Imputer(strategy = "mean").fit(features)
    features = imputer_object.transform(features)
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
    standardizer_object = preprocessing.StandardScaler().fit(features)
    features = standardizer_object.transform(features)
    


    ''' Machine learning: main '''        


    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, response, test_size=0.3)


    # Feed the classifier and parameters to the grid
    classifier = model_selection.GridSearchCV(estimator  = estimator, 
                                              param_grid = parameters, # see above for these two 
                                              scoring    = "roc_auc" , # want to maximize AUROC
                                              cv         = model_selection.StratifiedKFold(folds), # folds in stratified k-fold cross-validation
                                              verbose    = 1, # how much to output
                                              n_jobs     = 3 # do jobs in parallel
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
    report = metrics.classification_report(y_true, y_pred, digits = 3)
    print(report)                                     

    
    # Store all of the pickled results in file
    with open(file_out + "_fulldata.pickle","wb") as f:
        pickle.dump(classifier, f)
        pickle.dump(report, f)
        pickle.dump((X_train, X_test, y_train, y_test), f)
        pickle.dump(feature_names, f)
        pickle.dump(imputer_object, f)
        pickle.dump(standardizer_object, f)        
    
    
    # Store the bare-minimum needed to use the classifier elsewhere
    with open(file_out + ".pickle", "wb") as f:
        pickle.dump(classifier.best_estimator_, f)
        pickle.dump(imputer_object, f)
        pickle.dump(standardizer_object, f)


    
    ''' Finish measuring time '''
    
    
    elapsed_time = time.time() - start_time
    
    # Convert to hours, minutes, seconds
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    
    print("\nElapsed time (h:m:s):\n--- {:.0f}:{:02.0f}:{:02.0f} ---".format(h, m, s))

