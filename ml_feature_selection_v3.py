from __future__ import print_function
'''
Created on 2017-08-10

@author: Akhil Garg, akhil.garg@mail.mcgill.ca

Determines the most important features of a dataset.

Input:  cleaned pandas dataframe pickled in file 'modified_renamed_optima_data_cleaned.pickle'.
Output: plots and stdout of the most important features

'''



''' Imports '''


import numpy as np # matrix and math tools, version 1.13.1
import pandas as pd # data manipulation, version 0.20.2
import pickle # restore and write serialized data from/to file
import matplotlib.pyplot as plt # version 2.0.2
import tqdm # time for loops, version 4.15.0

# scikit-learn version 0.18.2: tools for machine learning
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import svm


def load_data(file_name):
    """
    Loads a pickled file containing cleaned data.
    
    file_name must be a pickled pandas dataframe object.
    
    Returns the dataframe.
    """

    print("Loading file {}.".format(file_name))
    with open(file_name, "rb") as f:
        data_frame = pickle.load(f)
    
    return data_frame

    
def manual_feature_response_selection(data, response = "death"):
    """
    Gives the opportunity to manually select features and response .
    
    data must be a pandas dataframe object containing both features and response.
    response can be one of "death" or "any_complication". Default is "death".
    
    Returns a tuple of two objects: (X, y).
        X is a pandas dataframe containing the features.
        y is a pandas series object containing the response.
    """    


    # Features to use (pick one)

    # Features from Canadian ACS Risk Score 
    ### Original parameters: age >= 75, Killip > 1, systolic BP < 100, HR > 100 bpm
    ### See more: https://www.ncbi.nlm.nih.gov/pubmed/23816022
    #X = data_frame[["age"]] # We only have age, need others

    # All features
    X = data_frame.drop(["shock", "stroke", "mechanicalventilation", "chf", 
                                "cardiogenicshock", "ventriculararrythmia", "atrialfibrillation", 
                                "bradyarrythmia", "arrrythmia", "cardiacarrest", "timibleed", 
                                "gibleed", "infection", "death", "any_complication"],
                                axis = 1)
                   


    # Response variable
    y = data_frame[response]
    
    return (X, y)


def preprocess(X, y):
    """
    Deletes missing responses, and imputes missing features.
    
    X is a pandas dataframe object containing the features.
    y is a list or vector-like object, such as a numpy array or pandas series.
    
    Returns a tuple of three objects (X, y, X_labels).
        X is the features matrix as a numpy matrix.
        y is the response vector as a numpy array.
        X_labels is a list of labels corresponding to each feature, in the order that was given.
    """ 

    print("Preprocessing data.")
    
    # Get column names since we lose this info later
    X_labels = list(features) 

    # Drop rows that are missing a response
    available_responses = y.apply(lambda z: z in [0,1])
    X = X[available_responses]
    y = y[available_responses]

    # Impute missing values
    X = preprocessing.Imputer(strategy = "mean").fit_transform(X)
    # For categorical data, rounding will convert imputed means back to 1/0 categories
    # Thus imputing categorical values as most common value
    # Rounding the floats (age, weight) to the nearest whole number will not significantly affect the results
    X = X.round()

    # Reshape response to feed to sklearn
    y = y.as_matrix()

    return (X, y, X_labels)


def variance_feature_selection(X, y, X_labels, min_variance = 0.16):  
    """
    Selects features that have a variance greater than a threshold.
    For example, considering a 1/0 (Bernoulli) variable,
        if at least 80% of a column is the same, 
        the variance is given as 0.8 * (1 - 0.8).
    More generally, the variance is p*(1-p) for a given column probability.
    
    X is the features matrix.
    y is the response vector.
    X_labels is a vector of labels corresponding to each feature, sorted in the same way as the features matrix.
    min_variance is a float. Features above this threshold are selected.
    
    Returns X_new, a matrix of only the selected features.
    """

    
    print("\nSelecting features with the highest variance.")
    print("Original number of features: {}".format(X.shape[1]))

    variance_select = feature_selection.VarianceThreshold(min_variance)
    X_new = variance_select.fit_transform(X)
    X_new_names = [feature for (feature, mask) in zip(X_labels, variance_select.get_support()) if mask]

    print("Minimum variance: {:0.2f}".format(min_variance))
    print("New number of features: {}".format(X_new.shape[1]))
    print("Remaining features: {}".format(X_new_names), end = "\n\n")
    
    return X_new
        

def correlation_feature_selection(X, y, X_labels, print_best = 10):    
    """ 
    Sorts features based on their univariate correlation with the response.
    Uses the Pearson correlation F score (ANOVA).
    # Code modified from http://scikit-learn.org/stable/auto_examples/feature_selection/plot_feature_selection.html
    
    X is the features matrix.
    y is the response vector.
    X_labels is a vector of labels corresponding to each feature, sorted in the same way as the features matrix.
    print_best is a positive integer indicating how many of the best features to print.
    
    Returns a tuple of (Fscores, Fscore_sorted_indices):
        Fscores is a numpy array of F scores for each feature, in the same order as given.
        Fscore_sorted_indices is a numpy array of indices. 
            If Fscores was sorted according to the indices given by Fscore_sorted_indices,
            Fscores would be in order from greatest to smallest.
    """


    selector = feature_selection.SelectKBest(feature_selection.f_classif)
    selector.fit(X, y)

    # Standardize, clean, and reverse sort scores
    Fscores = -np.log10(selector.pvalues_)
    Fscores[Fscores == np.inf] = np.nan
    Fscores /= np.nanmax(Fscores)
    Fscores = np.nan_to_num(Fscores)
    sorted_Fscores = np.sort(Fscores)[::-1]
    Fscore_sorted_indices = Fscores.argsort()[::-1]

    print("\n{} features with the highest Pearson correlation coefficients (F-scores):".format(print_best))
    print([X_labels[i] for i in Fscore_sorted_indices[:print_best]])

    return (Fscores, Fscore_sorted_indices)
    
    
def SVM_feature_selection(X, y, X_labels, print_best = 10):
    """ 
    Sorts features based on their weights in a linear support vector machine.
    
    X is the features matrix.
    y is the response vector.
    X_labels is a vector of labels corresponding to each feature, sorted in the same way as the features matrix.
    print_best is a positive integer indicating how many of the best features to print.
    
    Returns a tuple of (svm_weights, svm_sorted_indices):
        svm_weights is a numpy array of svm weights for each feature, in the same order as given.
        svm_sorted_indices is a numpy array of indices. 
            If svm_weights was sorted according to the indices given by svm_sorted_indices,
            svm_weights would be in order from greatest to smallest.
    """

    # Standardize data for sklearn (increases stability of SVM algorithm)
    X = preprocessing.StandardScaler().fit_transform(X)

    clf = svm.SVC(kernel = 'linear')
    clf.fit(X, y)

    svm_weights = (clf.coef_ ** 2).sum(axis=0)
    svm_weights /= svm_weights.max()
    svm_sorted_indices = svm_weights.argsort()[::-1]

    print("\n{} features with the highest SVM weights:".format(print_best))
    print([X_labels[i] for i in svm_sorted_indices[:print_best]])

    return(svm_weights, svm_sorted_indices)


def plot_correlation_SVM(X, Fscores, svm_weights, X_labels, X_indices, y = None):
    """
    Produces a plot of features vs. F score and SVM weight.
    
    X is the features matrix.
    Fscores is a vector of feature importances, as determined by F score,
        ordered in the same way as X .
    svm_weights is a vector of feature importances, as determined by a linear SVM model,
        ordered in the same way as X.
    X_labels is a vector of labels corresponding to each feature, sorted in the same way as the features matrix.
    X_indices is the order by which to sort the features on the plot.
        Normally, this would be ordered greatest to least F score or SVM weight.
    y is the response vector, and is ignored.
    """

    
    # Create subplots of feature selections
    fig, ax = plt.subplots()

    X_locations = np.arange(X.shape[-1])
    width = 0.35 # the width of the bars

    # Reorder scores and labels according to indices
    Fscores =     [Fscores    [i] for i in X_indices]
    svm_weights = [svm_weights[i] for i in X_indices]
    X_labels =    [X_labels   [i] for i in X_indices]
    
    F_bar   = ax.bar(X_locations, Fscores, width)
    SVM_bar = ax.bar(X_locations + width, svm_weights, width)

    ax.set_title('Feature importance')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Score')
    ax.set_xticks(X_locations + width / 2)
    ax.set_xticklabels(X_labels, rotation = "vertical")
    ax.legend((F_bar[0], SVM_bar[0]), ('F score', 'SVM weight'))
    plt.tight_layout()
    plt.show()


def RFE_SVC_CV(X, y, X_labels = None):  
    """
    Performs recursive feature elimination using a support vector machine and cross-validation (CV).
    At each level of recursion, 90% of the features are selected which produce the best CV score.
    Score is measured by area under receiver operating characteristic curve (AUROC).
    Output is a plot of the number of features vs. score.
    
    X is the features matrix.
    y is the response vector.
    X_labels is ignored.
    """

    print("Selecting features based on recursive feature elimination.")
    print("Using SVM with linear kernel, and scoring (AUROC) using cross-validation.")
    # Create the RFE object and compute a cross-validated score.
    svc = svm.SVC(kernel = "linear")
    rfecv = feature_selection.RFECV(estimator = svc, 
                                    step      = 0.9, # 0.9: each iteration, pick 90% of the best features
                                    cv        = model_selection.StratifiedKFold(10),
                                    scoring   = "roc_auc",
                                    n_jobs    = 3,
                                    verbose   = 1)
    rfecv.fit(X, y)

    print("\nOptimal number of features, based on recursive feature elimination using SVM: {}".format(rfecv.n_features_))

    # Plot number of features vs. cross-validation scores
    plt.figure()
    plt.title("AUROC vs. number of features,\nmeasured by recursive feature elimination and SVM model with cross-validation.")
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (AUROC)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()   
    

def L1_feature_selection(X, y, X_labels):
    """
    Uses L1 regularization (a.k.a. LASSO) to perform feature selection.
    Produces a list of features, from most important to least.
    Features are ordered by varying the C parameter of a linear support vector machine,
        and then seeing which features have non-zero coefficients.
        The C parameter is inversely proportional to the L1 regularization penalty.
        A higher penalty means more features have a coefficient of 0, and thus are not selected.
    
    X is the features matrix.
    y is the response vector.
    X_labels is a vector of labels corresponding to each feature, sorted in the same way as the features matrix.
    
    Returns feature_order, a list of features from most important to least.
    """
    
    print("Selecting features based on L1 regularization.")
    #print("\nNumber of features before L1 regularization: {}\n".format(X.shape[1]))

    C_values = np.logspace(-5,2,8)
    feature_order = []

    for C in C_values:

        lsvc = svm.LinearSVC(C = C, penalty = "l1", dual = False).fit(X, y) 
        model = feature_selection.SelectFromModel(lsvc, prefit = True)                    
        new_features = model.transform(X)
        
        new_feature_names = [X_labels[i] for i in model.get_support(indices = True)]
        
        for feature in new_feature_names:
            if feature in feature_order: continue
            else: feature_order.append(feature)
        
        print("Number of features after L1 regularization with C parameter {:.2E}: {}".format(C, new_features.shape[1]))
        #print("Selected features: {}".format(new_feature_names))

    print("Feature order according to L1 reg:")
    print(feature_order)
    return feature_order


def L2_feature_selection(X, y, X_labels):
    """
    Uses L2 regularization (a.k.a. ridge regression) to perform feature selection.
    Features are ordered by varying the C parameter of a linear support vector machine,
        and then selecting the most important coefficients (those with the highest absolute value).
    
    X is the features matrix.
    y is the response vector.
    X_labels is a vector of labels corresponding to each feature, sorted in the same way as the features matrix.
    
    Returns feature_order_matrix:
        Each row in feature_order_matrix corresponds to a C value.
            The order of C values is printed to stdout.
        Each column in feature_order_matrix corresponds to a rank.
            Higher-ranked features are more important.
    """

    print("\nSelecting features based on L2 regularization.\n")

    C_values = np.logspace(-3,5,9)
    feature_order_matrix = []

    for C in tqdm.tqdm(C_values): # tqdm displays a progress bar

        lsvc = svm.LinearSVC(C = C, penalty = "l2").fit(X, y) 
        
        abs_coefficients = np.absolute(lsvc.coef_)
        feature_indices  = np.argsort(abs_coefficients)[0][::-1]
        feature_order    = [X_labels[i] for i in feature_indices]
        feature_order_matrix.append(feature_order)

    np.set_printoptions(1)    
    print("\nC values: {}\n".format(C_values))    

    # Print 10 best features for each C value
    for i in range(1, 11):
        column = [row[i] for row in feature_order_matrix]
        print("Rank {:>2d}: {}".format(i, column))
    
    # Print most common feature over all C values at each rank
    print("\nMost common feature at each rank:")
    for i in range(1, 11):
        column = [row[i] for row in feature_order_matrix]
        mode = max(set(column), key = column.count)
        print("Rank {:>2d}: {}".format(i, mode))
        
    return feature_order_matrix
    
    

if __name__ == "__main__":
    
    data_frame = load_data("modified_renamed_optima_data_cleaned.pickle")
    features, response = manual_feature_response_selection(data_frame)
    features, response, feature_names = preprocess(features, response)
    
    # Below here, pick methods of selecting features as desired (by uncommenting [removing # at beginning of line])
    
    #variance_feature_selection(features, response, feature_names, min_variance = 0.16)
    #RFE_SVC_CV(features, response)
    #L1_feature_selection(features, response, feature_names)
    #L2_feature_selection(features, response, feature_names)

    
    # For the plot to work, the lines below here must be uncommented
    #Fscores,  Fscore_sorted_indices = correlation_feature_selection(features, response, feature_names)
    #svm_weights, svm_sorted_indices = SVM_feature_selection(features, response, feature_names)
    #plot_correlation_SVM(features, Fscores, svm_weights,
    #                     X_indices = svm_sorted_indices, # choose one of `Fscore_sorted_indices` or `svm_sorted_indices` (determines order of bars)
    #                     X_labels  = feature_names)    