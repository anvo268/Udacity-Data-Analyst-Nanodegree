import pandas as pd
import numpy as np
import json

# Allows me to "pretty print" dictionaries when assessing pipelines
from pprint import pprint

# Used for 'imputing' values for null data points
from sklearn.preprocessing import Imputer
# Implementation of SMOTE oversampling technique for balancing classes when training
from imblearn.over_sampling import SMOTE

# PCA implementation
from sklearn.decomposition import PCA
# Standardizes data
from sklearn.preprocessing import StandardScaler

# All classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

# Cross-validation techniques
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Evaluation metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Necessary to use SMOTE in the same pipeline as sklearn
from imblearn.pipeline import Pipeline as imb_pipeline

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Added later so results can be replicable
# Set to None and functions that use this will use their default random state
RANDOM_STATE = 1

# Variables
base_pipeline = [
                 ('imputer', Imputer(strategy='median')), 
                 ('resampling', SMOTE(random_state=RANDOM_STATE)),
                 ('selection', SelectKBest(score_func=f_classif)),
                 ('scaler', StandardScaler()), 
                 ('pca', PCA())
                ]

# Stuff we want to test for each model before doing careful tuning
base_param_grid = {
                  'scaler': [None, StandardScaler()], 
                  'selection__k': [7, 10, 15],
                  'pca': [None, PCA(n_components=2), PCA(n_components=4), PCA(n_components=6)],
                  }

financial_features = ['salary', 
                      'deferral_payments', 
                      'total_payments',
                      'loan_advances',
                      'bonus',
                      'restricted_stock_deferred',
                      'deferred_income', 
                      'total_stock_value',
                      'expenses',
                      'exercised_stock_options',
                      'other',
                      'long_term_incentive',
                      'restricted_stock',
                      'director_fees'] 

email_features = ['to_messages', 
                 'from_poi_to_this_person',
                 'from_messages', 
                 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] 

all_features = financial_features + email_features


# Functions
def load_data():
    """ Loads the data from a json file and returns dataframe """
    d = json.load(open('final_project_dataset.json', 'r'))

    out = []
    for k, v in d.items():
        v['name'] = k.title()
        s = pd.Series(v).apply(lambda x: np.nan if x == 'NaN' else x)
        out.append(s)

    data = pd.DataFrame(out)

    return data


def model_metrics(X, y, pipeline):
    """Prints out scoring metrics from cross-validation"""
    N = 5
    scoring = {'acr': 'accuracy',
               'prec': 'precision',
               'rec': 'recall',
               'f1': 'f1'}

    acr = np.array([])
    prec = np.array([])
    rec = np.array([])
    f1 = np.array([])
    # Repeat N times for more consistent results
    for i in range(N):
        scores = cross_validate(pipeline, X, y,
            scoring=scoring,
            cv=RepeatedStratifiedKFold(n_splits=3, random_state = RANDOM_STATE),)
        acr = np.append(acr, scores['test_acr'])
        prec = np.append(prec, scores['test_prec'])
        rec = np.append(rec, scores['test_rec'])
        f1 = np.append(f1, scores['test_f1'])

    print('SCORES:\n')
    print('Average Accuracy:', acr.mean())
    print('Average Precision: ', prec.mean())
    print('Average Recall:', rec.mean())
    print('Average F1: ', f1.mean())


def show_confusion_matrix(X, y, model):
    """Plots the confusion matrix for the model as a heatmap"""
    N = 10

    tot_mat = np.array([[0, 0],
                        [0, 0]])
    # Repeat N times for more consistent results
    for i in range(N):
        # Making the confusion matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=RANDOM_STATE)

        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        tot_mat += confusion_matrix(y_test, predicted)

    sns.heatmap(tot_mat.T, square=True, annot=True, fmt='d', cbar=False,
     xticklabels=('Reg', 'Poi'), yticklabels=('Reg', 'Poi'))
    plt.xlabel('true label')
    plt.ylabel('predicted label')


def print_model_info(grid):
    """Prints out the parameters for a pipeline"""
    print('Pipeline Parameters:')
    pprint(grid.best_params_)
    print()

    best_model = grid.best_estimator_
    pca_used = grid.best_params_['pca'] != None
    
    k = grid.best_params_['selection__k']
    scores = pd.Series(best_model.named_steps['selection'].scores_, index=all_features).sort_values(ascending=False)

    print('Retained Features:\n')
    for f in scores[:k].keys():
        print(f)

    if pca_used:
        print("Number of principal components retained:", best_model.named_steps['pca'].n_components)


def evaluate_model(model, param_grid, X, y, optimize='f1'):
    """Gives a full evaluation of a given model"""
    grid = GridSearchCV(model, param_grid, scoring=optimize, 
    	cv=RepeatedStratifiedKFold(n_splits=3, random_state=RANDOM_STATE))
    grid.fit(X, y);
    model = grid.best_estimator_
    
    print()
    model_metrics(X, y, model)   
    print()

    show_confusion_matrix(X, y, model)

    print_model_info(grid)

    return grid

# Can't believe this is necessary
def convert_to_dict(dataset):
    """Converts a dataframe to dictionary for tester.py"""
    out = {}
    for i, row in dataset.iterrows():
        key = row['name']
        out[key] = dict(row)
    
    return out


def remove_outliers(data):
    """Returns data free from specified outliers. (Just makes final code neater)"""
    data = data[data.name != 'The Travel Agency In The Park']
    data = data[data.name != 'Lockhart Eugene E']
    data = data[data.name != 'Total']

    return data

