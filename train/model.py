"""Models that encaps all training process."""

# Authors: Wei-Kai Pan, Hsin-Pei Lin

import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# make choices for metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score, accuracy_score

# Utility
from utility import gettime

# Not used.
# from sklearn.metrics import mean_squared_error,
# from sklearn.linear_model import LogisticRegression
# from sklearn import metrics
# from sklearn.model_selection import PredefinedSplit
# import graphviz


class MLTask():
    """A class for a ML task."""

    def __init__(self, x, y, model_type='', model_name='', plot=True,
                 meta=None, cross_validate=False, standardized=False,
                 normalized=False, task_name='', params=None,
                 time_series=False, verbose=False, self_split=False,
                 estimators=10, depth=3, split=10, exp=False):
        """Init attributes."""
        self.model_type = model_type    # clf or reg.
        self.model_name = model_name    # RandomForest or DecisionTree
        # As well as name of directory under model_results

        self.plot = plot                # Plot coefficient.
        self.cv = cross_validate        # Gridsearch.
        self.X = x                      # Split data first.
        self.y = y                      # Expected pandas dataframe.
        self.task_name = task_name      # Will be as part of file name.
        self.params = params            # parameter used in gridsearch cv
        self.time_series = time_series  # Turn off train_test_split shuffle.
        self.feature_names = x.columns.values  # Assume that x is pandas df.
        self.normalized = normalized
        self.standardized = standardized
        self.verbose = verbose
        self.result = {}
        self.model = None                    # Save best performance model.
        self.cv_model = None                 # Save cross validation object.
        self.experiement_description = None  # Save optional description.
        self.summary = {}                    # {task_name: result{}}

        # Timer
        self.start_time = time.time()
        # Some non-reusable parts:
        self.self_split = self_split
        self.time_splits_ftr = [26638, 57510, 86153, 112408, 138742]
        self.time_period_ftr = [i for i in range(2002, 2007)]

        self.time_splits = [22370, 49009, 79881, 108524, 134779,
                            161113, 184593, 205668, 224758, 242409]
        self.time_period = [i for i in range(2001, 2011)]
        self.exp = exp

        # Self-defined parameter tuning
        self.estimators = estimators
        self.depth = depth
        self.split = split

    def batch(self):
        """Batch run total machine learning task."""
        print("= = = = = = = = = = = = = = = = = = =")
        print("Task '{}' Start".format(self.task_name))
        print("Start at {}".format(gettime(self.start_time)))

        if self.self_split:
            if self.exp:
                self.time_series_data_train_exp()
            else:
                self.time_series_data_train()
        else:
            X_train, X_test, y_train, y_test = self.data_split()
            model = self.train(X_train, y_train)
            self.evaluate(model, X_test, y_test)
            self.report_coefficients()
        pd.DataFrame.from_dict(self.summary).to_csv('./model_results/' +
                                                    self.model_name +
                                                    '/feature_info_' +
                                                    self.task_name + '_' +
                                                    str(self.estimators) + '_' +
                                                    str(self.depth) + '_' +
                                                    str(self.split) +
                                                    '_score.csv')
        print("Task {} End".format(self.task_name))
        print("= = = = = = = = = = = = = = = = = = =")
        return

    def time_series_data_train_exp(self):
        """For Year Prediction."""
        for i in range(len(self.time_splits_ftr) - 1):
            print("* * * * * * * * * * * * * * * * * * * * * * * *")
            print("Training year {}, Testing year {}".format(self.time_period_ftr[i],
                                                             self.time_period_ftr[i] + 1))
            start = self.time_splits_ftr[i]
            end = self.time_splits_ftr[i + 1]
            X = self.X[:end]
            y = self.y[:end]
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                shuffle=False,
                                                                train_size=start)

            print("train = {}, test = {}".format(X_train.shape, X_test.shape))

            model = self.train(X_train, y_train)
            self.evaluate(model, X_test, y_test, subscript=str(self.time_period_ftr[i]) +
                          str(self.time_period_ftr[i] + 1))

            if self.plot:
                self.plot_coefficients(subscript=str(self.time_period_ftr[i]) + '_' +
                                       str(self.time_period_ftr[i] + 1))
        return

    def time_series_data_train(self):
        """For Year Prediction."""
        for i in range(len(self.time_splits) - 1):
            print("* * * * * * * * * * * * * * * * * * * * * * * *")
            print("Training year {}, Testing year {}".format(self.time_period[i],
                                                             self.time_period[i] + 1))
            start = self.time_splits[i]
            end = self.time_splits[i + 1]
            X = self.X[:end]
            y = self.y[:end]
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                shuffle=False,
                                                                train_size=start)

            print("train = {}, test = {}".format(X_train.shape, X_test.shape))

            model = self.train(X_train, y_train)
            self.evaluate(model, X_test, y_test, subscript=str(self.time_period[i]) +
                          str(self.time_period[i] + 1))

            if self.plot:
                self.plot_coefficients(subscript=str(self.time_period[i]) + '_' +
                                       str(self.time_period[i] + 1))
        return

    def describe_exp(self, description, save_txt_path=None):
        """A function for tracing experiment steps."""
        self.experiement_description = description

        # Print experiment description.
        print(self.experiement_description)

        if save_txt_path is not None:
            with open(save_txt_path, 'w') as fout:
                fout.write(self.experiement_description)
            fout.close()
        return

    def select_model(self):
        """."""
        model_name = self.model_name
        if model_name == 'RandomForest':
            print("Selected Random Forest")
            return RandomForestClassifier(n_estimators=self.estimators,
                                          max_depth=self.depth,
                                          min_samples_split=self.split)
        elif model_name == 'DecisionTree':
            print("Selected Decision Tree")
            return tree.DecisionTreeClassifier(max_depth=self.depth)
        elif model_name == 'SVM':
            print("Selected SVM")
            return SVC()
        elif model_name == 'logistic_regression':
            print("Selected Logistic Regression")
            return
        else:
            print("No selected model to use.")
            return None

    def train(self, X_train, y_train):
        """A function for main training."""
        print("Trainig {}".format(self.model_name))
        print("Task Setting: %s" % self.task_name)

        # Print some meta data
        if self.verbose:
            print("X matrix = {}, y response = {}".format(self.X.shape,
                                                          self.y.shape))
            print("Model = {}\nParameters = {}\nFeatures = {}\n"
                  .format(self.model_name, self.params, self.feature_names))
        # Preprocessing
        self.preprocessing()

        if self.model_type == 'clf':
            model = self.classification(X_train, y_train)
        else:
            model = self.regression(X_train, y_train)

        self.model = model  # Save result in class.
        print("Training Ends, Total Elapsed {}".format(time.time() - self.start_time))
        return model

    def evaluate(self, model, X_test, y_test, subscript=''):
        """Calculate model scorer."""
        if self.model_type == 'clf':
            print("<Model: " + self.model_name + "Classifier>")
            accuracy = accuracy_score(model.predict(X_test), y_test)
            print("Accuracy = %.3f" % accuracy)
            auc = roc_auc_score(model.predict(X_test), y_test)
            print("Roc_Auc = %.3f" % auc)
            result = {}
            result['accuracy'] = accuracy
            result['roc_auc'] = auc
            self.summary[subscript] = result
        return

    def classification(self, X_train, y_train):
        """Classification training part."""
        clf = self.select_model()
        if clf is None:
            quit()

        if self.cv and self.params is not None:
            if self.time_series:
                # Assuming data is sorted by time first.
                ps = TimeSeriesSplit(n_splits=2).split(X_train)
            else:
                ps = None

            model = GridSearchCV(clf, self.params, cv=ps,
                                 scoring=make_scorer(accuracy_score),
                                 return_train_score=True)
            model.fit(X_train, y_train)
            self.cv_model = model  # Saved cross validation object.
            return model.best_estimator_
        else:
            # No cross validation
            clf.fit(X_train, y_train)
            return clf

    def regression(self, X_train, y_train):
        return

    def preprocessing(self):
        """For standardization, normalization, one hot encoding, dummify."""
        if self.standardized:
            scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)

            if self.verbose:
                print("Standarized data.")
                print

        if self.normalized:
            self.X = normalize(self.X, norm='l2')

            if self.verbose:
                print("Normalized data.")
        # TODO: dummy variables and one hot encode.
        return

    def data_split(self):
        """Split the data."""
        if self.time_series:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                shuffle=False,
                                                                test_size=0.3)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,
                                                                test_size=0.3)
        print("Train size = {}, Test size = {}".format(len(X_train),
                                                       len(X_test)))
        return X_train, X_test, y_train, y_test

    # Visualization.
    def cross_validation_log(self):
        """Print hyperparmeter results."""
        pd.set_option('display.max_rows', 100)
        df = pd.DataFrame(self.cv_model.cv_results_)
        df = df.sort_values(by=['meant_test_score'])
        df.to_csv('./model_results/' + self.model_name + '/' +
                  self.task_names + '_cv_log.csv')
        print("Saved " + self.model_name + " Cross Validation Results")
        return

    def report_coefficients(self, top_features=10, subscript=''):
        """A function for plotting results."""
        if self.model_name == 'RandomForest' or self.model_name == 'DecisionTree':
            coef = self.model.feature_importances_
        else:
            # Logistic Regression in this case.
            coef = self.model.coef_.ravel()

        if len(self.feature_names) < 2 * top_features:
            top_features = len(self.feature_names)

        # Arrange top coefficients
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients,
                                      top_positive_coefficients])
        # Save all numeric values
        all_coef = np.flip(np.argsort(np.abs(coef)), axis=0)

        df = pd.DataFrame()
        df['Top Features Names'] = self.feature_names[all_coef]
        df['Top Abs. Importance/ Coefficients'] = np.abs(coef[all_coef])
        df['Sign'] = np.sign(coef[all_coef])

        # modify this path or create the directory
        df.to_csv('./model_results/' + self.model_name + '/feature_info_' +
                  self.task_name + subscript + str(self.estimators) + '_' +
                  str(self.depth) + '_' + str(self.split) + '.csv')

        # Create plot
        if self.plot:
            plt.figure(figsize=(10, 15))

            colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
            plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
            feature_names = np.array(self.feature_names)
            plt.xticks(np.arange(1, 1 + 2 * top_features),
                       feature_names[top_coefficients], rotation=45, ha='right')
            plt.ylabel('Importance/ Coefficients')
            plt.title(self.task_name + '_' + self.model_name + '_' + subscript)
            plt.savefig('./model_results/' + self.model_name + '/plots/' +
                        self.task_name + '_' + subscript + str(self.estimators) +
                        '_' + str(self.depth) + '_' + str(self.split) + '.jpg')

        return
