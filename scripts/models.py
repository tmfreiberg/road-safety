import pandas as pd
import numpy as np
from typing import Type, Union, Optional, Tuple, Callable, List, Dict, Any
import copy
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    fbeta_score,
    matthews_corrcoef,
)
from sklearn.model_selection import GridSearchCV
from joblib import dump
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from evaluation import custom_confusion, weighted_average_f, confusion_matrix_with_weighted_fbeta, prediction
from processing import process
from variables import class_codes, inverse_class_codes
from pathlib import Path

import utils
from utils import display, clear_output, print_header

import copy
import json

import time
import itertools
from collections import OrderedDict
import joblib


# BEFORE THE MODEL CLASS, HERE ARE SOME USEFUL FUNCTIONS

# For making trivial predictions
def trivial(
        y_train: Union[np.ndarray, pd.DataFrame, pd.Series],  # targets from training set
        class_codes: Union[
            dict, None
        ] = None,  # or dictionary with items for form original_class_name : class_code
        class_probs: str = "zero_one",  # or 'proportion', or 'uniform'
        pos_label_code: Union[int, None] = None,  # code of desired positive label
        pos_label: Union[
            int, str
        ] = "majority_class",  # or 'minority_class', or a key in class_codes (i.e. original_class_name)
        num_preds: Union[int, None] = None,  # number of trivial predictions to be made
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # We'll deal with numpy arrays only...
    if isinstance(y_train, (pd.DataFrame, pd.Series)):
        y_train_arr = y_train.values
    else:
        y_train_arr = y_train

    # We assume the target is 1-dimensional
    if np.ravel(y_train_arr).shape[0] != y_train_arr.shape[0]:
        return

    # If class_codes dictionary is not given, we'll re-create it
    if class_codes is None:
        classes = np.unique(y_train_arr)
        try:
            classes.sort()
        except:
            pass
        class_codes = {
            classes_element: idx for idx, classes_element in enumerate(classes)
        }

    # Create dictionary whose items are of the form cc : f, where cc is a class code and f is its frequency
    # Initialize the dictionary with frequencies set to zero
    value_counts_dict = dict.fromkeys(class_codes.keys(), 0)

    # Produce list of values and list of counts, where values[i] has frequency counts[i]
    values, counts = np.unique(y_train_arr, return_counts=True)

    # For values (class codes) with count (frequency) at least one, update the corresponding dictionary item
    for value, count in zip(values, counts):
        value_counts_dict[value] = count

    # Find the maximum frequency among class code frequencies
    M = max([v for k, v in value_counts_dict.items()])

    # The first class code with this frequency will be called the majority class
    majority_class_code = next(
        (k for k, v in value_counts_dict.items() if v == M), None
    )

    # We now set the prediction_code, i.e. the class code of the trivial prediction we want to make
    if pos_label_code is not None and pos_label_code in class_codes.values():
        prediction_code = pos_label_code
    elif type(pos_label) == str and "maj" in pos_label:
        pos_label = "majority_class"
        prediction_code = majority_class_code
    elif type(pos_label) == str and "min" in pos_label:
        pos_label = "minority_class"
        m = min([v for k, v in value_counts_dict.items() if v != 0])
        minority_class_code = next(
            (k for k, v in value_counts_dict.items() if v == m), None
        )
        prediction_code = minority_class_code
    else:
        try:
            prediction_code = class_codes[pos_label]
        except:
            pos_label = "majority_class"
            prediction_code = majority_class_code

    # Produce an array of desired dimensions, with the trivial prediction code
    if num_preds is None:
        num_preds = y_train.shape[0]

    trivial_prediction_code = np.full(num_preds, fill_value=prediction_code)

    # And do the same but with the unencoded prediction
    inverse_class_codes = {v: k for k, v in class_codes.items()}
    trivial_prediction = np.full(
        num_preds, fill_value=inverse_class_codes[prediction_code]
    )

    # Now for the probabilities
    # Certain choices for class_probs are incompatible with choices for pos_label
    # E.g. class_probs = 'proportion' is incompatible with pos_label = 'minority_class'
    if "p" in class_probs and pos_label == "minority_class":
        class_probs = "zero_one"  # could also be 'uniform'

    # Initialize the probabilities array
    trivial_probabilities = np.zeros((num_preds, len(class_codes)))

    # Fill in the probabilities
    if "z" in class_probs:
        class_probs = "zero_one"
        trivial_probabilities[:, prediction_code] = 1
    elif "p" in class_probs:
        class_probs = "proportion"
        for c, cc in class_codes.items():
            trivial_probabilities[:, cc] = value_counts_dict[cc] / y_train_arr.shape[0]
    else:
        class_probs = "uniform"
        trivial_probabilities += 1 / len(class_codes)

    return trivial_prediction, trivial_prediction_code, trivial_probabilities


# For balancing a training set
def balance(X: pd.DataFrame,
            y: pd.DataFrame,
            X_supp: Union[None, pd.DataFrame] = None,
            y_supp: Union[None, pd.DataFrame] = None,
            max_repeats: int = 1,
            random_seed: Union[None, int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Balances the class distribution in a dataset by upsampling classes in `y` to a maximum limit
    determined by the highest class frequency or a specified multiplier `max_repeats`. Optionally
    incorporates supplementary data to augment before balancing.

    Parameters:
    - X (pd.DataFrame): The feature set.
    - y (pd.DataFrame): The target variable DataFrame, with exactly one column.
    - X_supp (pd.DataFrame, optional): Supplementary feature set to concatenate with X before balancing.
    - y_supp (pd.DataFrame, optional): Supplementary target set to concatenate with y before balancing.
    - max_repeats (int, optional): The maximum multiplier to apply to the class frequencies during upsampling.
      If `max_repeats` is 1, the function returns the original datasets without upsampling.
    - random_seed (int, optional): A seed for the random number generator to ensure reproducibility of the sampling process.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the upsampled feature and target DataFrames.

    Raises:
    - ValueError: If the input `y` or `y_supp` does not contain exactly one column.

    Example:
    ```python
    balanced_X, balanced_y = balance_test(X_train, y_train, X_supp=X_additional, y_supp=y_additional,
                                          max_repeats=2, random_seed=42)
    ```
    """
    # Early return if no upsampling is required
    if max_repeats == 1:
        return X, y

    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure y has exactly one target column
    target = y.columns[0]
    if len(y.columns) != 1:
        raise ValueError("The input `y` and `y_supp` must contain exactly one column.")

    # Calculate class frequencies and determine the maximum size each class should reach
    value_counts_dict = {k: v for k, v in y[target].value_counts().items()}
    M = max(value_counts_dict.values())
    upsampled_class_sizes_dict = {k: min(M, v * max_repeats) for k, v in value_counts_dict.items()}

    # Optionally concatenate supplementary data
    if X_supp is not None and y_supp is not None:
        X_ = pd.concat([X, X_supp])
        y_ = pd.concat([y, y_supp])
    else:
        X_ = X
        y_ = y

    # Recalculate class frequencies after potential data augmentation
    value_counts_dict_ = {k: v for k, v in y_[target].value_counts().items()}
    upsample_quotient_dict = {}
    upsample_remainder_dict = {}
    for k, a in upsampled_class_sizes_dict.items():
        b = value_counts_dict_[k]
        q, r = divmod(a, b)
        upsample_quotient_dict[k] = q
        upsample_remainder_dict[k] = r

    # Generate indices for repeated sampling according to calculated quotients
    repeated_indices = np.concatenate(
        [np.repeat(idx, upsample_quotient_dict.get(y_.loc[idx, target], 1)) for idx in X_.index]
    )

    # Initialize list for storing all indices including those sampled stochastically
    aggregate_indices = [repeated_indices]
    for k, r in upsample_remainder_dict.items():
        filtered_y_ = y_[y_[target] == k]
        sampled_df = filtered_y_.sample(n=r, replace=False)
        sampled_indices = sampled_df.index
        aggregate_indices.append(sampled_indices)

    # Concatenate all indices and shuffle to mix upsampled and sampled indices
    shuffled_indices = np.concatenate(aggregate_indices)
    np.random.shuffle(shuffled_indices)

    # Index into the augmented datasets using shuffled indices to form balanced datasets
    X_out = X_.loc[shuffled_indices]
    y_out = y_.loc[shuffled_indices]

    return X_out, y_out


# HERE IS THE model CLASS

class model:
    def __init__(
            self,
            data: Type[process] = None,  # processed data, instance of 'process' class
            folds: Union[int, None] = None,
            impute_strategy: Union[dict, None] = None,
            classifier: Union[
                None,
                DecisionTreeClassifier,
                GradientBoostingClassifier,
                LogisticRegression,
                RandomForestClassifier,
                XGBClassifier,
            ] = None,
            balance: Union[int, None] = None,
            grid_search_scoring: Union[None, str, Callable] = None,
            param_grid: Union[dict, None] = None,
            filename_stem: Union[str, None] = None,
            model_dir: Union[Path, None] = None,
            beta: Union[np.ndarray, None] = None,
            weights: Union[np.ndarray, None] = None,
            threshold_dict_help: Union[None, OrderedDict[str, float]] = None,
            threshold_dict_hinder: Union[None, OrderedDict[str, float]] = None,
            stop_before_preprocessing: Union[bool, None] = None,
    ) -> None:

        self.data = data
        self.folds = folds
        self.impute_strategy = impute_strategy
        self.classifier = classifier
        self.balance = balance
        self.grid_search_scoring = grid_search_scoring
        self.param_grid = param_grid
        self.filename_stem = filename_stem
        self.model_dir = model_dir
        self.preprocessor = None
        self.beta = beta
        self.weights = weights
        self.threshold_dict_help = threshold_dict_help
        self.threshold_dict_hinder = threshold_dict_hinder
        self.stop_before_preprocessing = stop_before_preprocessing

        # New attributes
        self.class_codes = copy.deepcopy(class_codes)
        self.inverse_class_codes = copy.deepcopy(inverse_class_codes)

        # SEPARATE FEATURES X FROM TARGETS y (X_train, y_train, X_test, y_test)
        print_header("Separating features from targets")
        print("self.X_train/self.X_test, self.y_train/self.y_test")
        self.separate()

        # MAP VALUES (ORDINAL AND CATEGORICAL)
        # E.g. 0,1,2,9 -> 0,1,2,3; 1,2,9 -> 1,2,3, <50, 50, 60, 70, 80, 90, 100 -> 0, 5, 6, 7, 8, 9, 10, etc.
        # Mappings given in 'class_codes' dictionary from variables.py
        print_header("Mapping ordinal feature/target codes")
        self.mappings()

        # IMPUTE MISSING VALUES (IF APPLICABLE): FIRST STEP
        # X_test, y_test (if any), and 'RDWX' column in X_train, if a feature.
        # 'RDWX' is a special case: all missing will be 'N'/0.
        # Other missing values will be filled in inside cross-validation loop.
        # Also, initialize imputer_ordinal and imputer_categorical.
        if self.impute_strategy is not None:
            self.impute_first_step()

        # COMBINE MULTIPLE TARGETS INTO A SINGLE TUPLE (IF APPLICABLE)
        if len(self.data.targets) > 1:
            print_header("Combining multiple targets into a single tuple")
            self.target_tuple()

        if stop_before_preprocessing:
            return
        # FURTHER PREPROCESSING
        # Create a ColumnTransformer for ordinal and one-hot encoding with custom mapping
        self.preprocess()

        # ASSEMBLE PIPELINE
        self.pipe()

        # PERFORM GRID SEARCH (IF APPLICABLE)
        #####################################
        ########### GRID SEARCH #############
        #####################################

        if self.param_grid is not None and self.grid_search_scoring is not None:
            if self.folds is None:
                # New attribute
                self.cv = 5
            else:
                self.cv = self.folds
            print_header(
                f"Performing grid search with {self.cv}-fold cross-validation"
            )
            # New attribute
            self.grid_search = GridSearchCV(
                estimator=self.pipeline,
                param_grid=self.param_grid,
                cv=self.cv,
                scoring=self.grid_search_scoring,
                n_jobs=-1,
            )
            display(self.grid_search.fit(self.X_train, self.y_train))
            print(
                "Best hyperparameters: self.grid_search.best_params_ = ",
                self.grid_search.best_params_,
            )
            print(
                "\nBest score: self.grid_search.best_score_ =",
                self.grid_search.best_score_,
            )

        elif self.grid_search_scoring is None:
            #####################################
            ######### NO GRID SEARCH ############
            #####################################

            # TRAIN/VAL SPLIT IF NO k-FOLD CROSS-VALIDATION
            self.v_not_cv: bool = False
            if self.folds is None:
                # We will still do validation, just not cross-validation.
                # But we can use the kfold cross-validation code even in this case.
                # If test set is 10% of entire dataset, a 9-fold split will result in train/val/test = 80/10/10
                self.folds = 9
                # New attribute
                self.v_not_cv: bool = True

            # SET UP k-FOLD CROSS-VALIDATION (IF APPLICABLE) AND FIT MODEL
            self.kfold()
            self.fit(self.X_train, self.y_train)

            # SAVE THE PIPELINE
            if self.filename_stem:
                if isinstance(self.classifier, XGBClassifier):
                    suffix = "_xgb"
                elif isinstance(self.classifier, RandomForestClassifier):
                    suffix = "_rf"
                elif isinstance(self.classifier, DecisionTreeClassifier):
                    suffix = "_dt"
                elif isinstance(self.classifier, LogisticRegression):
                    suffix = "_lr"
                elif isinstance(self.classifier, GradientBoostingClassifier):
                    suffix = "_gbm"
                else:
                    suffix = ""
                # New attribute
                self.filepath = self.model_dir.joinpath(filename_stem + suffix + ".joblib")
                print_header("Saving pipeline")
                print(f"{self.filepath}")
                dump(self.pipeline, self.filepath)

            # EVALUATE THE MODEL
            # New attributes
            print(
                "\nMetrics (including confusion matrices) contained in dictionary self.eval"
            )
            self.eval = self.evaluate(self.preds)
            self.triv_eval = self.evaluate(self.triv_preds)
            self.triv_min_eval = self.evaluate(self.triv_preds_min)
            print(
                "Metrics (excluding confusion matrices) also contained in dataframe self.eval_df"
            )
            self.eval_df = evaluation_df(self.eval)
            self.triv_eval_df = evaluation_df(self.triv_eval)
            self.triv_min_eval_df = evaluation_df(self.triv_min_eval)

            # DISPLAY EVALUATION METRICS
            print_header("Other metrics (rows correspond to validation folds if cross-validation has been done)")
            display(self.eval_df)

            # DISPLAY CONFUSION MATRICES
            if self.weights is None:
                if self.data.targets[0] == "TNRY_SEV":
                    self.weights = np.array([1, 2, 3])
                elif self.data.targets[0] == "SEVERITY":
                    self.weights = np.array([1, 2, 3, 4])
                elif self.data.targets == "VICTIMS":
                    self.weights = np.array([1, 2])
                else:
                    pass

            if self.beta is None:
                if self.data.targets[0] == "TNRY_SEV":
                    self.beta = np.array([0.5, 1, 2])
                elif self.data.targets[0] == "SEVERITY":
                    self.beta = np.array([0.5, 0.5, 1, 2])
                elif self.data.targets == "VICTIMS":
                    self.beta = np.array([1, 2])
                else:
                    pass

            if utils.widgets is None:
                display_confusion_matrices(self.eval,
                                           confusion_matrix_with_weighted_fbeta,
                                           beta=self.beta,
                                           weights=self.weights)
            else:
                confusion_matrix_widget(self.eval,
                                        confusion_matrix_with_weighted_fbeta,
                                        beta=self.beta,
                                        weights=self.weights)

                # MERGE AND DISPLAY FEATURE IMPORTANCES
            if self.feature_importances is not None:
                print(
                    "\nFeature importances corresponding to validation set(s) contained in self.feature_importances and self.feature_importances_df.\n"
                    )
                # New attribute
                self.feature_importances_df = merge_feature_importances(self.feature_importances)
                print_header("Feature importance")
                if utils.widgets is None:
                    display_sorted_dataframe(
                        self.feature_importances_df,
                        self.feature_importances_df.columns[0],
                    )
                else:
                    df_display_widget(self.feature_importances_df)

                # Animate (or plot, if just one column in dataframe) feature importances
                self.plot_feature_importance(animate=True)

            if self.threshold_dict_help is not None or self.threshold_dict_hinder is not None:
                self.adjusted_preds = adjusted_predictions(self, self.threshold_dict_help, self.threshold_dict_hinder)
                print(f"\nAdjusted predictions stored in self.adjusted_preds")

    ###############
    ### METHODS ###
    ###############

    def separate(self) -> None:
        # Create copies of self.df_train, self.df_test, but only include feature columns.
        # New attributes...
        self.X_train = self.data.df_train[self.data.features].copy(deep=True)
        self.X_test = self.data.df_test[self.data.features].copy(deep=True)

        # Separate features from targets for training and fitting models.
        self.y_train = self.data.df_train[self.data.targets].copy(deep=True)
        self.y_test = self.data.df_test[self.data.targets].copy(deep=True)
        # New attribute
        # This will tell us whether we're dealing with a binary classification problem or a multiclass one.
        self.max_classes = max(
            [self.y_train[col].nunique() for col in self.y_train.columns]
        )
        for col in self.y_train.columns:
            if self.y_train[col].nunique() < len(self.class_codes[col]):
                actual_classes = self.y_train[col].unique()
                updated_class_codes = {}
                i = 0
                for k, v in self.class_codes[col].items():
                    if k in actual_classes:
                        updated_class_codes[k] = i
                        i += 1
                self.class_codes[col] = updated_class_codes
                self.inverse_class_codes[col] = {
                    value: key for key, value in self.class_codes[col].items()
                }

    def mappings(self) -> None:
        # Define the mapping for ordinal encoding
        # New attributes
        self.categorical_feature_mapping = {
            feature: self.class_codes[feature]
            for feature in self.data.categorical_features
        }
        self.ordinal_feature_mapping = {
            feature: self.class_codes[feature] for feature in self.data.ordinal_features
        }
        # and targets...
        self.categorical_target_mapping = {
            target: self.class_codes[target] for target in self.data.categorical_targets
        }
        self.ordinal_target_mapping = {
            target: self.class_codes[target] for target in self.data.ordinal_targets
        }

        # Apply the mappings
        for feature in self.data.ordinal_features:
            print(f"{feature}: {self.ordinal_feature_mapping[feature]}")
            self.X_train[feature] = self.X_train[feature].map(
                self.ordinal_feature_mapping[feature]
            )
            self.X_test[feature] = self.X_test[feature].map(
                self.ordinal_feature_mapping[feature]
            )

        for target in self.data.ordinal_targets:
            print(f"{target}: {self.ordinal_target_mapping[target]}")
            self.y_train[target] = self.y_train[target].map(
                self.ordinal_target_mapping[target]
            )
            self.y_test[target] = self.y_test[target].map(
                self.ordinal_target_mapping[target]
            )

        print_header("Mapping categorical feature/target codes")
        for feature in self.data.categorical_features:
            print(f"{feature}: {self.categorical_feature_mapping[feature]}")
            self.X_train[feature] = self.X_train[feature].map(
                self.categorical_feature_mapping[feature]
            )
            self.X_test[feature] = self.X_test[feature].map(
                self.categorical_feature_mapping[feature]
            )

        for target in self.data.categorical_targets:
            print(f"{target}: {self.categorical_target_mapping[target]}")
            self.y_train[target] = self.y_train[target].map(
                self.categorical_target_mapping[target]
            )
            self.y_test[target] = self.y_test[target].map(
                self.categorical_target_mapping[target]
            )

    def impute_first_step(self) -> None:
        # First, make copies in case we need to examine the unimputed data later
        # X...
        self.X_train_unimputed = self.X_train.copy(deep=True)
        self.X_test_unimputed = self.X_test.copy(deep=True)
        # y...
        self.y_train_unimputed = self.y_train.copy(deep=True)
        self.y_test_unimputed = self.y_test.copy(deep=True)

        if "RDWX" in self.data.features:
            # 'RDWX' column is a special case: the only value is 'Y'.
            # We assume that in all other cases, the accident did not occur where there were road works.
            print("\nImputing missing 'RDWX' value as 0 (i.e. 'N').")
            print(
                "This is a special case. Other values will be imputed inside cross-validation loop."
            )
            self.X_train["RDWX"] = self.X_train["RDWX"].fillna(0)
            self.X_test["RDWX"] = self.X_test["RDWX"].fillna(0)
        # For everything else...  we can at least define imputation strategies here before entering cross-validation loop.
        # Create separate imputers for categorical and ordinal columns
        # New attributes
        self.imputer_categorical = SimpleImputer(
            strategy=self.impute_strategy["categorical"],
            fill_value=self.impute_strategy["constant"],
        )
        self.imputer_ordinal = SimpleImputer(
            strategy=self.impute_strategy["ordinal"],
            fill_value=self.impute_strategy["constant"],
        )

        # We can impute the test set at this point as well, regardless of whether we impute separately on each fold for the train/val sets.
        # This is the same imputation process as will be used (if at all) for the training data.
        # We must be wary of data leakage/snooping: the test set is meant to be left untouched until final model evaluation.
        # This is the only manipulation of the test set throughout.
        # We believe the scikit-learn methods below are self-contained and will not affect anything else.
        # Of course, this doesn't even happen here if self.impute_strategy is None.
        self.X_test[self.data.ordinal_features] = self.imputer_ordinal.fit_transform(
            self.X_test[self.data.ordinal_features]
        )
        self.X_test = pd.DataFrame(
            self.imputer_categorical.fit_transform(self.X_test),
            columns=self.X_test.columns,
        )
        # Targets..
        # Actually, there are likely no missing values among the target variables.
        # If there are, we should think about just removing the record from the datatable at the outset.
        # In any case, if there are few or no missing target values, the difference between imputing and not imputing will be negligible.
        self.y_test[self.data.ordinal_targets] = self.imputer_ordinal.fit_transform(
            self.y_test[self.data.ordinal_targets]
        )
        self.y_test = pd.DataFrame(
            self.imputer_categorical.fit_transform(self.y_test),
            columns=self.y_test.columns,
        )

    def impute(self, X, y):
        X_imputed = X.copy(deep=True)
        y_imputed = y.copy(deep=True)

        # Apply imputation to the appropriate columns in X_imputed
        # Note that self.imputer_categorical was defined earlier, in self.impute_first_step
        X_imputed.loc[:, self.data.ordinal_features] = (
            self.imputer_ordinal.fit_transform(
                X_imputed.loc[:, self.data.ordinal_features]
            )
        )
        X_imputed = pd.DataFrame(
            self.imputer_categorical.fit_transform(X_imputed), columns=X_imputed.columns
        )

        # Apply imputation to the appropriate columns in y_imputed.
        # Actually, there are likely no missing values among the target variables.
        # If there are, we should think about just removing the record from the datatable at the outset.
        # In any case, if there are few or no missing target values, the difference between imputing and not imputing will be negligible.
        # Note that self.imputer_ordinal was defined earlier, in self.impute_first_step
        y_imputed.loc[:, self.data.ordinal_targets] = (
            self.imputer_ordinal.fit_transform(
                y_imputed.loc[:, self.data.ordinal_targets]
            )
        )
        y_imputed = pd.DataFrame(
            self.imputer_categorical.fit_transform(y_imputed), columns=y_imputed.columns
        )
        return X_imputed, y_imputed

    def target_tuple(self) -> None:
        # Overwrite y_train etc. with version consisting of tuples of values from columns in self.targets
        y_train_tuple = pd.DataFrame()
        y_test_tuple = pd.DataFrame()
        y_train_tuple["TARGET_TUPLE"] = self.y_train.apply(
            lambda row: tuple(row), axis=1
        )
        y_test_tuple["TARGET_TUPLE"] = self.y_test.apply(lambda row: tuple(row), axis=1)
        self.y_train = y_train_tuple
        self.y_test = y_test_tuple
        # Same for y_train_unimputed etc. (just in case we need them later), if applicable
        if self.impute_strategy is not None:
            y_train_unimputed_tuple = pd.DataFrame()
            y_test_unimputed_tuple = pd.DataFrame()
            y_train_unimputed_tuple["TARGET_TUPLE"] = self.y_train_unimputed.apply(
                lambda row: tuple(row), axis=1
            )
            y_test_unimputed_tuple["TARGET_TUPLE"] = self.y_test_unimputed.apply(
                lambda row: tuple(row), axis=1
            )
            self.y_train_unimputed = y_train_unimputed_tuple
            self.y_test_unimputed = y_test_unimputed_tuple
        # Update the max classes attribute
        self.max_classes = max(
            [self.y_train[col].nunique() for col in self.y_train.columns]
        )
        # Now map the tuples to indices (zero-indexed) for classifier
        yy = [self.y_train, self.y_test]
        if self.impute_strategy is not None:
            yy = yy + [self.y_train_unimputed, self.y_test_unimputed]
        Y = [set(y["TARGET_TUPLE"].unique()) for y in yy]
        tuples = set().union(*Y)
        self.tuples_map = {tup: idx for idx, tup in enumerate(list(tuples))}
        # Add this to the self.class_codes dictionary
        tuple_key = "_".join([target for target in self.data.targets])
        self.class_codes[tuple_key] = self.tuples_map
        self.inverse_class_codes[tuple_key] = {
            value: key for key, value in self.class_codes[tuple_key].items()
        }
        print(
            f"\nMapping target tuples for classifier: self.tuples_map = {self.tuples_map}"
        )
        for y in yy:
            y["TARGET_TUPLE"] = y["TARGET_TUPLE"].map(self.tuples_map)

    def preprocess(self) -> None:
        # New attributes self.categories, self.preprocessor, and self.pipeline
        self.categorical_value_lists = list(
            sorted(list(self.categorical_feature_mapping[key].values()))
            for key in self.categorical_feature_mapping.keys()
        )
        self.ordinal_value_lists = list(
            sorted(list(self.ordinal_feature_mapping[key].values()))
            for key in self.ordinal_feature_mapping.keys()
        )
        if self.impute_strategy is not None:
            if self.impute_strategy["ordinal"] == "constant":
                # In this case, imputing has introduced -1 (or some chosen constant) wherever there were missing ordinal values
                self.ordinal_value_lists = [
                    [self.impute_strategy["constant"]] + value_list
                    for value_list in self.ordinal_value_lists
                ]

            # Define the transformer for ordinal features
            ordinal_transformer = OrdinalEncoder(categories=self.ordinal_value_lists)

            # Define the transformer for categorical features
            categorical_transformer = OneHotEncoder(
                categories=self.categorical_value_lists, handle_unknown="ignore"
            )

            # Define the preprocessor as a ColumnTransformer
            self.preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "ordinal_imputer",
                        self.imputer_ordinal,
                        self.data.ordinal_features,
                    ),
                    (
                        "categorical_imputer",
                        self.imputer_categorical,
                        self.data.categorical_features,
                    ),
                    ("ordinal", ordinal_transformer, self.data.ordinal_features),
                    ("onehot", categorical_transformer, self.data.categorical_features),
                ],
                remainder="passthrough",  # Pass through the remaining columns
            )
        else:
            # Define the OrdinalEncoder for ordinal columns with handling of NaN values
            ordinal_transformer = OrdinalEncoder(
                categories=self.ordinal_value_lists,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan,
            )

            categorical_transformer = OneHotEncoder(
                categories=self.categorical_value_lists,
                handle_unknown="ignore",  # Handle unknown categories by ignoring them
            )
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("ordinal", ordinal_transformer, self.data.ordinal_features),
                    ("onehot", categorical_transformer, self.data.categorical_features),
                ],
                remainder="passthrough",  # Pass through the remaining columns
            )

    def pipe(self) -> None:
        # Create a pipeline with the preprocessor and classifier
        self.pipeline = Pipeline(
            [
                ("preprocessor", self.preprocessor),
                ("classifier", self.classifier),
            ]
        )

    def kfold(self) -> None:
        # New attribute
        if self.data.stratify:
            self.kfold = StratifiedKFold(
                n_splits=self.folds, shuffle=True, random_state=self.data.seed
            )
        else:
            self.kfold = KFold(
                n_splits=self.folds, shuffle=True, random_state=self.data.seed
            )

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        # New attribute: dictionary to store ground truth labels, predictions, and probabilities.
        self.preds = {}
        # ... and trivial predictions...
        self.triv_preds = {}
        self.triv_preds_min = {}
        # New attributes: dictionaries to store feature importances
        self.feature_importances = {}
        self.feature_importances_ohe = {}

        if self.v_not_cv:  # Validation but not cross-validation
            generator = self.kfold.split(X, y)
            # Just get the first tuple of indices and make the train/val set from this
            train_idx, val_idx = next(generator)
            print(
                "Splitting self.X_train and self.y_train into self.X_train/self.X_val and self.y_train/self.y_val"
            )
            if self.impute_strategy is None:
                X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            else:
                print_header("Imputing missing values (separately for train/val sets)")
                print(f"{self.impute_strategy}")
                X_train, y_train = self.impute(X.iloc[train_idx], y.iloc[train_idx])
                X_val, y_val = self.impute(X.iloc[val_idx], y.iloc[val_idx])

            # Balance the training set (if applicable), and fit the model 
            if self.balance is not None:
                if hasattr(self.data, 'supplement'):
                    print("\nBalancing with supplementary data".upper(), end=", ")
                    X_balanced, y_balanced = balance(
                        X_train, y_train,
                        self.data.supplement.X_train, self.data.supplement.y_train,
                        max_repeats=self.balance,
                        random_seed=self.data.seed,
                    )
                else:
                    print("\nBalancing".upper(), end=", ")
                    X_balanced, y_balanced = balance(
                        X_train, y_train,
                        max_repeats=self.balance,
                        random_seed=self.data.seed,
                    )

                print("Fitting".upper())
                display(self.pipeline.fit(X_balanced, np.ravel(y_balanced)))
            else:
                print("Fitting".upper())
                display(self.pipeline.fit(X_train, np.ravel(y_train)))

            # Make inferences/predictions
            print("Making predictions on validation set".upper())
            y_val_pred = self.pipeline.predict(X_val)
            y_val_prob = self.pipeline.predict_proba(X_val)

            # Store in dictionary
            self.preds[0] = {
                "target": y_val.values.reshape(
                    -1,
                ),
                "prediction": y_val_pred,
                "probabilities": y_val_prob,
            }
            # trivial predictions...
            self.triv_preds[0] = dict.fromkeys(
                ["target", "prediction", "probabilities"]
            )
            self.triv_preds_min[0] = dict.fromkeys(
                ["target", "prediction", "probabilities"]
            )
            self.triv_preds[0]["target"] = y_val.values.reshape(-1)
            self.triv_preds_min[0]["target"] = y_val.values.reshape(-1)
            tgt = y_train.columns[0]
            cc = self.class_codes[tgt]
            _, self.triv_preds[0]["prediction"], self.triv_preds[0]["probabilities"] = (
                trivial(
                    y_train,
                    class_codes=cc,
                    class_probs="zero_one",
                    pos_label="majority_class",
                    num_preds=y_val.shape[0],
                )
            )
            _, self.triv_preds_min[0]["prediction"], self.triv_preds_min[0]["probabilities"] = (
                trivial(
                    y_train,
                    class_codes=cc,
                    class_probs="zero_one",
                    pos_label="minority_class",
                    num_preds=y_val.shape[0],
                )
            )

            # Get feature importances, too
            self.feature_importances[0], self.feature_importances_ohe[0] = (
                self.feature_importance()
            )

            # Print helpful message
            print(
                "\nTarget, prediction, and probabilities for validation stored in self.preds dictionary."
            )
            print(
                "\nFeature importances based on validation set stored in self.feature_importances dictionary."
            )

        else:  # Cross-validation
            print_header(f"Fitting data in {self.folds}-fold cross-validation process")
            if self.impute_strategy is not None:
                print(
                    "\nSeparate imputation for each fold (inside cross-validation loop):"
                )
                print(f"\n{self.impute_strategy}")

            for i, (train_idx, val_idx) in enumerate(self.kfold.split(X, y)):
                # For the ith partitioning of the training set:
                print(f"\nCV {i}: ".upper(), end=" ")
                if self.impute_strategy is None:
                    print(f"Folding".upper(), end=", ")
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                else:
                    print(f"Imputing".upper(), end=" ")
                    X_train, y_train = self.impute(X.iloc[train_idx], y.iloc[train_idx])
                    X_val, y_val = self.impute(X.iloc[val_idx], y.iloc[val_idx])

                # Balance the training set (if applicable), and fit the model 
                if self.balance is not None:
                    if hasattr(self.data, 'supplement'):
                        print("Balancing with supplementary data".upper(), end=", ")
                        X_balanced, y_balanced = balance(
                            X_train, y_train,
                            self.data.supplement.X_train, self.data.supplement.y_train,
                            max_repeats=self.balance,
                            random_seed=self.data.seed,
                        )
                    else:
                        print("Balancing".upper(), end=", ")
                        X_balanced, y_balanced = balance(
                            X_train, y_train,
                            max_repeats=self.balance,
                            random_seed=self.data.seed,
                        )

                    print("Fitting".upper())
                    if i == 0:
                        display(self.pipeline.fit(X_balanced, np.ravel(y_balanced)))
                    else:
                        self.pipeline.fit(X_balanced, np.ravel(y_balanced))
                else:
                    print("Fitting".upper())
                    if i == 0:
                        display(self.pipeline.fit(X_train, np.ravel(y_train)))
                    else:
                        self.pipeline.fit(X_train, np.ravel(y_train))

                # Make inferences/predictions
                print(f"\n       Making predictions on validation fold".upper())
                y_val_pred = self.pipeline.predict(X_val)
                y_val_prob = self.pipeline.predict_proba(X_val)

                # Store in dictionary
                self.preds[i] = {
                    "target": y_val.values.reshape(
                        -1,
                    ),
                    "prediction": y_val_pred,
                    "probabilities": y_val_prob,
                }
                # Trivial predictions...
                self.triv_preds[i] = dict.fromkeys(
                    ["target", "prediction", "probabilities"]
                )
                self.triv_preds_min[i] = dict.fromkeys(
                    ["target", "prediction", "probabilities"]
                )
                self.triv_preds[i]["target"] = y_val.values.reshape(-1)
                self.triv_preds_min[i]["target"] = y_val.values.reshape(-1)
                tgt = y_train.columns[0]
                cc = self.class_codes[tgt]
                (
                    _,
                    self.triv_preds[i]["prediction"],
                    self.triv_preds[i]["probabilities"],
                ) = trivial(
                    y_train,
                    class_codes=cc,
                    class_probs="zero_one",
                    pos_label="majority_class",
                    num_preds=y_val.shape[0],
                )
                (
                    _,
                    self.triv_preds_min[i]["prediction"],
                    self.triv_preds_min[i]["probabilities"],
                ) = trivial(
                    y_train,
                    class_codes=cc,
                    class_probs="zero_one",
                    pos_label="minority_class",
                    num_preds=y_val.shape[0],
                )

                # Get feature imortances, too
                self.feature_importances[i], self.feature_importances_ohe[i] = (
                    self.feature_importance()
                )

            # Print helpful message after coming out of cross-validation loop
            print(
                "\nTarget, prediction, and probabilities for each validation fold stored in self.preds dictionary."
            )
            print(
                "\nFeature importances corresponding to each validation fold stored in self.feature_importances dictionary."
            )

    def evaluate(self, prediction_dictionary: dict) -> dict:
        metrics = [
            "ACC",
            "BACC",
            "precision",
            "recall",
            "F1",
            "F2",
            "MCC",
            "ROC-AUC",
            "confusion",
        ]
        data = {metric: [] for metric in metrics}

        for fold, dictionary in prediction_dictionary.items():
            target = dictionary["target"]
            prediction = dictionary["prediction"]
            probabilities = dictionary["probabilities"]

            accuracy = accuracy_score(target, prediction)
            data["ACC"].append(accuracy)

            balanced_acc = balanced_accuracy_score(target, prediction)
            data["BACC"].append(balanced_acc)

            precision = precision_score(
                target, prediction, average="macro", zero_division=np.nan
            )
            data["precision"].append(precision)

            recall = recall_score(
                target, prediction, average="macro", zero_division=np.nan
            )
            data["recall"].append(recall)

            f1 = f1_score(target, prediction, average="macro")
            data["F1"].append(f1)

            f2 = fbeta_score(target, prediction, beta=2, average="macro")
            data["F2"].append(f2)

            mcc = matthews_corrcoef(target, prediction)
            data["MCC"].append(mcc)

            if self.max_classes == 2:
                try:
                    roc_auc = roc_auc_score(
                        target, probabilities[:, 1], average="weighted"
                    )
                except:
                    roc_auc = np.nan
            else:
                try:
                    roc_auc = roc_auc_score(
                        target, probabilities, average="weighted", multi_class="ovr"
                    )
                except:
                    roc_auc = np.nan
            data["ROC-AUC"].append(roc_auc)

            confusion = pd.crosstab(target, prediction, margins=True).to_dict()
            data["confusion"].append(confusion)

        return data

    def feature_importance(self) -> dict:
        # Access the classifier from the pipeline
        classifier = self.pipeline.named_steps["classifier"]
        try:
            # Attempt to get feature importances
            # Should work for: GradientBoostingClassifier, XGBClassifier, RandomForestClassifier, DecisionTreeClassifier
            feature_importances = classifier.feature_importances_
        except AttributeError:
            try:
                # If AttributeError occurs (e.g., for LogisticRegression), try another method
                feature_importances = classifier.coef_[0]
            except AttributeError:
                # Handle the case where neither method works
                print("Could not obtain feature importances.")
        # feature_importances = array([0.02891089, 0.00525912,...], dtype=float32)
        # This is an array with shape (92,), but we only have 20 features! (Or something like that.)
        # feature_importances gives a number for every one-hot-encoded feature, not just every feature.
        # E.g. if the first column of self.X_train is 'MONTH', then the first twelve elements of the feature_importances will correspond to the one-hot-encoded variables 'MONTH_1',...,'MONTH_12'.
        # We need to obtain all variables (one-hot-encoded caategorical variables, and ordinal variables), in the right order, and match them with the correct number in the feature_importances array.
        onehot_encoded_feature_names = (
            self.pipeline.named_steps["preprocessor"]
            .named_transformers_["onehot"]
            .get_feature_names_out(input_features=self.data.categorical_features)
        )
        # E.g. one_hot_encoded_feature_names = ['RD_CONFG_0.0', 'RD_CONFG_1.0', 'RD_CONFG_2.0', 'RD_CONFG_3.0','RD_CONFG_nan', 'WEATHER_0.0', 'WEATHER_1.0',...]
        # We want to map 'RD_CONFG_0.0' etc. to 'RD_CONFG', and likewise for the other one-hot-encoded feature names.
        # We strip the '_0.0' from the end of the string and make it a value of a dictionary corresponding to key 'RD_CONFG'.
        feature_mapping = {
            o: "_".join(o.split("_")[:-1]) for o in onehot_encoded_feature_names
        }
        # Now feature_mapping = {'RD_CONFG_0.0': 'RD_CONFG', 'RD_CONFG_1.0': 'RD_CONFG',...}.
        # What we really want is {'RD_CONFG' : ['RD_CONFG_0.0', 'RD_CONFG_1.0',...], ...}.
        # Also, we want to include ordinal features.
        inverse_feature_mapping = {feature: [] for feature in self.X_train.columns}
        for k, v in feature_mapping.items():
            inverse_feature_mapping[v].append(k)
        for k, v in inverse_feature_mapping.items():
            if v == []:
                v.append(k)
        # inverse_feature_mapping = {'MONTH': ['MONTH_1', 'MONTH_2',...], ..., 'RD_CONFG' : ['RD_CONFG_0.0', 'RD_CONFG_1.0',...], ...}
        # New attribute
        feature_importances_dict = {feature: [] for feature in inverse_feature_mapping}
        counter = 0
        for feature, expanded_list in inverse_feature_mapping.items():
            for f in expanded_list:
                feature_importances_dict[feature].append(feature_importances[counter])
                counter += 1
        # Now, the inverse_feature_mapping['MONTH'] = ['MONTH_1', 'MONTH_2',...] corresponds to feature_importances_dict['MONTH'] = [0.028910894, 0.0052591194,...] and so on.
        # We'll combine one-hot-encoded feature importances for a given categorical variable into a single feature importance sore for that variable by simply summing them.
        combined_feature_importances = {
            feature: sum(values) for feature, values in feature_importances_dict.items()
        }
        return combined_feature_importances, feature_importances_dict

    def plot_feature_importance(self, animate: bool = False, fold: int = None) -> FuncAnimation:
        if (
                not isinstance(self.feature_importances_df, pd.DataFrame)
                or self.feature_importances_df.empty
        ):
            return

        # Create the figure and axes for the animation
        fig, ax = plt.subplots(figsize=(10, 8))

        # Set the x-axis limits based on the maximum value in self.feature_importances_df
        max_value = self.feature_importances_df.abs().max().max()

        if not animate:
            if fold is None:
                fold = 0
            # Get the column name corresponding to the fold
            col = self.feature_importances_df.columns[fold]
            # Drop the row with index None (i.e. NoneType)
            bars = self.feature_importances_df[[col]].dropna()

            # Sort in order of absolute value, but don't change the values (maintain the signs)
            bars = bars.reindex(bars[col].abs().sort_values(ascending=True).index)

            # Create a color map for positive and negative values
            colors = ["darkblue" if val >= 0 else "darkred" for val in bars[col].values]

            # _Now_ take absolute values, but create a new dataframe for them, so we still have bars if we need it
            abs_bars = bars.abs()

            ax.set_xlim(
                0, 1.01 * max_value
            )  # Set upper x limit to the maximum (absolute) value
            ax.barh(bars.index, abs_bars[col], color=colors, alpha=1, zorder=4)
            ax.grid(True, alpha=0.5, zorder=0)
            ax.set_xlabel("Importance")
            ax.set_ylabel("Feature")
            ax.set_title(f"Feature Importance ({col})")

            plt.show()
            return fig, ax

        if animate:
            # Function to update the plot for each frame
            def update(frame):
                ax.clear()
                col = self.feature_importances_df.columns[frame]
                bars = self.feature_importances_df[[col]].dropna()
                bars = bars.reindex(bars[col].abs().sort_values(ascending=True).index)
                colors = [
                    "darkblue" if val >= 0 else "darkred" for val in bars[col].values
                ]
                abs_bars = bars.abs()

                ax.set_xlim(
                    0, 1.01 * max_value
                )  # Set upper x limit to the maximum (absolute) value
                ax.barh(bars.index, abs_bars[col], color=colors, alpha=1, zorder=4)
                ax.grid(True, alpha=0.5, zorder=0)
                ax.set_xlabel("Importance")
                ax.set_ylabel("Feature")
                ax.set_title(f"Feature Importance ({col})")

                if isinstance(self.classifier, LogisticRegression):
                    # Create custom legend handles and labels for positive and negative values
                    legend_handles = [
                        plt.Rectangle((0, 0), 1, 1, color="darkblue", alpha=0.8),
                        plt.Rectangle((0, 0), 1, 1, color="darkred", alpha=0.8),
                    ]
                    legend_labels = ["Positive", "Negative"]

                    # Display legend in the bottom right corner with custom handles and labels
                    ax.legend(legend_handles, legend_labels, loc="lower right")

            # Create the animation
            anim = FuncAnimation(
                fig,
                update,
                frames=len(self.feature_importances_df.columns),
                repeat=False,
            )

            # Display the animation in the notebook using HTML (if available)
            display(utils.HTML(anim.to_jshtml()))
            plt.close()
            return anim


def evaluation_df(data: dict) -> pd.DataFrame:
    data_copy = data.copy()
    data_copy.pop("confusion", None)
    eval_df = pd.DataFrame(data_copy)
    if len(eval_df) > 1:
        eval_df.loc["mean"] = eval_df.mean()
    return eval_df


def display_confusion_matrices(which_model: dict,
                               func: Callable[[pd.DataFrame,
                                               Union[np.ndarray, None],
                                               Union[np.ndarray, None]], float] = None,
                               beta: Union[float, None] = None,
                               weights: Union[np.ndarray, None] = None) -> None:
    if which_model["confusion"] is None:
        return

    if func is None:
        for i, d in enumerate(which_model["confusion"]):
            if type(d) == dict:
                d_df = pd.DataFrame(d)
                formatted_df = d_df.map(lambda x: f"{int(x):,}" if pd.notna(x) and x == int(x) else x).fillna('-')
                display(i, formatted_df)

    else:
        for i, d in enumerate(which_model["confusion"]):
            if beta is None:
                beta = np.ones(pd.DataFrame(d).shape)
            if weights is None:
                weights = np.ones(pd.DataFrame(d).shape)

            d_augmented = func(d, beta=beta, weights=weights)
            d_augmented = d_augmented  # .fillna('_')
            formatted_df = d_augmented.map(lambda x: f"{int(x):,}" if pd.notna(x) and x == int(x) else x).fillna('-')
            display(i, formatted_df)


def confusion_matrix_widget(which_model: dict,
                            func: Callable[[pd.DataFrame,
                                            Union[np.ndarray, None],
                                            Union[np.ndarray, None]], float] = None,
                            beta: Union[float, None] = None,
                            weights: Union[np.ndarray, None] = None) -> None:
    if which_model["confusion"] is None:
        return

    if func is None:
        dataframes = {
            i: pd.DataFrame(d)
            for i, d in enumerate(which_model["confusion"])
        }
        to_print = "Confusion matrix"

    else:
        dataframes = {
            i: func(d, beta=beta, weights=weights)
            for i, d in enumerate(which_model["confusion"])
        }
        print_weights = {i: float(w) for i, w in enumerate(weights)}
        print_beta = {i: float(b) for i, b in enumerate(beta)}
        to_print = f"Confusion matrix: weighted average class-wise F_beta scores at bottom right (beta values {print_beta}, weights {print_weights})"

    # Dropdown widget to select DataFrame
    dropdown = utils.widgets.Dropdown(options=list(dataframes.keys()))

    # Output widget to display the selected DataFrame
    output = utils.widgets.Output()

    # Function to update displayed DataFrame based on dropdown value
    def update_dataframe(change):
        output.clear_output()  # Clear the output before displaying the selected DataFrame
        selected_df = dataframes[change.new]
        formatted_df = selected_df.map(lambda x: f"{int(x):,}" if pd.notna(x) and x == int(x) else x).fillna('-')
        with output:
            display(formatted_df)

    # Register the function to update the displayed DataFrame when dropdown value changes
    dropdown.observe(update_dataframe, names="value")

    # Initial display of the first DataFrame
    initial_df = dataframes[dropdown.value]
    formatted_df = initial_df.map(lambda x: f"{int(x):,}" if pd.notna(x) and x == int(x) else x).fillna('-')
    with output:
        display(formatted_df)

    # Display the dropdown widget and output widget
    print_header(to_print)
    if len(which_model["confusion"]) > 1:
        display(dropdown)
    display(output)


def merge_feature_importances(fi: dict = None) -> pd.DataFrame:
    dfs = [
        pd.DataFrame(fi[i], index=[i])
        for i in fi.keys()
    ]
    merged_df = pd.concat(dfs, ignore_index=True).T
    # Sort by absolute value of values in column 0.
    # Though sorted by absolute value, values are not changed, i.e. signs are maintained.
    # For XGBClassifier/RandomForesstClassifier/DecisionTreeClassifier, values will be non-negative anyway.
    merged_df = merged_df.reindex(
        merged_df[0].abs().sort_values(ascending=False).index
    )
    return merged_df


def display_sorted_dataframe(df: pd.DataFrame, sort_column) -> None:
    sorted_df = df.reindex(df[sort_column].abs().sort_values(ascending=False).index)
    display(sorted_df)


def df_display_widget(df: pd.DataFrame) -> None:
    # Create the dropdown widget for sorting
    sort_dropdown = utils.widgets.Dropdown(
        options=df.columns,
        description="Sort by:",
        disabled=False,
    )
    # Create the interactive widget
    interactive_sort = utils.widgets.interact(
        display_sorted_dataframe, df=utils.widgets.fixed(df), sort_column=sort_dropdown
    )


# THRESHOLD-ADJUSTED PREDICTIONS

def adjusted_predictions(instance: Type[model],
                         threshold_dict_help: Union[None, OrderedDict[Union[int, str], float]] = None,
                         threshold_dict_hinder: Union[None, OrderedDict[Union[int, str], float]] = None) -> Dict:
    """
    Adjusts the predictions of a model using specified thresholds and returns the updated predictions.
    This function alters the class prediction probabilities by applying the 'help' and 'hinder' thresholds
    to increase or decrease the classification threshold of certain classes.

    Parameters:
    -----------
    instance : Type[model]
        The model instance containing predictions, targets, and evaluation methods.

    threshold_dict_help : Union[None, OrderedDict[Union[int, str], float]], optional
        A dictionary where the keys represent class indices and the values represent threshold values.
        If the class probability exceeds the threshold, it is boosted to 1.
        (It's actually more nuanced than that: see the threshold function for details.)

    threshold_dict_hinder : Union[None, OrderedDict[Union[int, str], float]], optional
        A dictionary where the keys represent class indices and the values represent threshold values.
        If the class probability falls below the threshold, it is reduced to 0.

    Returns:
    --------
    Dict
        A dictionary containing the adjusted predictions for each fold, along with the target labels
        and probabilities before and after applying the thresholds.

    Example:
    --------
    >>> threshold_help = OrderedDict([(1, 0.3), (2, 0.4)])
    >>> threshold_hinder = OrderedDict([(0, 0.2)])
    >>> adjusted_predictions(model_instance, threshold_dict_help=threshold_help, threshold_dict_hinder=threshold_hinder)

    This adjusts the prediction probabilities based on the provided thresholds and returns the modified predictions.

    Notes:
    ------
    - Both `threshold_dict_help` and `threshold_dict_hinder` cannot be `None` simultaneously.
      At least one of them must be provided to modify the predictions.
    - This function displays evaluation metrics (confusion matrices and other metrics) for the adjusted predictions.
    """

    # Ensure that at least one threshold adjustment is provided
    if threshold_dict_help is None and threshold_dict_hinder is None:
        print("At least one threshold adjustment must be specified in order to alter predictions.")
        return

    adjusted_preds = {}  # Dictionary to store the adjusted predictions for each fold

    # Iterate over the prediction data and apply the thresholds
    for idx, dictionary in instance.preds.items():
        adjusted_preds[idx] = {}
        adjusted_preds[idx]['target'] = instance.preds[idx]['target']  # Original target labels
        adjusted_preds[idx]['probabilities'] = instance.preds[idx]['probabilities']  # Original prediction probabilities
        # Apply the threshold adjustments to generate new predictions
        adjusted_preds[idx]['prediction'] = prediction(adjusted_preds[idx]['probabilities'],
                                                       threshold_dict_help=threshold_dict_help,
                                                       threshold_dict_hinder=threshold_dict_hinder)

    # Evaluate the model with the adjusted predictions
    adjusted_eval = instance.evaluate(prediction_dictionary=adjusted_preds)

    # Convert threshold values to a more readable format (rounded to two decimal places)
    print_help_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in threshold_dict_help.items())
    print_hinder_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in threshold_dict_hinder.items())

    # Print the threshold adjustments used
    print_header(f"Threshold adjustments: \'help\' {print_help_dict}', \'hinder\' {print_hinder_dict}")

    # Display the confusion matrix for the adjusted predictions
    confusion_matrix_widget(adjusted_eval,
                            confusion_matrix_with_weighted_fbeta,
                            beta=instance.beta,
                            weights=instance.weights)

    # Convert evaluation results to a DataFrame for easy display and analysis
    adjusted_eval_df = evaluation_df(adjusted_eval)

    # Display other evaluation metrics (precision, recall, F* score, etc.)
    print_header(f'Other metrics (rows correspond to validation folds if cross-validation has been done)')

    display(adjusted_eval_df)

    return adjusted_preds  # Return the adjusted predictions dictionary


# CUSTOM GRID SEARCHES

def threshold_grid_search(instance: Type[model],
                          priority: List[Tuple[int, str]],  # e.g. [(2, 'help'), (1, 'help'), (0, 'hinder')]
                          threshold: np.ndarray,
                          # e.g. np.array([np.arange(0.05, 0.5, 0.05), np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.95, 0.05)])
                          timeout: Union[int, None]) -> Tuple[float, np.ndarray]:
    """
    Performs a grid search to optimize threshold values for adjusting class predictions based on probability estimates.
    The function evaluates different thresholds by modifying the prediction probabilities and selecting the threshold
    combination that maximizes the F* (F-star) score, averaged over validation folds. The search is terminated if a timeout
    occurs or all threshold combinations are evaluated.

    Parameters:
    -----------
    instance : Type[model]
        The model instance containing prediction probabilities, target labels, and evaluation methods.

    priority : List[Tuple[int, str]]
        A list of tuples representing the class index and whether the threshold is a 'help' (increases probability)
        or 'hinder' (decreases probability) type. For example, [(2, 'help'), (1, 'help'), (0, 'hinder')] defines which
        thresholds should be elevated or reduced for specific classes.

    threshold : np.ndarray
        A numpy array of possible threshold values to test for each class. The grid search will iterate over all possible
        combinations of these threshold values.

    timeout : Union[int, None]
        An optional timeout value (in seconds). If specified, the grid search will stop once this time limit is reached.

    Returns:
    --------
    Tuple[float, np.ndarray]
        A tuple containing the maximum F* score found and the corresponding optimal threshold combination (as dictionaries
        of 'help' and 'hinder' thresholds).

    Example:
    --------
    >>> max_score, optimal_thresholds = threshold_grid_search(
            instance=model_instance,
            priority=[(2, 'help'), (1, 'help'), (0, 'hinder')],
            threshold=np.array([np.arange(0.05, 0.5, 0.05), np.arange(0.05, 0.5, 0.05), np.arange(0.5, 0.95, 0.05)]),
            timeout=300)

    This will search for the optimal thresholds over the specified grid of threshold values, stopping if it takes
    longer than 300 seconds, and return the best F* score and threshold values.
    """

    preds = instance.preds  # The predicted probabilities and targets stored in the model instance
    beta = instance.beta  # Beta value used for F* score calculation (determines precision-recall tradeoff)
    weights = instance.weights  # Class weights for F* score calculation

    max_score: float = 0.  # Initialize the maximum F* score
    argmax: Union[None, Tuple[Union[None, OrderedDict[str, float]], Union[
        None, OrderedDict[str, float]]]] = None  # Initialize the optimal thresholds

    tic: float = time.time()  # Start the timer for timeout control
    time_period: int = 1  # Time checkpoint for periodic status updates

    # Iterate over all combinations of threshold values
    for t in itertools.product(*threshold):
        # Create help and hinder threshold dictionaries based on the priority list
        threshold_dict_help: Union[None, OrderedDict[str, float]] = OrderedDict(
            [(priority[i][0], t[i]) for i in range(len(priority)) if priority[i][1] == 'help'])
        threshold_dict_hinder: Union[None, OrderedDict[str, float]] = OrderedDict(
            [(priority[i][0], t[i]) for i in range(len(priority)) if priority[i][1] == 'hinder'])

        adjusted_preds = {}  # Dictionary to store the adjusted predictions for each fold

        # Iterate over the predictions and apply thresholds
        for idx, dictionary in preds.items():
            adjusted_preds[idx] = {}
            adjusted_preds[idx]['target'] = preds[idx]['target']
            adjusted_preds[idx]['probabilities'] = preds[idx]['probabilities']
            adjusted_preds[idx]['prediction'] = prediction(adjusted_preds[idx]['probabilities'],
                                                           threshold_dict_help=threshold_dict_help,
                                                           threshold_dict_hinder=threshold_dict_hinder)

        # Evaluate the model with the adjusted predictions
        adjusted_eval = instance.evaluate(prediction_dictionary=adjusted_preds)

        mean_wtd_fbeta = 0.  # Variable to accumulate weighted F* scores across folds
        folds = 0  # Fold counter

        # Compute the weighted F* score for each fold
        for fold, cm_dict in enumerate(adjusted_eval['confusion']):
            folds += 1
            cm = pd.DataFrame(cm_dict)  # Convert confusion matrix dictionary to DataFrame

            try:
                # Calculate precision and recall for each class
                diagonal = np.diag(cm)
                recall = (diagonal / cm['All'])[:-1].values  # Ignore the last row (totals)
                precision = (diagonal / cm.loc['All'])[:-1].values  # Ignore the last column (totals)

                # Calculate weighted F* score for this fold
                wtd_fbeta = weighted_average_f(beta=beta,
                                               weights=weights,
                                               precision=precision,
                                               recall=recall)
                mean_wtd_fbeta += wtd_fbeta  # Accumulate the F* score

            except Exception as e:
                pass  # Continue if an error occurs (e.g., division by zero)

        # Average the F* score across folds
        mean_wtd_fbeta /= folds

        # Update the maximum score and optimal thresholds if the current score is better
        if mean_wtd_fbeta > max_score:
            max_score = mean_wtd_fbeta
            argmax = threshold_dict_help, threshold_dict_hinder  # Store the best thresholds

            # Print status updates
            if folds > 1:
                print(f'Best mean F* score so far (F* averaged over validation folds): {max_score}')
            else:
                print(f'Best score so far: {max_score}')

            # Print the current best thresholds
            print_argmax_help_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in argmax[0].items())
            print_argmax_hinder_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in argmax[1].items())
            print(f'Corresponding thresholds: help {print_argmax_help_dict}, hinder {print_argmax_hinder_dict}')
            print(f'Time taken: {time.time() - tic:.2f} seconds and counting\n')

        # Periodic status updates
        if time.time() - tic > time_period * 60:
            print_help_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in threshold_dict_help.items())
            print_hinder_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in threshold_dict_hinder.items())
            print(f'Status update at {time_period} minutes: help {print_help_dict}, hinder {print_hinder_dict}')
            if folds > 1:
                print(f'Mean F* score (F* averaged over validation folds): {mean_wtd_fbeta}\n')
            else:
                print(f'F* score: {mean_wtd_fbeta}\n')
            time_period += 1

        # Stop the search if the timeout is reached
        if timeout and time.time() - tic > timeout:
            print(f'Stopping search after {timeout} seconds.\n'.upper())
            break

    toc = time.time()  # End the timer

    # Print the best score and thresholds found during the search
    if folds > 1:
        print(f'Best mean F* score (F* averaged over validation folds): {max_score}')
    else:
        print(f'Best F* score: {max_score}')
    print(f'Corresponding thresholds: help {print_argmax_help_dict}, hinder {print_argmax_hinder_dict}')
    print(f'Time taken: {toc - tic:.2f} seconds')

    return max_score, argmax


def custom_xgb_grid_search(instance: Type[model],
                           balance_range: Union[List, np.array],
                           max_depth_range: Union[List, np.array],
                           n_estimators_range: Union[List, np.array],
                           results_filename: str,
                           timeout: Union[int, None] = None, ) -> None:
    """
    Performs a grid search optimization specifically for an XGBClassifier within a custom model class instance.
    
    Parameters:
    - instance (Type[model]): An instance of a custom model class, expected to contain an XGBClassifier.
    - balance_range (Union[List, np.array]): A range of values to iterate over for the 'balance' parameter.
    - max_depth_range (Union[List, np.array]): A range of values to iterate over for the 'max_depth' parameter of XGBClassifier.
    - n_estimators_range (Union[List, np.array]): A range of values to iterate over for the 'n_estimators' parameter of XGBClassifier.
    - results_filename (str): The filename under which to save the results of the grid search.
    - timeout (Union[int, None], optional): The maximum amount of time in seconds that the grid search should run (the current grid point evaluation will finish and then the function will terminate). Defaults to None.
    
    Returns:
    - None: This function does not return any value but saves the grid search results to a file.
    
    Raises:
    - Exception: Generic exceptions caught during loading of previous grid search results, with an error message printed.
    """
    # Ensure the classifier in the instance is of XGBClassifier type
    if not isinstance(instance.classifier, XGBClassifier):
        print(
            f"The current model instance uses {type(instance.classifier).__name__}, but we only perform grid search using XGBClassifier.")
        return

    # Attempt to load existing grid search results or initialize an empty dictionary
    try:
        results_dict = load_gs_items(model_dir=instance.model_dir, filename=results_filename)
    except Exception as e:
        print(f"Failed to load existing results: {str(e)}")
        print(f"This is expected if this is the first time a grid search has been performed.")
        results_dict = {}

    # Calculate the total number of configurations to be tested
    total_grid_points = len(n_estimators_range) * len(max_depth_range) * len(balance_range)
    terminate_search = False
    tic = time.time()
    counter = 0

    # Iterate over all combinations of balance, max_depth, and n_estimators
    for balance in balance_range:
        if terminate_search:
            break
        for max_depth in max_depth_range:
            if terminate_search:
                break
            for n_estimators in n_estimators_range:

                # Monitor time and progress
                toc = time.time() - tic
                print(f'Number of grid points checked: {counter} of {total_grid_points}.')
                print(f'Time taken: {toc // 60} minutes, {toc - (toc // 60) * 60:.2f} seconds.')

                # Check if timeout has been exceeded and terminate if true
                if timeout and time.time() - tic > timeout:
                    print(f'Stopping search as time has exceeded {timeout} seconds.\n'.upper())
                    terminate_search = True

                # Break the loop if termination flag is set
                if terminate_search:
                    break

                # Perform grid search if this configuration has not been evaluated yet
                key = str((balance, max_depth, n_estimators))
                if key not in results_dict:
                    print_header(f"(balance, max_depth, n_estimators) = ({balance}, {max_depth}, {n_estimators})")
                    classifier = XGBClassifier(objective='multi:softmax',
                                               eval_metric='mlogloss',
                                               max_delta_step=1,
                                               importance_type='weight',
                                               max_depth=max_depth,
                                               n_estimators=n_estimators,
                                               nthread=-1)

                    current_model_instance = model(data=instance.data,
                                                   folds=instance.folds,
                                                   impute_strategy=instance.impute_strategy,
                                                   classifier=instance.classifier,
                                                   balance=balance,
                                                   filename_stem=None,
                                                   model_dir=instance.model_dir,
                                                   beta=instance.beta,
                                                   weights=instance.weights)

                    results_dict[key] = {
                        'evaluation': copy.deepcopy(current_model_instance.eval),
                        'feature_importance': current_model_instance.feature_importances
                    }

                counter += 1

                # Clear console output to keep the output window clean
                clear_output()

    clear_output()
    if terminate_search:
        print(f'Stopped search as more than {timeout} seconds have elapsed.\n'.upper())
    print(f'Time taken: {toc // 60} minutes, {toc - (toc // 60) * 60:.2f} seconds.')
    print(f'Cumulative data for {counter} of {total_grid_points} grid points will now be stored in a .txt file.')

    # Save the accumulated results to a file
    save_gs_items(model_dir=instance.model_dir,
                  filename=results_filename,
                  gs_dict=results_dict)


def convert(item):
    """
    Recursively convert numpy data types in the given item to native Python types.
    
    Parameters:
        item: A complex data structure potentially containing numpy data types.
        
    Returns:
        A version of the input where all numpy data types have been converted to native Python types.
    """
    if isinstance(item, dict):
        return {key: convert(value) for key, value in item.items()}
    elif isinstance(item, list):
        return [convert(x) for x in item]
    elif isinstance(item, np.generic):
        return item.item()
    return item


def save_gs_items(model_dir: Path, filename: str, gs_dict: dict):
    """
    Saves the entire dictionary to a text file in JSON format after converting numpy data types to native Python types.
    
    This function handles the conversion of complex data types such as numpy floats which are not directly serializable by the json module.
    
    Parameters:
    - model_dir (Path): The directory where the JSON file will be saved. If the directory does not exist, it will be created.
    - filename (str): The name of the file to save the data. If '.txt' is not specified, it will be added.
    - gs_dict (dict): The dictionary containing the data to be saved.
    
    Returns:
    - None: The function writes to a file and does not return any value.
    
    Raises:
    - IOError: If the file cannot be written.
    - ValueError: If there are issues with the format of the dictionary.
    
    Example:
    >>> save_gs_items(Path('/path/to/directory'), 'results.txt', {'example_key': 'data'})
    This will write the dictionary to '/path/to/directory/results.txt'.
    """
    if not filename.endswith('.txt'):
        filename += '.txt'
    file_path = model_dir / filename

    # Ensure the directory exists; if not, create it
    model_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy data types to native Python types
    gs_dict = convert(gs_dict)

    # Write the entire dictionary at once, using indentation for better readability
    with open(file_path, 'w') as file:
        json.dump(gs_dict, file, indent=4)  # Use indent for pretty-printing

    print("File saved successfully to:", file_path)


# And recover the dictionary from a text file...

def convert_keys_to_int(d: Any) -> Any:
    """
    Recursively converts dictionary keys to integers if they are numeric strings, otherwise leaves them as is.

    This function is useful when JSON keys expected to be integers are parsed as strings by the json module, which is common
    when keys in JSON objects represent numeric identifiers.

    Parameters:
    - d (Any): A dictionary or part of a nested data structure to process.

    Returns:
    - Any: A dictionary with the same structure as input, but with keys converted to integers where applicable.
    """
    if isinstance(d, dict):
        return {int(k) if k.isdigit() else k: convert_keys_to_int(v) for k, v in d.items()}
    return d


def load_gs_items(model_dir: Path, filename: str) -> Dict:
    """
    Loads a dictionary from a text file containing JSON-formatted text.

    This function reads a single JSON object from a file and optionally processes it to convert numeric string keys to integer keys,
    which is particularly useful when the JSON data represents structured data with potentially numeric keys.

    Parameters:
    - model_dir (Path): The directory containing the JSON file.
    - filename (str): The name of the file from which to load the data, with '.txt' automatically appended if not present.

    Returns:
    - Dict: The reconstructed dictionary from the JSON data in the file, with keys that were numeric strings converted to integers.

    Raises:
    - FileNotFoundError: If the specified file does not exist.
    - JSONDecodeError: If the JSON data in the file is not correctly formatted.
    - IOError: If there is an error opening or reading the file.
    """
    # Ensure filename ends with '.txt'
    if not filename.endswith('.txt'):
        filename += '.txt'
    file_path = model_dir / filename

    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.read()  # Read the entire file content at once
            gs_dict = json.loads(data)  # Load the whole JSON data at once

        # Convert dictionary keys where needed
        for parameter_string in gs_dict.keys():
            if "evaluation" in gs_dict[parameter_string]:
                for i, d in enumerate(gs_dict[parameter_string]["evaluation"]["confusion"]):
                    gs_dict[parameter_string]["evaluation"]["confusion"][i] = convert_keys_to_int(d)
            if "feature_importance" in gs_dict[parameter_string]:
                gs_dict[parameter_string]["feature_importance"] = {int(k) if k.isdigit() else k: v for k, v in
                                                                   gs_dict[parameter_string][
                                                                       "feature_importance"].items()}
    else:
        raise FileNotFoundError(f"No file found at {file_path}")

    return gs_dict


def grid_search_results(instance: Type[model],
                        results_filename: str,
                        sort_by: Union[None, str]) -> pd.DataFrame:
    """
    Processes and summarizes the results of a grid search over multiple hyperparameter combinations,
    computing average precision, recall, and F-star score across folds. Returns a DataFrame with the
    summarized metrics for each hyperparameter combination.

    Parameters:
    -----------
    instance : Type[model]
        The model instance that contains the hyperparameters, beta, and class weights necessary for
        computing the weighted F-star score.

    results_filename : str
        The filename of the grid search results file, stored in the model's directory. This file contains
        the evaluation metrics and confusion matrices for each hyperparameter combination tested during the grid search.

    sort_by : Union[None, str]
        The column by which the final results should be sorted. If None, the results are not sorted.
        If a valid column name is provided, the results are sorted in descending order based on that column.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the average F-star score, precision, and recall for each class,
        averaged over all cross-validation folds. If `sort_by` is provided, the DataFrame is sorted
        by the specified column.

    Example:
    --------
    >>> grid_search_results(instance=model_instance, results_filename="grid_search_results", sort_by="F*")

    This will load the grid search results, compute evaluation metrics, and return a DataFrame sorted by the F-star score.
    """

    # Load the grid search results dictionary from the specified file
    gs_dict = load_gs_items(model_dir=instance.model_dir, filename=results_filename)

    # Initialize a dictionary to store the results
    results_dict = {}

    # Iterate over each set of grid search parameters (hyperparameter combinations)
    for gs_parameters in gs_dict.keys():
        folds = 0  # Counter for the number of cross-validation folds
        Fstar_average = 0  # To accumulate the F-star score over folds

        # Iterate over each fold's confusion matrix in the evaluation results
        for i, cm in enumerate(gs_dict[gs_parameters]['evaluation']['confusion']):
            cm_df = pd.DataFrame(cm).iloc[:-1, :-1]  # Extract confusion matrix without totals row/column
            cm_arr = cm_df.to_numpy()  # Convert DataFrame to NumPy array for computation

            # Sum rows and columns to calculate precision and recall
            row_sums = cm_arr.sum(axis=1)  # Sum of each row (true positives + false negatives)
            column_sums = cm_arr.sum(axis=0)  # Sum of each column (true positives + false positives)
            precision = np.diag(cm_arr) / column_sums  # Precision: TP / (TP + FP)
            recall = np.diag(cm_arr) / row_sums  # Recall: TP / (TP + FN)

            # Compute weighted F-star score for this fold using beta and weights
            Fstar_average += weighted_average_f(instance.beta, instance.weights, precision, recall)

            # Accumulate precision and recall over folds
            if i == 0:
                precision_average = precision
                recall_average = recall
            else:
                precision_average += np.diag(cm_arr) / column_sums
                recall_average += np.diag(cm_arr) / row_sums

            folds += 1  # Increment the fold counter

        # Average the precision, recall, and F-star score across all folds
        precision_average /= folds
        recall_average /= folds
        Fstar_average /= folds

        # Store the results for this grid search parameter combination in the results dictionary
        results_dict[gs_parameters] = {
            **{'F*': float(Fstar_average)},  # Average F-star score
            **{f'precision {i}': float(precision[i]) for i in range(len(precision))},  # Precision for each class
            **{f'recall {i}': float(recall[i]) for i in range(len(recall))}  # Recall for each class
        }

    # Convert results_dict to a DataFrame
    output = pd.DataFrame(results_dict).T

    # Sort the output DataFrame by the specified column, if provided
    if sort_by:
        try:
            output = output.sort_values(by=sort_by, ascending=False)
        except KeyError:
            print(f'Cannot sort by {sort_by}: no column with this name.')

    return output


def grid_point_summary(instance: Type[model],
                       results_filename: str,
                       gs_params: str) -> None:
    """
    Summarizes the evaluation metrics and confusion matrix for a particular set of grid search parameters,
    displaying the results and confusion matrix for a specific combination of hyperparameters.

    Parameters:
    -----------
    instance : Type[model]
        An instance of the model that contains relevant metadata, such as the model directory (model_dir),
        beta parameter, and class weights for generating the confusion matrix.

    results_filename : str
        The name of the file containing the grid search results. This file is loaded to retrieve
        the evaluation metrics and confusion matrices for the grid search results.

    gs_params : str
        A string representing a specific combination of grid search parameters (e.g., "(balance, max_depth, n_estimators)").
        This string is used to index the results in the grid search dictionary and extract the corresponding evaluation metrics.

    Returns:
    --------
    None
        The function displays the evaluation metrics and confusion matrix for the specified grid search parameters.

    Example:
    --------
    >>> grid_point_summary(instance=model_instance,
                           results_filename="grid_search_results",
                           gs_params="(balance, max_depth, n_estimators)")
    """

    # Load the grid search results dictionary from the specified file
    gs_dict = load_gs_items(model_dir=instance.model_dir, filename=results_filename)

    # Display a header with the selected grid search parameters
    print_header(f"(balance, max_depth, n_estimators) = {gs_params}")

    # Extract evaluation metrics, excluding confusion matrix, from the grid search results
    metrics = {k: v for k, v in gs_dict[gs_params]["evaluation"].items() if k != "confusion"}

    # Display the evaluation metrics in tabular format using a DataFrame
    display(evaluation_df(metrics))

    # Display the confusion matrix, along with weighted F-beta scores, using the specified class weights and beta
    confusion_matrix_widget(gs_dict[gs_params]['evaluation'],
                            confusion_matrix_with_weighted_fbeta,
                            beta=instance.beta,
                            weights=instance.weights)
    return


def create_supplementary_data(instance: Type[model],
                              remove_class: Union[list, None] = None) -> None:
    """
    Generates supplementary data for oversampling by extracting unused records from the source data,
    ensuring there is no overlap between the supplementary data and the current training data. Optionally,
    removes specific classes from the supplementary data.

    Parameters:
    -----------
    instance : Type[model]
        An instance of the model, which contains the main training data and other relevant configurations
        for generating supplementary data (like `data.source`, `data.features`, `data.targets`, etc.).

    remove_class : list or None, optional
        A list of classes to be removed from the supplementary data. If specified, all records belonging
        to the listed classes will be removed from the supplementary data. Default is None.

    Returns:
    --------
    None
        The function updates the `instance` by attaching the newly created supplementary data (if any)
        to `instance.data.supplement`.

    Example:
    --------
    >>> create_supplementary_data(instance=model_instance, remove_class=[2, 3])
    """

    # Printing process details
    print_header("Creating supplementary data for oversampling")
    print("- We extract unused records from source data, following the same initial steps as with current model data.\n"
          "- Note that what we remove here is precisely what we restricted to earlier.\n"
          "- Thus, there is no overlap whatsoever between supplementary data and current model data.")

    # Generate a dummy test set for validation purposes
    test_size = int(instance.y_train.nunique().iloc[0])
    print(
        f"- Dummy 'test' sets are only created to avoid errors. They contain only {test_size} records and can be ignored.")

    # Process the supplementary data from the source, following the same preprocessing steps
    supplement_ = process(
        source=instance.data.source,
        restrict_to=None,  # Not restricting to anything, thus retrieving unused data
        remove_if=instance.data.restrict_to,  # Remove records that were included in the original dataset
        drop_row_if_missing_value_in=instance.data.drop_row_if_missing_value_in,
        targets=instance.data.targets,
        features=instance.data.features,
        test_size=test_size,
        seed=instance.data.seed,
        stratify=None,
        stratify_by=None,
    )

    # Create a model object using the supplementary data
    supplement = model(
        data=supplement_,
        impute_strategy=instance.impute_strategy,
        stop_before_preprocessing=True,
    )

    # If certain classes are specified to be removed from the supplementary data
    if remove_class:
        print_header("Removing unwanted classes")
        target = instance.data.targets[0]
        # Filter out the specified classes from the target and corresponding features
        supplement.y_train = supplement.y_train[~supplement.y_train[target].isin(remove_class)]
        supplement.X_train = supplement.X_train.loc[supplement.y_train.index]
        print(f"Removing all records for which {target} is in {remove_class}.\n")

    # Retrieve the original training data (X, y) and supplementary data (X_supp, y_supp)
    X, y = instance.X_train, instance.y_train
    X_supp, y_supp = supplement.X_train, supplement.y_train

    # Calculate the class distribution in the original training data
    value_counts_dict = {k: v for k, v in y[target].value_counts().items()}

    # Get the size of the largest class in the original training data
    M = max(value_counts_dict.values())
    max_class = [k for k, v in value_counts_dict.items() if v == M]

    # Calculate the class distribution in the supplementary data
    value_counts_dict_supp = {k: v for k, v in y_supp[target].value_counts().items()}

    # Combine the class counts from both original and supplementary data
    value_counts_dict_combined = {}
    for k in set(value_counts_dict).union(value_counts_dict_supp):
        value_counts_dict_combined[k] = value_counts_dict.get(k, 0) + value_counts_dict_supp.get(k, 0)

    # Identify any classes that exceed the maximum class size and need trimming
    trim = {k: v - M for k, v in value_counts_dict_combined.items() if v > M}

    # For each class that exceeds the maximum size, randomly remove the excess records
    for k, v in trim.items():
        print_header("Removing excess records")
        print(
            f"- Including supplementary data, we have a combined total of {v + M} records in Class {k}.\n"
            f"- This is {v} more than the largest class (Class {max_class[0]}) in the original training set.\n"
            f"- We will now randomly remove {v} records from Class {k} in supplementary data."
        )
        # Select candidate indices to remove from the supplementary data
        candidate_indices = y_supp[y_supp[target] == k].index
        np.random.seed(supplement_.seed)
        trim_indices = np.random.choice(candidate_indices, v, replace=False)
        y_supp.drop(trim_indices, inplace=True)
        X_supp.drop(trim_indices, inplace=True)

    # Store the trimmed supplementary data in the instance
    instance.data.supplement = supplement

    print_header(f"Creating new attribute to store supplementary data")
    print(f"Supplementary data contained in self.data.supplement.X_train and self.data.supplement.y_train.")

    return


def inference(instance: Type[model],
              X: pd.DataFrame,
              y: Union[None, pd.DataFrame] = None) -> pd.DataFrame:
    """
    Perform inference using a pre-trained model instance. Predict probabilities, unadjusted predictions,
    and optionally adjusted predictions (if threshold dictionaries are provided).

    Parameters:
    instance (Type[model]): The model instance containing the pipeline and threshold information.
    X (pd.DataFrame): The feature set for which predictions are made.
    y (Union[None, pd.DataFrame], optional): Optional ground truth DataFrame for comparison or to append to results.

    Returns:
    pd.DataFrame: A DataFrame that contains the original features, predicted probabilities,
                  unadjusted predictions, and adjusted predictions (if applicable).
    """

    # Create a Path object for the model filepath based on the instance attributes
    model_path = Path(instance.filepath)

    # Load the trained model from disk if it exists, otherwise use the in-memory model pipeline
    if model_path.exists():
        trained = joblib.load(model_path)
    else:
        trained = instance.pipeline

    # Predict probabilities using the trained model
    probabilities = trained.predict_proba(X)

    # Retrieve optional threshold dictionaries for adjusting predictions
    threshold_dict_help = getattr(instance, 'threshold_dict_help', None)
    threshold_dict_hinder = getattr(instance, 'threshold_dict_hinder', None)

    # Generate initial unadjusted predictions based on probabilities
    unadjusted_predictions = prediction(probabilities=probabilities,
                                        threshold_dict_help=None,
                                        threshold_dict_hinder=None)

    # Calculate adjusted predictions if any threshold dictionaries are provided
    if threshold_dict_help is not None or threshold_dict_hinder is not None:
        adjusted_predictions = prediction(probabilities=probabilities,
                                          threshold_dict_help=threshold_dict_help,
                                          threshold_dict_hinder=threshold_dict_hinder)

    # Make a deep copy of the input DataFrame X to store predictions
    prediction_df = X.copy(deep=True)

    # If ground truth y is provided, add it to the prediction DataFrame
    if y is not None:
        target = y.columns[0]  # Extract the column name from y
        prediction_df[target] = y  # Add the target column to the prediction DataFrame
        y = np.ravel(y)  # Flatten y for easier manipulation

    # Create a DataFrame for predicted probabilities with appropriate column names
    probability_df = pd.DataFrame(probabilities, columns=[f'prob {i}' for i in range(probabilities.shape[1])])
    # Concatenate prediction_df and probability_df along the columns (axis=1), preserving the index of prediction_df
    prediction_df = prediction_df.join(probability_df.set_index(prediction_df.index))

    # Add unadjusted and adjusted predictions to the DataFrame
    prediction_df['pred'] = unadjusted_predictions
    prediction_df['adj pred'] = adjusted_predictions

    return prediction_df


def infer_eval(instance: Type[model],
               X: Union[None, pd.DataFrame] = None,
               y: Union[None, pd.DataFrame] = None,
               verbose: Union[None, List[str]] = None, ) -> Tuple[
    pd.DataFrame, Dict, Dict, Union[None, Dict], Union[None, Dict]]:
    """
    Perform inference on a given dataset or the test set of the model instance.
    This function can load the model from disk or use the in-memory model, generate predictions,
    and optionally evaluate the predictions using available threshold dictionaries (if applicable).

    Parameters:
    -----------
    instance: Type[model]
        The model instance, which contains model attributes such as `filepath`, `X_test`, `y_test`,
        and methods like `predict_proba` and `evaluate`.

    X: Union[None, pd.DataFrame], optional
        The feature set to perform inference on. If None, the test set (instance.X_test) is used.
        Default is None.

    y: Union[None, pd.DataFrame], optional
        The target labels corresponding to `X`. If None, the test set target (instance.y_test) is used.
        Default is None.

    verbose: Union[None, List[str]], optional
        List of verbosity options. Can include 'info', 'df', 'unadjusted_eval', and 'adjusted_eval' to
        control the output of various stages (e.g., model loading, prediction output, evaluation details).
        Default is None.

    Returns:
    --------
    Tuple containing the following elements:
    - pd.DataFrame: `prediction_df`, the DataFrame containing predictions and probabilities.
    - Dict: `unadjusted_eval_df`, a dictionary of evaluation results for unadjusted predictions.
    - Dict: `unadjusted_predictions_dict`, a dictionary of unadjusted predictions and probabilities.
    - Union[None, Dict]: `adjusted_eval_df`, evaluation dictionary for adjusted predictions (if applicable).
    - Union[None, Dict]: `adjusted_predictions_dict`, dictionary of adjusted predictions and probabilities
      (if applicable).

    Example:
    --------
    >>> predictions, unadjusted_eval, unadjusted_preds, adjusted_eval, adjusted_preds = inference(model_instance)
    """

    # Create a Path object for the model filepath from instance attributes
    model_path = Path(instance.filepath)

    if verbose is None:
        verbose = []

    # Check if the model exists on disk and load it; otherwise, use the in-memory model instance
    if model_path.exists():
        if 'info' in verbose:
            print("Loading model from file.")
        trained = joblib.load(model_path)
    else:
        if 'info' in verbose:
            print("Using in-memory model.")
        trained = instance.pipeline

    # If no dataset is provided (X and y are None), use the model's test set
    if (X is None and y is None):
        X_data = instance.X_test
        y_data = instance.y_test
        if 'info' in verbose:
            print_header("Evaluating model on test set")
    else:
        # If dataset is provided, use that for inference
        X_data = X
        y_data = y
        if 'info' in verbose:
            print_header("Making predictions on given data")

    # Retrieve optional threshold dictionaries (for adjusting predictions)
    threshold_dict_help = getattr(instance, 'threshold_dict_help', None)
    threshold_dict_hinder = getattr(instance, 'threshold_dict_hinder', None)

    # Prepare a copy of the input features for prediction results
    prediction_df = X_data.copy(deep=True)

    # Unadjusted predictions (without applying thresholds)
    unadjusted_predictions_dict = {0: {}}
    unadjusted_predictions_dict[0]["probabilities"] = trained.predict_proba(X_data)

    # Generate the initial prediction based on the probabilities
    unadjusted_predictions_dict[0]["prediction"] = prediction(
        probabilities=unadjusted_predictions_dict[0]['probabilities'],
        threshold_dict_help=None,
        threshold_dict_hinder=None
    )

    # If target labels (y_data) are available, store them and add to the prediction DataFrame
    if y_data is not None:
        unadjusted_predictions_dict[0]["target"] = np.ravel(y_data)
        target = y_data.columns[0]
        prediction_df[target] = y_data

    # Add probability columns to the prediction DataFrame (one column per class)
    for i in range(unadjusted_predictions_dict[0]["probabilities"].shape[1]):
        prediction_df['prob ' + str(i)] = unadjusted_predictions_dict[0]["probabilities"][:, i]

    # Add unadjusted predictions to the DataFrame
    prediction_df['pred'] = unadjusted_predictions_dict[0]["prediction"]

    # If threshold dictionaries are provided, calculate adjusted predictions
    if threshold_dict_help is not None or threshold_dict_hinder is not None:
        adjusted_predictions_dict = {0: {}}
        adjusted_predictions_dict[0]["probabilities"] = trained.predict_proba(X_data)
        adjusted_predictions_dict[0]["prediction"] = prediction(
            probabilities=adjusted_predictions_dict[0]["probabilities"],
            threshold_dict_help=threshold_dict_help,
            threshold_dict_hinder=threshold_dict_hinder
        )

        # If target labels are available, add them to the adjusted prediction dictionary
        if y_data is not None:
            adjusted_predictions_dict[0]["target"] = np.ravel(y_data)

        # Add adjusted predictions to the prediction DataFrame
        prediction_df['adj pred'] = adjusted_predictions_dict[0]["prediction"]

    # Optionally display the prediction DataFrame
    if 'df' in verbose:
        display(prediction_df)

    # If target labels are available, evaluate the unadjusted predictions
    if y_data is not None:
        unadjusted_eval = instance.evaluate(prediction_dictionary=unadjusted_predictions_dict)
        unadjusted_eval_df = evaluation_df(unadjusted_eval)

        # Optionally print the evaluation results for unadjusted predictions
        if 'unadjusted_eval' in verbose:
            print_header("Predictions based on max probability (not based on adjusted thresholds)")
            confusion_matrix_widget(unadjusted_eval,
                                    confusion_matrix_with_weighted_fbeta,
                                    beta=instance.beta,
                                    weights=instance.weights)

            print_header("Other metrics")
            display(unadjusted_eval_df)

        # If threshold dictionaries exist, evaluate adjusted predictions
        if threshold_dict_help is not None or threshold_dict_hinder is not None:
            adjusted_eval = instance.evaluate(prediction_dictionary=adjusted_predictions_dict)
            adjusted_eval_df = evaluation_df(adjusted_eval)

            # Optionally print evaluation results for adjusted predictions
            if 'adjusted_eval' in verbose:
                print_help_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in threshold_dict_help.items())
                print_hinder_dict = OrderedDict((k, int(v * 100) / 100.0) for k, v in threshold_dict_hinder.items())
                print_header(
                    f"Threshold-adjusted predictions: 'help' {print_help_dict}, 'hinder' {print_hinder_dict}")

                confusion_matrix_widget(adjusted_eval,
                                        confusion_matrix_with_weighted_fbeta,
                                        beta=instance.beta,
                                        weights=instance.weights)

                print_header("Other metrics")
                display(adjusted_eval_df)

            # Return the prediction DataFrame, unadjusted evaluation, and both sets of predictions
            return prediction_df, unadjusted_eval_df, unadjusted_predictions_dict, adjusted_eval_df, adjusted_predictions_dict

        # Return unadjusted evaluation results if no adjusted predictions are generated
        return prediction_df, unadjusted_eval_df, unadjusted_predictions_dict


def explain_balance(instance: Type[model],
                    max_repeats: int = 1,
                    augmented: Union[None, bool] = False, ) -> None:
    """
    Explains the balance technique applied to the training data by either simple or augmented oversampling minority classes.
    This function performs class balancing over multiple folds of cross-validation and prints both class distribution
    before and after balancing. It optionally handles an augmented oversampling approach that incorporates additional
    data to achieve balance.

    Parameters:
    -----------
    instance: Type[model]
        The model instance that contains training and validation data (X_train, y_train),
        cross-validation split functionality (kfold), and optional data augmentation information (instance.data.supplement).

    max_repeats: int, optional
        The maximum multiplier for oversampling the minority classes. Default is 1, meaning no oversampling will occur.
        If greater than 1, records from minority classes will be repeated until they match either the majority class
        or are repeated up to `max_repeats` times their original size.

    augmented: Union[None, bool], optional
        If True, this flag enables augmented oversampling, which involves adding additional data (if available)
        to the training dataset before applying oversampling. Default is False.

    Returns:
    --------
    None
        This function only prints information regarding the balancing process across cross-validation folds and
        does not return any value.

    Example:
    --------
    >>> explain_balance(model_instance, max_repeats=5, augmented=True)
    """

    # Iterate over each fold from the cross-validation split
    for fold, (train_idx, val_idx) in enumerate(instance.kfold.split(instance.X_train, instance.y_train), start=1):

        # If augmented oversampling is enabled, print description and fetch additional training data
        if augmented:
            print_header("Example of augmented oversampling")
            print("- This process adjusts the training dataset to balance class distribution."
                  "\n- It increases the representation of minority classes by first augmenting them with additional similar records."
                  f"\n- Records are repeated until their number matches either the majority class, or has increased {max_repeats}-fold, whichever results in fewer repetitions.")
            # Supplementary data for augmentation
            X_supp: Union[None, pd.DataFrame] = instance.data.supplement.X_train
            y_supp: Union[None, pd.DataFrame] = instance.data.supplement.y_train
        else:
            # If simple oversampling is enabled, print description and don't use any supplementary data
            print_header("Example of simple oversampling")
            print("- This process adjusts the training dataset to balance class distribution."
                  "\n- It increases the representation of minority classes."
                  f"\n- Records are repeated until their number matches either the majority class, or has increased {max_repeats}-fold, whichever results in fewer repetitions.")
            X_supp: Union[None, pd.DataFrame] = None
            y_supp: Union[None, pd.DataFrame] = None

        # Print the fold number
        print_header(f"Fold {fold - 1}")

        # Extract the training and validation sets for the current fold
        X_train: pd.DataFrame = instance.X_train.iloc[train_idx]
        y_train: pd.DataFrame = instance.y_train.iloc[train_idx]
        y_val: pd.DataFrame = instance.y_train.iloc[val_idx]

        # Measure the time taken to perform class balancing
        tic = time.time()

        # Call the balance function to perform the actual balancing or augmentation
        X_balance, y_balance = balance(X=X_train,
                                       y=y_train,
                                       X_supp=X_supp,
                                       y_supp=y_supp,
                                       max_repeats=max_repeats,
                                       random_seed=instance.data.seed, )

        toc = time.time()

        # Print the time taken to balance the training data
        print(f"Time taken to generate 'balanced' training data: {toc - tic:.2f} seconds.\n")

        # Print original training set class counts
        print("Original training set class counts:".upper())
        display(y_train.value_counts())

        # Print upsampled/balanced training set class sizes
        print("\nUpsampled training set class sizes:".upper())
        display(y_balance.value_counts())

        # Gather class counts from the balanced data
        target = y_balance.columns[0]
        class_index_counts = {k: {} for k in np.sort(y_balance[target].unique())}
        class_index_counts = {int(k): v.copy() for k, v in class_index_counts.items()}

        # For each class, check how often records have been repeated and log this information
        for k, v in class_index_counts.items():
            class_k = y_balance[y_balance[target] == k]
            for i in range(max_repeats + 1):
                number_indices_occuring_i_times = (class_k.index.value_counts() == i).sum()
                if number_indices_occuring_i_times > 0:
                    v[i] = int(number_indices_occuring_i_times)

        # Print the number of records repeated for each class
        print("")
        for k, v in class_index_counts.items():
            print(f"Class {k}:", end=" ")
            for i, m in v.items():
                print(f"{m} records repeated {i} times", end=", ")
            print("")

        # Inform the user that the validation set has not been altered
        print("\nNote that the validation set has NOT been altered.")
        print("\nValidation set class counts:".upper())
        display(y_val.value_counts())

        # Wait for user input to either continue or cancel further iteration
        response = input("\nPress enter to continue or enter c to cancel:")
        if response == 'c':
            break
        else:
            # Clear the output for the next iteration
            clear_output()
