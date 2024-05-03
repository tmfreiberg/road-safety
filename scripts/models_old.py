import pandas as pd
import numpy as np
from typing import Type, Union, Optional, Tuple, Callable
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
from evaluation import custom_confusion
from processing import process
from variables import class_codes, inverse_class_codes
from pathlib import Path

from utils import display, clear_output

try:
    # Check if in a Jupyter Notebook
    from IPython import get_ipython

    if get_ipython():
        # Use display function and import widgets
        from IPython.display import HTML
        import ipywidgets as widgets
    else:
        widgets = None
except ImportError:
    widgets = None
    
import copy
import json    

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
def balance(
    X: pd.DataFrame, y: pd.DataFrame, max_repeats: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    col = y.columns[0]

    value_counts_dict = {k: v for k, v in y[col].value_counts().items()}

    M = max(value_counts_dict.values())

    upsample_dict = {k: min(M // v, max_repeats) for k, v in value_counts_dict.items()}

    indices_with_multiplicity = np.concatenate(
        [np.repeat(idx, upsample_dict.get(y.loc[idx, col], 1)) for idx in X.index]
    )
    np.random.shuffle(indices_with_multiplicity)

    X_out = X.loc[indices_with_multiplicity]
    y_out = y.loc[indices_with_multiplicity]
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
        param_grid: dict = None,
        filename_stem: str = None,
        model_dir: Path = None,
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

        # New attributes
        self.class_codes = class_codes.copy()
        self.inverse_class_codes = inverse_class_codes.copy()

        # SEPARATE FEATURES X FROM TARGETS y (X_train, y_train, X_test, y_test)
        print("\nSeparating features from targets\n".upper())
        print("self.X_train/self.X_test, self.y_train/self.y_test")
        self.separate()

        # MAP VALUES (ORDINAL AND CATEGORICAL)
        # E.g. 0,1,2,9 -> 0,1,2,3; 1,2,9 -> 1,2,3, <50, 50, 60, 70, 80, 90, 100 -> 0, 5, 6, 7, 8, 9, 10, etc.
        # Mappings given in 'class_codes' dictionary from variables.py
        print("\nMapping ordinal feature/target codes\n".upper())
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
            print("\nCombining multiple targets into a single tuple".upper())
            self.target_tuple()

        # FURTHER PREPROCESSING
        # Create a ColumnTransformer for ordinal and one-hot encoding with custom mapping
        self.preprocess()

        # ASSEMBLE PIPELINE
        self.pipe()

        # PERFORM GRID SEARCH (IF APPLICABLE)
        if self.param_grid is not None and self.grid_search_scoring is not None:
            if self.folds is None:
                # New attribute
                self.cv = 5
            else:
                self.cv = self.folds
            print(
                f"\nPerforming grid search with {self.cv}-fold cross-validation".upper()
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
                "\nBest hyperparameters: self.grid_search.best_params_ = ",
                self.grid_search.best_params_,
            )
            print(
                "\nBest score: self.grid_search.best_score_ =",
                self.grid_search.best_score_,
            )

        elif self.grid_search_scoring is None:
            # TRAIN/VAL SPLIT IF NO k-FOLD CROSS-VALIDATION
            self.v_not_cv: bool = False
            if self.folds is None:
                # We will still do validation, just not cross-validation.
                # But we can use the kfold cross-validation code even in this case.
                # If test set is 10% of entire dataset, a 9-fold split will result in train/val/test = 80/10/10
                self.folds = 9
                # New attribute
                self.v_not_cv: bool = True

            # SET UP k-FOLD CROSS-VALIDATION (IF APPLICABLE) AND FIT MODEEL
            self.kfold()
            self.fit(self.X_train, self.y_train)

            # SAVE THE PIPELINE
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
            print("\nSaving pipeline:".upper(), f"{self.filepath}")
            dump(self.pipeline, self.filepath)

            # EVALUATE THE MODEL
            # New attributes
            print(
                "\nMetrics (including confusion matrices) contained in dictionary self.eval."
            )
            self.eval = self.evaluate(self.preds) 
            self.triv_eval = self.evaluate(self.triv_preds) 
            print(
                "Metrics (excluding confusion matrices) also contained in dataframe self.eval_df"
            )
            self.eval_df = evaluation_df(self.eval) 
            self.triv_eval_df = evaluation_df(self.triv_eval) 

            # DISPLAY EVALUATION METRICS
            print(
                "\n"
                + "=" * 32
                + "\nEvaluation metrics for each fold".upper()
                + "\n"
                + "=" * 32
            )
            display(self.eval_df)

            # DISPLAY CONFUSION MATRICES
            if widgets is None:
                display_confusion_matrices(self.eval) 
            else:
                confusion_matrix_widget(self.eval) 

            # MERGE AND DISPLAY FEATURE IMPORTANCES
            if self.feature_importances is not None:
                print("\nFeature importances corresponding to validation set(s) contained in self.feature_importances and self.feature_importances_df.\n"
    )
                # New attribute
                self.feature_importances_df = merge_feature_importances(self.feature_importances)
                print("=" * 19 + "\nFeature importance\n".upper() + "=" * 19)
                if widgets is None:
                    display_sorted_dataframe(
                        self.feature_importances_df,
                        self.feature_importances_df.columns[0],
                    )
                else:
                    df_display_widget(self.feature_importances_df)

                # Animate (or plot, if just one column in dataframe) feature importances
                self.plot_feature_importance(animate=True)

    ### METHODS ###

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
            self.X_train[feature].replace(
                self.ordinal_feature_mapping[feature], inplace=True
            )
            self.X_test[feature].replace(
                self.ordinal_feature_mapping[feature], inplace=True
            )

        for target in self.data.ordinal_targets:
            print(f"{target}: {self.ordinal_target_mapping[target]}")
            self.y_train[target].replace(
                self.ordinal_target_mapping[target], inplace=True
            )
            self.y_test[target].replace(
                self.ordinal_target_mapping[target], inplace=True
            )

        print("\nMapping categorical feature/target codes\n".upper())
        for feature in self.data.categorical_features:
            print(f"{feature}: {self.categorical_feature_mapping[feature]}")
            self.X_train[feature].replace(
                self.categorical_feature_mapping[feature], inplace=True
            )
            self.X_test[feature].replace(
                self.categorical_feature_mapping[feature], inplace=True
            )

        for target in self.data.categorical_targets:
            print(f"{target}: {self.categorical_target_mapping[target]}")
            self.y_train[target].replace(
                self.categorical_target_mapping[target], inplace=True
            )
            self.y_test[target].replace(
                self.categorical_target_mapping[target], inplace=True
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
            self.X_train.loc[:, "RDWX"].fillna(0, inplace=True)
            self.X_test.loc[:, "RDWX"].fillna(0, inplace=True)
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
        self.triv_preds = {}
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
                print(
                    f"\nImputing missing values".upper()
                    + "(separately for train/val sets):"
                )
                print(f"\n{self.impute_strategy}")
                X_train, y_train = self.impute(X.iloc[train_idx], y.iloc[train_idx])
                X_val, y_val = self.impute(X.iloc[val_idx], y.iloc[val_idx])

                # Balance the training set (if applicable), and fit the model !!!
                if self.balance is not None:
                    print("\nBalancing".upper(), end=" ")
                    X_balanced, y_balanced = balance(
                        X_train, y_train, max_repeats=self.balance
                    )

                    print("Fitting".upper())
                    display(self.pipeline.fit(X_balanced, np.ravel(y_balanced)))
                else:
                    print("Fitting".upper())
                    display(self.pipeline.fit(X_train, np.ravel(y_train)))

            # Make inferences/predictions
            print(f"\nMaking predictions on validation set".upper())
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
            self.triv_preds[0] = dict.fromkeys(
                ["target", "prediction", "probabilities"]
            )
            self.triv_preds[0]["target"] = y_val.values.reshape(-1)
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

            # Get feature imortances, too
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
            print(
                f"\nFitting data in {self.folds}-fold cross-validation process".upper()
            )
            if self.impute_strategy is not None:
                print(
                    "\nSeparate imputation for each fold (inside cross-validation loop):"
                )
                print(f"\n{self.impute_strategy}")

            for i, (train_idx, val_idx) in enumerate(self.kfold.split(X, y)):
                # For the ith partitioning of the training set:
                print(f"\nCV {i}: ".upper(), end=" ")
                if self.impute_strategy is None:
                    print(f"Folding".upper(), end=" ")
                    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
                    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
                else:
                    print(f"Imputing".upper(), end=" ")
                    X_train, y_train = self.impute(X.iloc[train_idx], y.iloc[train_idx])
                    X_val, y_val = self.impute(X.iloc[val_idx], y.iloc[val_idx])

                # Balance the training set (if applicable), and fit the model !!!
                if self.balance is not None:
                    print("Balancing".upper(), end=" ")
                    X_balanced, y_balanced = balance(
                        X_train, y_train, max_repeats=self.balance
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
                self.triv_preds[i] = dict.fromkeys(
                    ["target", "prediction", "probabilities"]
                )
                self.triv_preds[i]["target"] = y_val.values.reshape(-1)
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

    def infer(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        y_pred = self.pipeline.predict(X)
        y_prob = self.pipeline.predict_proba(X)
        return y_pred, y_prob

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
                target, prediction, average="micro", zero_division=np.nan
            )
            data["precision"].append(precision)

            recall = recall_score(
                target, prediction, average="micro", zero_division=np.nan
            )
            data["recall"].append(recall)

            f1 = f1_score(target, prediction, average="micro")
            data["F1"].append(f1)

            f2 = fbeta_score(target, prediction, beta=2, average="micro")
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

            confusion = custom_confusion(target, prediction)
            data["confusion"].append(
                {
                    "label_given_prediction": confusion[
                        "label_given_prediction_merged"
                    ],
                    "prediction_given_label": confusion[
                        "prediction_given_label_merged"
                    ],
                }
            )
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

    def plot_feature_importance(self, animate: bool = False, fold: int = None) -> None:
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

            # Display the animation in the notebook using HTML
            display(HTML(anim.to_jshtml()))

def evaluation_df(data: dict) -> pd.DataFrame:
    data_copy = data.copy()
    data_copy.pop("confusion", None)
    eval_df = pd.DataFrame(data_copy)
    if len(eval_df) > 1:
        eval_df.loc["mean"] = eval_df.mean()
    return eval_df

def display_confusion_matrices(which_model: dict) -> None:
    if which_model["confusion"] is None:
        return
    dataframes_lgp = {
        i: d["label_given_prediction"]
        for i, d in enumerate(which_model["confusion"])
    }

    dataframes_pgl = {
        i: d["prediction_given_label"]
        for i, d in enumerate(which_model["confusion"])
    }

    print(
        "\n" + "=" * 41 + "\nProb(label | prediction = column heading)\n" + "=" * 41
    )
    for i in dataframes_lgp.keys():
        display(i, dataframes_lgp[i])

    print("\n" + "=" * 35 + "\nProb(prediction | label = row name)\n" + "=" * 35)
    for i in dataframes_pgl.keys():
        display(i, dataframes_pgl[i])

def confusion_matrix_widget(which_model: dict) -> None:
    if which_model["confusion"] is None:
        return
    dataframes_lgp = {
        i: d["label_given_prediction"]
        for i, d in enumerate(which_model["confusion"])
    }

    # Dropdown widget to select DataFrame
    dropdown_lgp = widgets.Dropdown(options=list(dataframes_lgp.keys()))

    # Output widget to display the selected DataFrame
    output_lgp = widgets.Output()

    # Function to update displayed DataFrame based on dropdown value
    def update_dataframe_lgp(change):
        output_lgp.clear_output()  # Clear the output before displaying the selected DataFrame
        selected_df = dataframes_lgp[change.new]
        with output_lgp:
            display(selected_df)

    # Register the function to update the displayed DataFrame when dropdown value changes
    dropdown_lgp.observe(update_dataframe_lgp, names="value")

    # Initial display of the first DataFrame
    initial_df_lgp = dataframes_lgp[dropdown_lgp.value]
    with output_lgp:
        display(initial_df_lgp)

    # Display the dropdown widget and output widget
    print(
        "\n" + "=" * 41 + "\nProb(label | prediction = column heading)\n" + "=" * 41
    )
    display(dropdown_lgp)
    display(output_lgp)

    dataframes_pgl = {
        i: d["prediction_given_label"]
        for i, d in enumerate(which_model["confusion"])
    }

    # Dropdown widget to select DataFrame
    dropdown_pgl = widgets.Dropdown(options=list(dataframes_pgl.keys()))

    # Output widget to display the selected DataFrame
    output_pgl = widgets.Output()

    # Function to update displayed DataFrame based on dropdown value
    def update_dataframe_pgl(change):
        output_pgl.clear_output()  # Clear the output before displaying the selected DataFrame
        selected_df = dataframes_pgl[change.new]
        with output_pgl:
            display(selected_df)

    # Register the function to update the displayed DataFrame when dropdown value changes
    dropdown_pgl.observe(update_dataframe_pgl, names="value")

    # Initial display of the first DataFrame
    initial_df_pgl = dataframes_pgl[dropdown_pgl.value]
    with output_pgl:
        display(initial_df_pgl)

    # Display the dropdown widget and output widget
    print("\n" + "=" * 35 + "\nProb(prediction | label = row name)\n" + "=" * 35)
    display(dropdown_pgl)
    display(output_pgl)
    
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
    sort_dropdown = widgets.Dropdown(
        options=df.columns,
        description="Sort by:",
        disabled=False,
    )
    # Create the interactive widget
    interactive_sort = widgets.interact(
        display_sorted_dataframe, df=widgets.fixed(df), sort_column=sort_dropdown
    )
    

# CUSTOM GRID SEARCH
    
def custom_xgb_grid_search(data: Type[process] = None,    
                           folds: Union[int, None] = 5, 
                           impute_strategy: Union[dict, None] = None, 
                           balance: Union[int, None] = None,
                           filename_stem: str = "xgb_gs",
                           model_dir: Path = None,    
                           max_depth: int = 6, 
                           n_estimators: int = 100,) -> dict:
    
    if data is None:
        print("Cannot pass None to data parameter.")
        return
    if Path is None:
        print("Cannot pass None to model_dir.")
        return
    
    classifier = XGBClassifier(objective='multi:softmax', # 'binary:logistic', if two classes
                              eval_metric='mlogloss', # 'logloss', if two classes
                              max_delta_step=1,                                                   
                              importance_type='weight', 
                              max_depth = max_depth, 
                              n_estimators = n_estimators, 
                              nthread=-1,)   

    instance = model(data=data,
                      folds = folds,
                      impute_strategy=impute_strategy,
                      classifier=classifier,
                      balance=balance,
                      filename_stem=filename_stem,
                      model_dir=model_dir,)    
       
    output = {}
    output['evaluation'] = copy.deepcopy(instance.eval)
    output['feature_importance'] = instance.feature_importances
    for cm in output['evaluation']["confusion"]:
        cm['label_given_prediction'] = cm['label_given_prediction'].to_dict(orient='list') 
        # orient='index' is the wrong orientation; 
        # orient='series' will cause errors to be thrown when we do json.dump... something about series not being serializable.
        # just .to_dict() also will not work...
        cm['prediction_given_label'] = cm['prediction_given_label'].to_dict(orient='list')

    return output   

# Save the grid search results to a text file

def save_gs_items(model_dir: Path, filename: str, gs_dict: dict) -> None:
    if not model_dir.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
    
    if '.' in filename:
        filename = filename[:filename.rindex('.')]    
    filename = filename + ".txt"
    file_path = model_dir.joinpath(filename)
    
    with open(file_path, "a", encoding="utf-8") as file:
        for key, value in gs_dict.items():
            json.dump({key: value}, file, ensure_ascii=False)
            file.write("\n")  # Add a newline for readability if appending multiple items

# And recover the dictionary from a text file...

def load_gs_items(model_dir: Path, filename: str) -> dict:   
    if '.' in filename:
        filename = filename[:filename.rindex('.')]    
    filename = filename + ".txt"    
    file_path = model_dir.joinpath(filename)
    
    gs_dict = {}
    
    if file_path.is_file():
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                key = list(data.keys())[0]
                value = data[key]
                gs_dict[key] = value
    
    return gs_dict            
