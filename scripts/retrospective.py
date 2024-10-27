import pandas as pd
import numpy as np
from typing import Type, Union, List
from models import model, inference
from sklearn.neighbors import NearestNeighbors
from utils import display, clear_output, print_header, combined_value_counts, filter_columns_by_substrings, \
    format_floats_as_integers


def create_representative_rows(df: pd.DataFrame,
                               subset: list, ) -> pd.DataFrame:
    # Group by the subset of columns and keep the first row from each group
    return df.groupby(subset).first().reset_index()


def subtract_groups(A: pd.DataFrame, B: pd.DataFrame, subset: list) -> pd.DataFrame:
    # Step 1: Create A* and B* by keeping the first row from each group in A and B
    A_star = create_representative_rows(A, subset)
    B_star = create_representative_rows(B, subset)
    # Step 2: Merge A* and B* on the subset columns and find rows in B* that don't match A*
    C = B_star.merge(A_star, on=subset, how='left', indicator=True)

    # Step 3: Keep only the rows that are in B* but not in A*
    C = C[C['_merge'] == 'left_only'].drop(columns=['_merge'])

    return C


def compare_feature_combinations(X1: pd.DataFrame,
                                 X2: pd.DataFrame,
                                 features: Union[None, List[str]] = None,
                                 ignore: Union[None, List[str]] = None,
                                 fillna: bool = True,
                                 sentinel: int = -1) -> pd.DataFrame:
    # Ensure features and ignore lists are properly initialized
    if features is None:
        features = list(X1.columns)
    if ignore is None:
        ignore = []

    # Step 1: Preprocess to create valid 'subset' parameter
    subset = set(features).intersection(X1.columns, X2.columns) - set(ignore)
    subset = list(subset)

    # Create copies of the dataframes with only the relevant subset columns
    if fillna:
        X1_ = X1[subset].copy().fillna(sentinel)
        X2_ = X2[subset].copy().fillna(sentinel)
        X12_ = pd.concat([X1_, X2_])
    else:
        X1_ = X1[subset].copy()
        X2_ = X2[subset].copy()
        X12_ = pd.concat([X1_, X2_])

    # Step 2: Create X1* and X2* by keeping the first row from each group in X1_ and X2_
    X1_star = create_representative_rows(df=X1_, subset=subset)
    X2_star = create_representative_rows(df=X2_, subset=subset)
    X12_star = create_representative_rows(df=X12_, subset=subset)

    # Step 3: Merge X1* and X2* on the subset columns and find rows in X2* that don't match X1*
    Z = X2_star.merge(X1_star, on=subset, how='left', indicator=True)

    # Step 4: Keep only the rows that are in X2* but not in X1*
    Z = Z[Z['_merge'] == 'left_only']

    Z = Z[subset]

    return X12_star, X1_star, X2_star, Z


def find_matching_rows(A: pd.DataFrame, B: Union[pd.DataFrame, pd.Series], subset: List[str]) -> pd.DataFrame:
    """
    Find rows in DataFrame A that have matching rows in DataFrame B based on a specified subset of columns.
    This function ensures that the index of DataFrame A is preserved in the output DataFrame C.

    Parameters:
        A (pd.DataFrame): The first dataframe.
        B (Union[pd.DataFrame, pd.Series]): The second dataframe or a single-row series to match against.
        subset (List[str]): The list of columns to use for matching.

    Returns:
        pd.DataFrame: A dataframe containing rows from A that have matches in B on the specified subset of columns,
                      maintaining the original index of A.
    """

    # Convert Series to DataFrame if necessary
    if isinstance(B, pd.Series):
        B = B.to_frame().transpose()

    # Ensure subset exists in both A and B
    subset = [col for col in subset if col in A.columns and col in B.columns]

    # Save A's index in a column to preserve it through the merge
    A_indexed = A.copy()
    A_indexed['Original_Index'] = A_indexed.index

    # Prepare B by selecting relevant columns and dropping duplicates
    if isinstance(B, pd.DataFrame):
        B_ = B[subset].drop_duplicates()

    # Merge A with B, ensuring the subset is used for matching
    C = A_indexed.merge(B_, on=subset, how='inner')

    # Set the 'Original_Index' column as the actual index of C
    C.set_index('Original_Index', inplace=True)

    C.index.name = None  # Remove the index name
    return C


def train_val_feature_combination_table(instance: Type[model],
                                        train_idx: np.ndarray,
                                        val_idx: np.ndarray,
                                        classes: Union[None, List] = None,
                                        features: Union[None, List] = None,
                                        ignore: Union[None, List] = None,
                                        fillna: bool = True,
                                        sentinel: int = -1, ) -> None:
    X1 = instance.X_train.iloc[train_idx]
    X2 = instance.X_train.iloc[val_idx]
    y1 = instance.y_train.iloc[train_idx]
    y2 = instance.y_train.iloc[val_idx]

    if classes is None:
        classes = list(y1.iloc[:, 0].unique())
    if fillna:
        train_all = X1[y1.iloc[:, 0].isin(classes)].copy().fillna(sentinel)
        val_all = X2[y2.iloc[:, 0].isin(classes)].copy().fillna(sentinel)
    else:
        train_all = X1[y1.iloc[:, 0].isin(classes)].copy()
        val_all = X2[y2.iloc[:, 0].isin(classes)].copy()

    train_plus_val_all = pd.concat([train_all, val_all])

    train_plus_val_distinct, train_distinct, val_distinct, val_unseen_distinct = compare_feature_combinations(
        X1=train_all,
        X2=val_all,
        features=features,
        ignore=ignore,
        fillna=fillna,
        sentinel=sentinel, )

    val_unseen_all = find_matching_rows(A=val_all, B=val_unseen_distinct, subset=val_unseen_distinct.columns)
    val_seen_all = find_matching_rows(A=val_all, B=train_distinct, subset=train_distinct.columns)

    table = pd.DataFrame()
    table[''] = ['train + val',
                 'train',
                 'val',
                 'val unseen',
                 'val seen',
                 'seen/unseen']
    table['all'] = [len(train_plus_val_all),
                    len(train_all),
                    len(val_all),
                    len(val_unseen_all),
                    len(val_seen_all),
                    len(val_seen_all) / len(val_unseen_all)]
    table['distinct'] = [len(train_plus_val_distinct),
                         len(train_distinct),
                         len(val_distinct),
                         len(val_unseen_distinct),
                         len(val_distinct) - len(val_unseen_distinct),
                         (len(val_distinct) - len(val_unseen_distinct)) / len(val_unseen_distinct)]
    table['ratio'] = table['all'] / table['distinct']

    # Function to format numbers, handling strings and numbers differently
    def format_cell(x):
        if pd.api.types.is_number(x):
            if x == int(x):  # Check if the number is an integer
                return f"{int(x):,}"  # Format as an integer with comma separators
            else:
                return f"{x:,.4f}"  # Format as a float with two decimals
        return x  # Return the value unchanged if it's not a number

    formatted_table = table.map(format_cell)
    formatted_table.iloc[-1, -1] = '-'

    display(formatted_table)


def feature_combination_analysis(instance: Type[model],
                                 train_idx: np.ndarray,
                                 val_idx: np.ndarray,
                                 features: Union[None, List] = None,
                                 ignore: Union[None, List] = None,
                                 two_missing_values_considered_equal: bool = True,
                                 sentinel: int = -1, ) -> None:
    if ignore:
        print(f"- Ignoring: {ignore}")
    if two_missing_values_considered_equal:
        print(f"- Replacing NaN values by {sentinel} so that two missing values are considered the same.")
    print(
        "- Two feature vectors are considered identical if they agree in all coordinates except, possibly, those corresponding to ignored feature(s).")
    print("- Otherwise, two feature vectors are considered distinct.")
    print("- Table rows...: 'train + val' all records, whether from training set or validation set")
    print("                 'train' records from training set only")
    print("                 'val' records from validation set only")
    print(
        "                 'val seen' validation set records with feature combinations that also appear in the training set")
    print(
        "                 'val unseen' validation set records with feature combinations that do not appear in the training set")
    print("                 'seen/unseen' ratio of val seen to val unseen")
    print("- Table columns: 'all' records (feature combinations counted with multiplicity)")
    print(
        "                 'distinct' feature combinations among all records (feature combinations counted without multiplicity)")
    print(
        "                 'ratio' or proportion of all records that are distinct (average multiplicity of each feature combination)")

    for i in [None] + [[int(k)] for k in np.sort(instance.y_train.iloc[:, 0].unique())]:
        classes: Union[None, List] = i
        if i:
            print(f"\nClass {i[0]}".upper())
        else:
            print(f"\nAll".upper())
        train_val_feature_combination_table(instance=instance,
                                            train_idx=train_idx,
                                            val_idx=val_idx,
                                            classes=classes,
                                            features=features,
                                            ignore=ignore,
                                            fillna=two_missing_values_considered_equal,
                                            sentinel=sentinel)


def feat_comb_mult(X: pd.DataFrame,
                   y: Union[None, pd.DataFrame] = None,
                   features: Union[None, List[str]] = None,
                   ignore: Union[None, List[str]] = None,
                   drop_row_if_more_than_this_many_missing_values: Union[None, int] = None,
                   two_missing_values_considered_equal: bool = True,
                   sentinel: int = -1) -> pd.DataFrame:
    """
    Analyze feature combinations in a DataFrame by handling missing values, grouping data,
    and optionally merging with a target DataFrame to enrich feature-based analysis.

    Parameters:
        X (pd.DataFrame): The primary DataFrame to process.
        y (pd.DataFrame, optional): An optional target DataFrame to merge with for adding class distinction analysis.
        features (List[str], optional): Specific features to consider; defaults to all columns in X if None.
        ignore (List[str], optional): Features to ignore when considering combinations.
        drop_row_if_more_than_this_many_missing_values (int, optional): Maximum allowed number of missing values per row;
            rows exceeding this will be dropped, considering only specified features.
        two_missing_values_considered_equal (bool): If True, treats all missing values as equal, replacing them with `sentinel`.
        sentinel (int): Value to replace missing values with if `two_missing_values_considered_equal` is True.

    Returns:
        pd.DataFrame: A DataFrame with processed data including 'multiplicity' of feature combinations and,
        if `y` is provided, 'distinct classes' per feature combination.
    """
    # Determine the subset of features to consider, removing any to ignore
    if features is None:
        features = list(X.columns)
    if ignore is None:
        ignore = []
    subset = list(set(features).intersection(set(X.columns)) - set(ignore))

    # Merge with `y` DataFrame if provided to add information on distinct classes
    if y is not None:
        if not all(X.index == y.index):
            print("X-y index mismatch. Ensure that X and y have the same indices.")
            return
        X_ = X[subset].copy()
        target = y.columns[0]
        X_[target] = y
    else:
        X_ = X[subset].copy()

    # Drop rows based on the total count of missing values within specified features
    if drop_row_if_more_than_this_many_missing_values is not None:
        counts = X_[subset].isna().sum(axis=1)  # Count NaNs only within the subset
        X_ = X_[counts <= drop_row_if_more_than_this_many_missing_values]

    # Handle missing values and compute multiplicity of each unique feature combination
    if two_missing_values_considered_equal:
        X_ = X_.fillna(sentinel)

    # Group by subset to calculate multiplicity, then merge with distinct classes if y is not None
    multiplicity = X_.groupby(subset).size().reset_index(name='multiplicity')
    if y is not None:
        distinct_classes = X_.groupby(subset).nunique().reset_index()
        distinct_classes['distinct classes'] = distinct_classes.drop(columns=subset).sum(axis=1)
        multiplicity = multiplicity.merge(distinct_classes[subset + ['distinct classes']], on=subset, how='left')

    # Replace sentinel with NaN to clean up the final DataFrame
    multiplicity = multiplicity.replace(sentinel, np.nan)

    return multiplicity


def extreme_examples(instance: Type[model],
                     train_idx: np.ndarray,
                     val_idx: np.ndarray,
                     features: Union[None, list] = None,
                     ignore: Union[None, list] = None,
                     drop_row_if_more_than_this_many_missing_values: int = 0,
                     two_missing_values_considered_equal: bool = True,
                     sentinel: int = -1,
                     distinct_classes: int = 1,
                     high_to_low_multiplicity: bool = True,
                     ) -> None:
    X_train = instance.X_train.iloc[train_idx]
    y_train = instance.y_train.iloc[train_idx]
    X_val = instance.X_train.iloc[val_idx]
    y_val = instance.y_train.iloc[val_idx]

    Z = feat_comb_mult(X=X_val,
                       y=y_val,
                       features=features,
                       ignore=ignore,
                       drop_row_if_more_than_this_many_missing_values=drop_row_if_more_than_this_many_missing_values,
                       two_missing_values_considered_equal=two_missing_values_considered_equal,
                       sentinel=sentinel, )

    Z1 = Z[Z['distinct classes'] == distinct_classes]
    Z2 = Z1['multiplicity'].sort_values(ascending=not high_to_low_multiplicity)

    for i, multiplicity in enumerate(Z2):
        feature_combination = Z1.loc[Z2.index[i]][:-2]

        train_match = find_matching_rows(A=X_train,
                                         B=feature_combination,
                                         subset=feature_combination.index, )

        val_match = find_matching_rows(A=X_val,
                                       B=feature_combination,
                                       subset=feature_combination.index, )

        if not train_match.empty and not val_match.empty:
            train_pred_df = inference(instance=instance,
                                      X=train_match,
                                      y=instance.y_train.loc[train_match.index], );

            val_pred_df = inference(instance=instance,
                                    X=val_match,
                                    y=instance.y_train.loc[val_match.index], );

            target = y_train.columns[0]
            train_class_distribution = combined_value_counts(df=train_pred_df, column=target)
            val_class_distribution = combined_value_counts(df=val_pred_df, column=target)
            best_case_train = int(train_class_distribution['%'].max())
            best_case_val = int(val_class_distribution['%'].max())
            predicted_probabilities = filter_columns_by_substrings(train_pred_df, ['prob', 'pred']).iloc[0].to_frame().T
            predicted_probabilities = format_floats_as_integers(predicted_probabilities)
            pred = predicted_probabilities['pred'].iloc[0]
            try:
                pred_accuracy = val_class_distribution.loc[val_class_distribution[target] == pred, '%'].values[0]
            except:
                pred_accuracy = 0
            if predicted_probabilities.columns.str.contains('adj').any():
                try:
                    adj_pred = predicted_probabilities['adj pred'].iloc[0]
                    adj_pred_accuracy = \
                    val_class_distribution.loc[val_class_distribution[target] == adj_pred, '%'].values[0]
                except:
                    adj_pred_accuracy = 0

            enough_training_samples: int = 5
            not_enough_training_samples: int = 4
            look_for_similar_samples: bool = False

            print("- The feature combination\n")
            display(feature_combination.to_frame().T.style.format(precision=0, na_rep='').hide(axis='index'))
            print(f"\n  appears {len(train_match)} times in training, and {len(val_match)} times in validation.")
            print("\n- For this specific feature combination, in the training set, the class distribution is\n")
            display(train_class_distribution.style.hide(axis="index"))
            if len(train_match) >= enough_training_samples:
                print("\n- A reasonable model might assign probabilities to each class according to this distribution.")
            if len(train_match) <= not_enough_training_samples:
                look_for_similar_samples = True
                print(
                    "\n- A model might not be able to assign meaningful probabilities to each class, unless there are 'similar' feature combinations in the training data.")
            print("\n- For the same feature combination, in the validation set, the class distribution is \n")
            display(val_class_distribution.style.hide(axis="index"))
            if best_case_train < 100 and best_case_val < 100:
                print(
                    f"\n- No model can correctly classify more than ~{best_case_train}% (resp. ~{best_case_val}%) of this training (resp. validation) sample.".upper())
            elif best_case_train < 100:
                print(
                    f"\n- No model can correctly classify more than ~{best_case_train}% of this training sample.".upper())
            elif best_case_val < 100:
                print(
                    f"\n- No model can correctly classify more than ~{best_case_val}% of this validation sample.".upper())
            else:
                pass
            print("\n- Our classifier gives the following probabilities and prediction(s):\n")
            display(predicted_probabilities.style.hide(axis='index'))
            print(
                f"\n- Prediction based on maximum probability correctly classifies {pred_accuracy:.2f}% of this validation sample.")
            if adj_pred_accuracy is not None:
                print(
                    f"\n- Prediction based on adjusted thresholds correctly classifies {adj_pred_accuracy:.2f}% of this validation sample.")

            total_matches: int = len(feature_combination)
            while look_for_similar_samples and total_matches >= 0:
                total_matches -= 1
                similar_df = similar_feature_combinations(df=X_train,
                                                          feature_combination=feature_combination,
                                                          exact_matches=None,
                                                          total_matches=total_matches,
                                                          match_condition='>=', )

                if len(similar_df) >= enough_training_samples:
                    look_for_similar_samples = False

            if total_matches > 0:
                print("\nSimilar feature combinations".upper())
                print(
                    f"\n- Training set contains {len(similar_df)} records that agree with the above feature combination on {total_matches} features.")
                similar_pred_df = inference(instance=instance,
                                            X=similar_df,
                                            y=instance.y_train.loc[similar_df.index], );
                similar_class_distribution = combined_value_counts(df=similar_pred_df, column=target)
                print(f"\n- Over these similar feature combinations, in the training set, the class distribution is\n")
                display(similar_class_distribution.style.hide(axis="index"))
                print(f"\n- Does this reflect the probabilities given by the classifer? Should it?")

            response = input("\nPress enter to continue or enter c to cancel:")
            if response == 'c':
                break
            else:
                clear_output()


def similar_feature_combinations(df: pd.DataFrame,
                                 feature_combination: pd.Series,
                                 exact_matches: Union[None, list],
                                 total_matches: Union[None, int],
                                 match_condition: str = '=', ) -> pd.DataFrame:
    """
    Filter a DataFrame to return a subset of rows based on specified matching conditions.

    This function identifies rows in the DataFrame that meet two conditions:
    - Rows where specific columns match exactly with the values specified in `exact_matches`.
    - Rows where the number of columns matching the values in `feature_combination` is compared to `total_matches`
      using the specified `match_condition` operator.

    Parameters:
    df (pd.DataFrame): The input DataFrame to filter.
    feature_combination (pd.Series): A Series where the index specifies columns to check, and the values
                                     are the target values for comparison.
    exact_matches (list or None): A list of column names that must match exactly between `df` and
                                  `feature_combination`.
    total_matches (int or None): The target number of columns in `feature_combination` that must match in each row.
                                 If None, defaults to the length of `exact_matches` if provided,
                                 otherwise to the length of `feature_combination`.
    match_condition (str): Specifies the comparison operator for `feature_matches_count` relative to `total_matches`.
                           Allowed values are:
                           - '=': Rows must have exactly `total_matches` columns matching.
                           - '<': Rows must have fewer than `total_matches` columns matching.
                           - '>': Rows must have more than `total_matches` columns matching.
                           - '<=': Rows must have at most `total_matches` columns matching.
                           - '>=': Rows must have at least `total_matches` columns matching.

    Returns:
    pd.DataFrame: A subset of `df` where each row meets the conditions specified by `exact_matches`, `total_matches`,
                  and `match_condition`.

    Raises:
    ValueError: If `match_condition` is invalid or if `total_matches` violates constraints with `feature_combination`
                or `exact_matches`.
    """

    if match_condition not in ['=', '<', '>', '<=', '>=']:
        raise ValueError("Error: match_condition must be one of '=', '<', '>', '<=', '>='.")
    # Ensure feature_combination columns are part of df
    if not set(feature_combination.index).issubset(df.columns):
        raise ValueError("Error: Some features in feature_combination are not in df columns.")

    # Ensure exact_matches columns are within feature_combination
    if exact_matches is not None and not set(exact_matches).issubset(feature_combination.index):
        raise ValueError("Error: Some features in exact_matches are not in feature_combination.")

    # Default total_matches if it's None
    if total_matches is None:
        total_matches = len(set(exact_matches)) if exact_matches else len(feature_combination)

    if match_condition == '=' or match_condition == '<=':
        if exact_matches is not None and total_matches < len(exact_matches):
            raise ValueError(
                "Error: total_matches should be greater than or equal to the number of features in exact_matches.")

    if match_condition == '<':
        if exact_matches is not None and total_matches <= len(exact_matches):
            raise ValueError("Error: total_matches should be greater than the number of features in exact_matches.")

    if match_condition == '=' or match_condition == '>=':
        if total_matches > len(feature_combination):
            raise ValueError(
                "Error: total_matches should be less than or equal to the number of features in feature_combination.")

    if match_condition == '>':
        if total_matches >= len(feature_combination):
            raise ValueError("Error: total_matches should be less than the number of features in feature_combination.")

    # Create boolean masks for feature matches
    feature_mask = df[feature_combination.index] == feature_combination.values

    # Apply the exact match conditions if applicable
    if exact_matches is not None and len(exact_matches) > 0:
        # Ensure ordering is consistent
        exact_match_columns = [col for col in exact_matches if col in feature_combination.index]
        exact_match_mask = df[exact_match_columns] == feature_combination[exact_match_columns].values

    # Sum the True values row-wise to count how many columns match in each row
    feature_matches_count = feature_mask.sum(axis=1)

    # Filter rows where the number of feature matches equals total_matches
    if match_condition == '=':
        filtered_df = df[feature_matches_count == total_matches]
    elif match_condition == '<=':
        filtered_df = df[feature_matches_count <= total_matches]
    elif match_condition == '<':
        filtered_df = df[feature_matches_count < total_matches]
    elif match_condition == '>':
        filtered_df = df[feature_matches_count > total_matches]
    else:
        filtered_df = df[feature_matches_count >= total_matches]

    # Further restrict rows to those that match exact_matches
    if exact_matches is not None and len(exact_matches) > 0:
        filtered_df = filtered_df[exact_match_mask.all(axis=1)]

    return filtered_df


def min_distance_approx(df: pd.DataFrame) -> pd.Series:
    """
    Calculate the minimum distance between each row and all other rows using
    a custom distance metric, where the distance is defined as the number of
    columns in which the values differ between two rows.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.Series: A Series where the index corresponds to each row and the value
               is the minimum distance to any other row in the DataFrame.
    """
    # Convert DataFrame to NumPy array for computation
    df_array = df.to_numpy()

    # Use Nearest Neighbors algorithm with Hamming distance
    nbrs = NearestNeighbors(n_neighbors=2, metric='hamming').fit(df_array)

    # Find the indices and distances of the nearest neighbors
    distances, indices = nbrs.kneighbors(df_array)

    # Convert fractional Hamming distance to differing column count
    differing_columns_count = distances[:, 1] * df.shape[1]  # Multiply by the number of columns

    # Return as a pandas Series
    return pd.Series(differing_columns_count, index=df.index)
