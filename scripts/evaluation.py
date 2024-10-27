import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, List
import copy
from collections import OrderedDict


def weighted_average_f(beta: Union[np.ndarray, None],  # Union[float, None],
                       weights: Union[np.ndarray, None],
                       precision: Union[np.ndarray, pd.DataFrame],
                       recall: Union[np.ndarray, pd.DataFrame]) -> float:
    assert np.ravel(precision).shape == np.ravel(recall).shape, 'precision and recall arrays must be of the same length'

    if precision.all() == 0 or recall.all() == 0:
        return np.nan

    if beta is None:
        beta = np.ones(np.ravel(precision).shape)

    if weights is None:
        weights = np.ones(np.ravel(precision).shape)

    assert np.ravel(precision).shape == np.ravel(
        weights).shape, 'weights must have the same length as precision and recall'
    if weights.all() == 0:
        return np.nan

    fbeta = (1 + beta ** 2) / (1 / precision + beta ** 2 / recall)

    weighted_average_fbeta = np.sum(weights * fbeta) / np.sum(weights)

    return weighted_average_fbeta


def confusion_matrix_with_weighted_fbeta(AxB: Union[pd.DataFrame, dict],
                                         beta: Union[np.ndarray, None] = None,
                                         weights: Union[np.ndarray, None] = None,
                                         percentage: bool = False, ) -> pd.DataFrame:
    # Make a copy of AxB so that this process does not alter it.
    df = copy.deepcopy(AxB)

    if type(df) == dict:
        df = pd.DataFrame.from_dict(df)

    df = df.astype(object)

    # AxB is supposed to be the result of pd.crosstab(A, B, margins=True).
    # Let's check to see whether both margins are included.
    # If not, add them.
    # Could be some corner cases when the last column of AxB sans margin just so happens to be the sum of the previous columns.
    # We won't worry about that at this stage.

    is_last_column_sum = df.iloc[:, :-1].sum(axis=1).equals(df.iloc[:, -1])
    if not is_last_column_sum:
        df['All'] = df.sum(axis=1)

    is_last_row_sum = df.iloc[:-1, :].sum(axis=0).equals(df.iloc[-1, :])
    if not is_last_row_sum:
        df.loc['All'] = df.sum(axis=0)

    # Pad if there are classes that are not predicted at all (missing columns)
    all_classes = sorted(set(df.index) - {'All'} | set(df.columns) - {'All'})
    df = df.reindex(index=all_classes + ['All'], columns=all_classes + ['All'], fill_value=0)

    # Calculate recall
    df['recall'] = df.apply(lambda row: row[row.name] / row['All'] if row['All'] != 0 else 0, axis=1)

    # Calculate precision (considering not to add a dummy 0 which might be erroneous)
    precisions = {col: df.loc[col, col] / df.loc['All', col] if df.loc['All', col] != 0 else 0 for col in
                  df.columns[:-1]}
    df.loc['precision'] = pd.Series(precisions)

    # Extract the recall and precision arrays
    recall = df['recall'][:-2]
    precision = df.loc['precision'][:-2]

    # Calculate the weighted average fbeta score and put it in the bottom right corner of df
    try:
        df.iloc[-1, -1] = weighted_average_f(beta, weights, precision, recall)
    except Exception as e:
        print(f"Error calculating weighted average fbeta score: {e}")

    # Insert blanks where we have no meaningful values
    df.iloc[-1, -2] = np.nan
    df.iloc[-2, -1] = np.nan

    # Label the index and the columns
    df = df.rename_axis(index='actual', columns='predicted')

    return df


def custom_confusion(
        labels: pd.Series,
        predictions: pd.Series,
        label_dictionary: dict = None,
        dropna: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not dropna:
        # Replace NaN values with a placeholder before computing crosstabs
        labels = labels.fillna("NaN")
        predictions = predictions.fillna("NaN")

    # Standard crosstabs with margins
    # Convention: ground truth labels along rows, predicted labels along columns
    labels_by_predictions = pd.crosstab(
        labels, predictions, dropna=dropna, margins=True
    )
    # Normalizing by 'columns' will give probability of label given a prediction, e.g. probability of actually having melanoma given the prediction is nevus.
    prob_label_given_prediction = (
        pd.crosstab(
            labels, predictions, normalize="columns", dropna=dropna, margins=True
        )
        .mul(100)
        .round(2)
    )
    # Normalizing by 'rows' (or 'index') will give probability of prediction given a label, e.g. probability of predicting melanoma given oen actually has a nevus.
    prob_prediction_given_label = (
        pd.crosstab(labels, predictions, normalize="index", dropna=dropna, margins=True)
        .mul(100)
        .round(2)
    )

    # The normalized crosstab will not contain column sums.
    # Even each column sums to 100%, we still want this:
    column_sums = pd.DataFrame(
        prob_label_given_prediction.sum(axis=0).round().astype(int), columns=["All"]
    ).T
    prob_label_given_prediction_with_sum = pd.concat(
        [prob_label_given_prediction, column_sums]
    )

    row_sums = pd.DataFrame(
        prob_prediction_given_label.sum(axis=1).round().astype(int), columns=["All"]
    )
    prob_prediction_given_label_with_sum = pd.concat([prob_prediction_given_label, row_sums], axis=1)

    # Now we merge the two label_given_prediction crosstabs, displaying (a,b) in each cell, where a is an integer and b is a percentage.
    label_given_prediction_combined_df = pd.merge(
        labels_by_predictions,
        prob_label_given_prediction_with_sum,
        left_index=True,
        right_index=True,
    )
    for column in label_given_prediction_combined_df.columns:
        if column.endswith("_x"):
            column_name = column[:-2]
            label_given_prediction_combined_df[column_name] = list(
                zip(
                    label_given_prediction_combined_df[column_name + "_x"],
                    label_given_prediction_combined_df[column_name + "_y"],
                )
            )

    filtered_label_given_prediction_combined_df = (
        label_given_prediction_combined_df.filter(regex="^(?!.*(_x|_y)$)")
    )

    # Convert string column names to integers
    converted_columns = pd.to_numeric(filtered_label_given_prediction_combined_df.columns, errors='coerce')
    # Replace NaN values with the original column names
    filtered_label_given_prediction_combined_df.columns = converted_columns.where(pd.notnull(converted_columns),
                                                                                  filtered_label_given_prediction_combined_df.columns)
    # Convert float column names to integers
    filtered_label_given_prediction_combined_df.columns = [int(col) if isinstance(col, float) else col for col in
                                                           filtered_label_given_prediction_combined_df.columns]

    # And similar for the prediction_given_label crosstabs...
    prediction_given_label_combined_df = pd.merge(
        labels_by_predictions,
        prob_prediction_given_label_with_sum,
        left_index=True,
        right_index=True,
    )
    for column in prediction_given_label_combined_df.columns:
        if column.endswith("_x"):
            column_name = column[:-2]
            prediction_given_label_combined_df[column_name] = list(
                zip(
                    prediction_given_label_combined_df[column_name + "_x"],
                    prediction_given_label_combined_df[column_name + "_y"],
                )
            )

    filtered_prediction_given_label_combined_df = (
        prediction_given_label_combined_df.filter(regex="^(?!.*(_x|_y)$)")
    )
    # Convert string column names to integers
    converted_columns_2 = pd.to_numeric(filtered_prediction_given_label_combined_df.columns, errors='coerce')
    # Replace NaN values with the original column names
    filtered_prediction_given_label_combined_df.columns = converted_columns_2.where(pd.notnull(converted_columns_2),
                                                                                    filtered_prediction_given_label_combined_df.columns)
    # Convert float column names to integers
    filtered_prediction_given_label_combined_df.columns = [int(col) if isinstance(col, float) else col for col in
                                                           filtered_prediction_given_label_combined_df.columns]

    # Finally, we format so that (a,b) is displayed as a (↓b%) for the label given prediction probabilities.
    # (The down arrow aids the viewer, indicating that one should be looking down a given column (prediction)).
    def format_label_given_prediction(a, b):
        return f"{a} (↓{b}%)"

    label_given_prediction_formatted_df = (
        filtered_label_given_prediction_combined_df.map(
            lambda x: format_label_given_prediction(*x) if isinstance(x, tuple) else x
        )  # applymap
    )
    label_given_prediction_formatted_df.columns.name = "prediction"
    label_given_prediction_formatted_df.index.name = "actual"

    # For prediction given label, we look along a row, hence the right arrow.
    def format_prediction_given_label(a, b):
        return f"{a} (→{b}%)"

    prediction_given_label_formatted_df = (
        filtered_prediction_given_label_combined_df.map(
            lambda x: format_prediction_given_label(*x) if isinstance(x, tuple) else x
        )  # applymap
    )

    prediction_given_label_formatted_df.columns.name = "prediction"
    prediction_given_label_formatted_df.index.name = "actual"

    # Depending on the situation, we might want numbers, percentages, or the combined table:
    output = {}
    output["labels_by_predictions"] = labels_by_predictions
    output["prob_label_given_prediction"] = prob_label_given_prediction
    output["prob_prediction_given_label"] = prob_prediction_given_label
    output["label_given_prediction_merged"] = label_given_prediction_formatted_df
    output["prediction_given_label_merged"] = prediction_given_label_formatted_df

    if label_dictionary is not None:
        for key, value in output.items():
            value.rename(index=label_dictionary, inplace=True)
            value.rename(columns=label_dictionary, inplace=True)

    return output


def prediction(probabilities: np.ndarray,
               threshold_dict_help: Union[None, OrderedDict[Union[int, str], float]] = None,
               threshold_dict_hinder: Union[None, OrderedDict[Union[int, str], float]] = None) -> np.ndarray:
    """
    Processes the probabilities to predict classes, optionally applying thresholds to adjust probabilities
    before determining the most likely class for each instance.

    Parameters:
        probabilities (np.ndarray): A 2D numpy array where each row represents probabilities across classes.
        threshold_dict_help (OrderedDict[str, float], optional): A dictionary of thresholds to elevate 
            probabilities to 1 if they exceed these values.
        threshold_dict_hinder (OrderedDict[str, float], optional): A dictionary of thresholds to reduce 
            probabilities to 0 if they fall below these values.
    
    Returns:
        np.ndarray: An array of indices indicating the predicted class for each instance.

    Examples:
    ---------
    >>> probabilities = np.array([[0.2, 0.8], [0.6, 0.4]])
    >>> threshold_dict_help = OrderedDict([('1', 0.5)])
    >>> threshold_dict_hinder = OrderedDict([('0', 0.3)])
    >>> prediction(probabilities, threshold_dict_help, threshold_dict_hinder)
    array([1, 1])    
    """
    if threshold_dict_help is None and threshold_dict_hinder is None:
        return np.argmax(probabilities, axis=1)

    adjusted_probabilities = threshold(probabilities=probabilities,
                                       threshold_dict_help=threshold_dict_help,
                                       threshold_dict_hinder=threshold_dict_hinder)
    return np.argmax(adjusted_probabilities, axis=1)


def threshold(probabilities: np.ndarray,
              threshold_dict_help: Union[None, OrderedDict[Union[int, str], float]],
              threshold_dict_hinder: Union[None, OrderedDict[Union[int, str], float]]) -> np.ndarray:
    """
    Adjusts probabilities based on help and hinder thresholds using vectorized operations.

    Parameters:
        probabilities (np.ndarray): 2D array with shape (n_samples, n_classes).
        threshold_dict_help (OrderedDict): Ordered dictionary of thresholds to promote probabilities.
        threshold_dict_hinder (OrderedDict): Ordered dictionary of thresholds to demote probabilities.

    Returns:
        np.ndarray: Adjusted probabilities.
    """
    adjusted_probabilities = probabilities.copy()
    num_rows, num_classes = adjusted_probabilities.shape

    adjusted = np.zeros(num_rows, dtype=bool)  # Tracks whether a row has been adjusted to 1

    # Process threshold_dict_help
    if threshold_dict_help is not None:
        for class_, thres in threshold_dict_help.items():
            class_idx = int(class_)
            # For rows not yet adjusted
            not_adjusted = ~adjusted
            # Rows where probability > threshold
            condition = (adjusted_probabilities[:, class_idx] > thres) & not_adjusted
            # Set probabilities to 1 where condition is met
            adjusted_probabilities[condition, class_idx] = 1
            # Update adjusted status for these rows
            adjusted[condition] = True
            # Note: We continue to the next threshold; the 'adjusted' array ensures that once a row is adjusted, it won't be processed further in threshold_dict_help

    # Process threshold_dict_hinder
    if threshold_dict_hinder is not None:
        for class_, thres in threshold_dict_hinder.items():
            class_idx = int(class_)
            # Rows where the probability is not already set to 1
            not_set_to_one = adjusted_probabilities[:, class_idx] != 1
            # Rows where probability < threshold
            condition = (adjusted_probabilities[:, class_idx] < thres) & not_set_to_one
            # Set probabilities to 0 where condition is met
            adjusted_probabilities[condition, class_idx] = 0

    return adjusted_probabilities
