import pandas as pd
import numpy as np
from typing import Union, Tuple
import copy

def weighted_average_f(beta: Union[float, None], 
                       weights: Union[np.ndarray, None], 
                       precision: Union[np.ndarray, pd.DataFrame], 
                       recall: Union[np.ndarray,pd.DataFrame]) -> float:
    
    assert np.ravel(precision).shape == np.ravel(recall).shape, 'precision and recall arrays must be of the same length'
    
    if precision.all() == 0 or recall.all() == 0:
        return np.nan
    
    if beta is None:
        beta = 1
    
    if weights is None:
        weights = np.ones(np.ravel(precision).shape)
        
    assert np.ravel(precision).shape == np.ravel(weights).shape, 'weights must have the same length as precision and recall'
    if weights.all() == 0:
        return np.nan
    
    fbeta = (1 + beta**2)/(1/precision + beta**2/recall)
    
    weighted_average_fbeta = np.sum(weights*fbeta)/np.sum(weights)
    
    return weighted_average_fbeta

def confusion_matrix_with_weighted_fbeta(AxB: Union[pd.DataFrame,dict],
                                        beta: Union[float, None] = None,
                                        weights: Union[np.ndarray, None] = None,
                                        percentage: bool = False,) -> pd.DataFrame:
    
    # Make a copy of AxB so that this process does not alter it.
    df = copy.deepcopy(AxB)
    
    if type(df) == dict:
        df = pd.DataFrame(df)
    
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

    diagonal_entries = np.diag(df.values)
    # Add a recall column 
    last_col_name = df.columns[-1]
    df['recall'] = diagonal_entries/df[last_col_name]

    # And a precision row
    diagonal_entries = np.append(diagonal_entries, [0])
    df.loc['precision'] = diagonal_entries/df.iloc[-1]

    # Extract the recall and precision arrays
    recall = df['recall'][:-2]
    precision = df.loc['precision'][:-2] 

    # Calculate the weighted average fbeta score and put it in the bottom right corner of df
    try:
        df.iloc[-1, -1] = weighted_average_f(beta, weights, precision, recall)
    except Exception as e:
        print(f"Error calculating weighted average fbeta score: {e}")

    # df should have originally been all integers, but now we have floats, so let's format the original part
    df.iloc[:-1,:-1,] = df.iloc[:-1,:-1,].map(lambda x : f"{int(x):,}") # applymap
    '''
    Throws: 'FutureWarning: Setting an item of incompatible dtype is deprecated and will raise an error in a future version of pandas. 
    Value '['44,163' '6,961' '165' '51,289']' has dtype incompatible with float64, please explicitly cast to a compatible dtype first.'
    Is this a bug? See https://github.com/pandas-dev/pandas/issues/55025
    Workaround: added df = df.astype(object) above to avoid the warning. 
    '''

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
    prob_prediction_given_label_with_sum = pd.concat([prob_prediction_given_label, row_sums], axis = 1)

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
    filtered_label_given_prediction_combined_df.columns = converted_columns.where(pd.notnull(converted_columns), filtered_label_given_prediction_combined_df.columns)
    # Convert float column names to integers
    filtered_label_given_prediction_combined_df.columns = [int(col) if isinstance(col, float) else col for col in filtered_label_given_prediction_combined_df.columns]

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
    filtered_prediction_given_label_combined_df.columns = converted_columns_2.where(pd.notnull(converted_columns_2), filtered_prediction_given_label_combined_df.columns)
    # Convert float column names to integers
    filtered_prediction_given_label_combined_df.columns = [int(col) if isinstance(col, float) else col for col in filtered_prediction_given_label_combined_df.columns]

    # Finally, we format so that (a,b) is displayed as a (↓b%) for the label given prediction probabilities.
    # (The down arrow aids the viewer, indicating that one should be looking down a given column (prediction)).
    def format_label_given_prediction(a, b):
        return f"{a} (↓{b}%)"

    label_given_prediction_formatted_df = (
        filtered_label_given_prediction_combined_df.map(
            lambda x: format_label_given_prediction(*x) if isinstance(x, tuple) else x
        ) # applymap
    )
    label_given_prediction_formatted_df.columns.name = "prediction"
    label_given_prediction_formatted_df.index.name = "actual"
    # For prediction given label, we look along a row, hence the right arrow.
    def format_prediction_given_label(a, b):
        return f"{a} (→{b}%)"

    prediction_given_label_formatted_df = (
        filtered_prediction_given_label_combined_df.map(
            lambda x: format_prediction_given_label(*x) if isinstance(x, tuple) else x
        ) # applymap
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