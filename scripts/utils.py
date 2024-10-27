import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
import inspect
import pandas as pd
import matplotlib.pyplot as plt

try:
    from IPython import get_ipython

    if get_ipython():
        from IPython.display import HTML
        import ipywidgets as widgets
    else:
        widgets = None


        def HTML(*args, **kwargs):
            print("HTML display is not available.")
except ImportError:
    widgets = None


    def HTML(*args, **kwargs):
        print("HTML display is not available.")


def set_widgets(enable: bool = True, disable: bool = False) -> None:
    global widgets
    global HTML
    if enable:
        try:
            from IPython import get_ipython
            if get_ipython():
                from IPython.display import HTML
                import ipywidgets as widgets
            else:
                widgets = None

                def HTML(*args, **kwargs):
                    print("HTML display is not available.")
        except ImportError:
            widgets = None

            def HTML(*args, **kwargs):
                print("HTML display is not available.")
    if disable:
        widgets = None

        def HTML(*args, **kwargs):
            print("HTML display is not available.")


def get_subdirectories(base_path: Path,
                       depth: int = 0,
                       ignore: List[str] = None) -> dict:
    """
    Creates a dictionary with keys as the immediate subdirectories of a specified directory
    and values as Path objects pointing to those directories, ignoring specified subdirectories,
    and limiting to a specified depth.

    Args:
        base_path (Path): The path to the base directory.
        depth (int, optional): The depth of subdirectories to include. Defaults to 0.
        ignore (List[str], optional): A list of subdirectory names to ignore. Defaults to None.

    Returns:
        dict: A dictionary with subdirectory names as keys and Path objects as values.
    """
    if not base_path.is_dir():
        raise ValueError(f"The path {base_path} is not a directory.")

    if ignore is None:
        ignore = ['__pycache__', '.ipynb_checkpoints']

    def get_subdirs_at_depth(current_path: Path, current_depth: int) -> dict:
        if current_depth == depth:
            return {
                subdir.name: subdir
                for subdir in current_path.iterdir()
                if subdir.is_dir() and subdir.name[0] not in {'.', '_'} and subdir.name not in ignore
            }

        subdirs = {}
        for subdir in current_path.iterdir():
            if subdir.is_dir() and subdir.name[0] not in {'.', '_'} and subdir.name not in ignore:
                deeper_subdirs = get_subdirs_at_depth(subdir, current_depth + 1)
                subdirs.update(deeper_subdirs)
        return subdirs

    return get_subdirs_at_depth(base_path, 0)


def get_directory_tree(base_path: Path,
                       base_name: str,
                       print_paths: bool = True,
                       ignore: List[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Prints a directory tree and creates dictionaries mapping directory names to path objects.

    Args:
        base_path (Path): The base path to start from.
        base_name (str): The name to use for the base path in the returned dictionary.
        print_paths (bool): If True, print the paths.
        ignore (List[str]): List of directory names to ignore.

    Returns:
        Tuple[Dict[str, Any], str]: A tuple containing the nested dictionary structure representing the directory tree and the string representation of the tree.
    """
    if ignore is None:
        ignore = ['__pycache__', '.ipynb_checkpoints']

    def create_path_dict(path: Path) -> Dict[str, Any]:
        """
        Recursively creates a nested dictionary of paths starting from the given path, including only directories.
        """
        path_dict = {}
        for item in path.iterdir():
            if item.is_dir() and item.name[0] not in {'.', '_'} and item.name not in ignore:
                subdirectory = create_path_dict(item)
                path_dict[item.name] = {"path": item, "subdirectories": subdirectory}
        return path_dict

    # Start with the base path
    path_structure = {base_name: {"path": base_path, "subdirectories": create_path_dict(base_path)}}

    def print_path_dict(d: Dict[str, Any], prefix: str = "", output: str = "") -> str:
        """
        Recursively prints the nested dictionary of paths in a tree format and accumulates the output in a string.
        """
        for key, value in d.items():
            if isinstance(value, dict):
                output += prefix + "├─ " + key + "/\n"
                output = print_path_dict(value["subdirectories"], prefix + "│  ", output)
            else:
                output += prefix + "└─ " + key + ": " + str(value['path']) + "\n"
        return output

    tree_output = print_path_dict(path_structure[base_name]['subdirectories'])

    if print_paths:
        print(tree_output)

    return path_structure, tree_output


# USEFUL FUNCTIONS

try:
    from IPython.display import display as ipy_display
except ImportError:
    ipy_display = None


def display(*args):
    if ipy_display is not None:
        for arg in args:
            if arg is not None:
                ipy_display(arg)
    else:
        for arg in args:
            if arg is not None:
                print(arg)


# def display(*args):
#     try:
#         # Check if running in a Jupyter Notebook environment
#         from IPython.display import display as ipy_display
#
#         # Display each argument using IPython display function
#         for arg in args:
#             ipy_display(arg)
#     except ImportError:
#         # If IPython is not available, just print each argument
#         for arg in args:
#             print(arg)

try:
    from IPython.display import clear_output as ipy_clear_output
except ImportError:
    ipy_clear_output = None
try:
    from IPython.display import clear_output as ipy_clear_output
except ImportError:
    ipy_clear_output = None


def clear_output():
    if ipy_clear_output is not None:
        ipy_clear_output(wait=True)
    else:
        # Fallback to clearing the terminal if not in a Jupyter environment
        import os
        if os.name == "nt":  # Windows
            os.system("cls")
        else:  # Unix-based systems (Linux, macOS)
            os.system("clear")


def get_variable_name(var):
    # Get the name of the variable using the inspect module
    for name, value in inspect.currentframe().f_back.f_locals.items():
        if value is var:
            return name
    return None  # Variable name not found


def save_df(df: pd.DataFrame, save_as: Union[str, Path], tack_on: Union[str, None] = None) -> None:
    # If filename is a Path object, convert it to a string
    if isinstance(save_as, str):
        save_as = Path(save_as)

    # Insert suffix before the file extension
    if tack_on is not None:
        save_as = save_as.with_name(f"{save_as.stem}_{tack_on}{save_as.suffix}")

    # Determine the file extension
    file_extension = save_as.suffix.lower()

    # Choose the saving method based on the file extension
    if file_extension == '.csv':
        df.to_csv(save_as, index=False)
    elif file_extension == '.xlsx':
        df.to_excel(save_as, index=False)
    elif file_extension == '.html':
        df.to_html(save_as, index=False)
    elif file_extension == '.json':
        df.to_json(save_as, orient='records')
    elif file_extension == '.png':
        plt.axis('off')  # Turn off the axis
        plt.table(cellText=df.values, colLabels=df.columns, loc='center')  # Display DataFrame as a table
        plt.savefig(save_as)  # Save as PNG
    elif file_extension == '.txt':
        # Convert DataFrame to dictionary
        df_dict = df.to_dict()
        # Save dictionary as text file
        with open(save_as, 'w') as file:
            file.write(str(df_dict))
    else:
        print(f"Unsupported file extension: {file_extension}. Please use csv, xlsx, html, or json.")


def print_header(header: str) -> None:
    length = len(header)
    output_sequence = ["", "=" * length, header.upper(), "=" * length, ""]
    output = '\n'.join(output_sequence)
    print(output)


def combined_value_counts(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Generate a DataFrame showing the count and percentage of occurrences
    for each unique value in a specified column of the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    column (str): The column name for which value counts are to be calculated.

    Returns:
    pd.DataFrame: A DataFrame with the unique values in the specified column,
                  their corresponding counts, and their proportions (as percentages).
    """
    value_counts_combined = pd.DataFrame({
        'count': df[column].value_counts(),
        '%': 100 * df[column].value_counts(normalize=True)
    }).reset_index()

    value_counts_combined.rename(columns={'index': column}, inplace=True)

    return value_counts_combined


def filter_columns_by_substrings(df: pd.DataFrame, substrings: list) -> pd.DataFrame:
    """
    Filter and return a DataFrame containing only the columns whose names
    include any of the specified substrings.

    Parameters:
    df (pd.DataFrame): The input DataFrame to filter.
    substrings (list): A list of substrings to look for in column names.

    Returns:
    pd.DataFrame: A DataFrame containing only the columns that match any of the substrings.
    """
    filtered_columns = [col for col in df.columns if any(sub in col for sub in substrings)]

    return df[filtered_columns]


def format_floats_as_integers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format floats in the DataFrame as integers if the float value is an integer.

    Parameters:
    df (pd.DataFrame): The input DataFrame to format.

    Returns:
    pd.DataFrame: A DataFrame with float values formatted as integers where applicable.
    """

    def format_value(x):
        if isinstance(x, float) and x.is_integer():
            return int(x)
        return x

    return df.map(format_value)
