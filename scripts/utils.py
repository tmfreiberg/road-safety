import os
from pathlib import Path
import inspect
import pandas as pd
from typing import Union
import matplotlib.pyplot as plt

# SETTING PATHS IN A PLATFORM-INDEPENDENT WAY

class path_setup:
    default_env_var = "ROAD_SAFETY"
    @staticmethod
    def subfolders(
        base_path: Path, base_name: str = "project", Print: bool = True
    ) -> dict:
        folders = [
            f
            for f in os.listdir(base_path)
            if os.path.isdir(base_path.joinpath(f)) and f[0] != "."
        ]
        path = dict.fromkeys(folders)
        path[base_name] = base_path
        if Print:
            print("\nDictionary 'path' contains paths to subdirectories\n".upper())
            print(f"path['{base_name}'] : {path[base_name]}")
        for folder in folders:
            path[folder] = base_path.joinpath(folder)
            if Print:
                print(f"path['{folder}'] : {path[folder]}")
        return path

# USEFUL FUNCTIONS

def display(*args):
    try:
        # Check if running in a Jupyter Notebook environment
        from IPython.display import display as ipy_display

        # Display each argument using IPython display function
        for arg in args:
            ipy_display(arg)
    except ImportError:
        # If IPython is not available, just print each argument
        for arg in args:
            print(arg)

def clear_output():
    # Check if running in a Jupyter Notebook environment
    try:
        from IPython.display import clear_output as ipy_clear_output

        if ipy_clear_output is not None:
            ipy_clear_output(wait=True)
            return
    except ImportError:
        pass  # Continue if IPython is not available or running in a script

    # Check the operating system and clear output accordingly
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # Unix-based systems (Linux, macOS)
        os.system("clear")

    return

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
    output_sequence = ["", "="*length, header.upper(), "="*length, ""]
    output = '\n'.join(output_sequence)
    print(output)        
