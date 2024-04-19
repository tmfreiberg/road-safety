# ROAD SAFETY

# Save this as foo.py and run it with 'streamlit run foo.py [-- script args]'.
# Alternatively, run 'python -m streamlit run foo.py'.

import os
from pathlib import Path
import sys

# Otherwise, we use the environment variable on our local system:
project_environment_variable = "ROAD_SAFETY"

# Path to the root directory of the project:
project_path = Path(os.environ.get("ROAD_SAFETY"))

# Relative path to /scripts (from where custom modules will be imported):
scripts_path = project_path.joinpath("scripts")

# Add this path to sys.path so that Python will look there for modules:
sys.path.append(str(scripts_path))

# Now import path_step from our custom utils module to create a dictionary to all subdirectories in our root directory:
from utils import path_setup
path = path_setup.subfolders(base_path = project_path)

import streamlit as st  
import numpy as np 
import pandas as pd 
import joblib
import random

instance = path["models"].joinpath("mtl_3sev_xgb.joblib")

trained = joblib.load(instance)

selection_dictionary = {'MONTH': {1: 1,
  2: 2,
  3: 3,
  4: 4,
  5: 5,
  6: 6,
  7: 7,
  8: 8,
  9: 9,
  10: 10,
  11: 11,
  12: 12},
 'HOUR': {'00:00:00-03:59:00': 0,
  '04:00:00-07:59:00': 1,
  '08:00:00-11:59:00': 2,
  '12:00:00-15:59:00': 3,
  '16:00:00-19:59:00': 4,
  '20:00:00-23:59:00': 5},
 'WKDY_WKND': {'WKDY': 0, 'WKND': 1},
 'NUM_VEH': {1: 0, 2: 1, 9: 2},
 'SPD_LIM': {'<50': 0, 50: 5, 60: 6, 70: 7, 80: 8, 90: 9, 100: 10},
 'ACCDN_TYPE': {'vehicle': 0,
  'pedestrian': 1,
  'cyclist': 2,
  'animal': 3,
  'fixed object': 4,
  'no collision': 5,
  'other': 6},
 'RD_COND': {11: 0,
  12: 1,
  13: 2,
  14: 3,
  15: 4,
  16: 5,
  17: 6,
  18: 7,
  19: 8,
  20: 9,
  99: 10},
 'LIGHT': {1: 3, 2: 2, 3: 1, 4: 0},
 'ZONE': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 9: 6},
 'PUB_PRIV_RD': {1: 0, 2: 1},
 'ASPECT': {'Straight': 0, 'Curve': 1},
 'LONG_LOC': {12: 0, 33: 1, 34: 2, 40: 3, 69: 4, 99: 5},
 'RD_CONFG': {1: 0, 23: 1, 45: 2, 9: 3},
 'RDWX': {'N': 0, 'Y': 1},
 'WEATHER': {11: 0,
  12: 1,
  13: 2,
  14: 3,
  15: 4,
  16: 5,
  17: 6,
  18: 7,
  19: 8,
  99: 9},
 'LT_TRK': {'N': 0, 'Y': 1},
 'HVY_VEH': {'N': 0, 'Y': 1},
 'MTRCYC': {'N': 0, 'Y': 1},
 'BICYC': {'N': 0, 'Y': 1},
 'PED': {'N': 0, 'Y': 1}}

st.markdown("<h1 style='text-align: center; text-transform: uppercase;'>Road accident severity</h1>", unsafe_allow_html=True)

# Create tabs with file folder appearance
tab1, tab2 = st.tabs(["Predictions", "Dictionary"])

# Content for each tab
with tab1:

    # Make predictions and get probabilities
    predictions_placeholder = st.empty()
    probabilities_placeholders = [st.empty() for _ in range(3)]  # One placeholder for each class


    # Display predictions and probabilities at the top of the page
    predictions_placeholder.write("Prediction: severity level ")
    probabilities_placeholders[0].write(f"Probability for severity level 0 (material damage only): ")
    probabilities_placeholders[1].write(f"Probability for severity level 1 (minor injuries): ")
    probabilities_placeholders[2].write(f"Probability for severity level 2 (serious/fatal injuries): ")

    # Horizontal line to separate sections
    st.markdown("<hr>", unsafe_allow_html=True)

    # Initialize a dictionary to store selected options
    selected_options_dict = {}

    num_rows = 3

    col1, col2, col3 = st.columns(3)  # Create three columns
    for idx, (feature, options) in enumerate(selection_dictionary.items()):
        if idx%num_rows == 0:
            selected_option = col1.selectbox(f"{feature}", list(options))
        elif idx%num_rows == 1:
            selected_option = col2.selectbox(f"{feature}", list(options))
        else:
            selected_option = col3.selectbox(f"{feature}", list(options))

        # Store the selected option in the dictionary
        selected_options_dict[feature] = selection_dictionary[feature][selected_option]

    # Create a DataFrame from selected options
    selected_options_df = pd.DataFrame(selected_options_dict, columns=selection_dictionary.keys(), index=[0])

    # Make predictions and get probabilities
    predictions = trained.predict(selected_options_df)
    probabilities = trained.predict_proba(selected_options_df)

    # Display predictions and probabilities
    predictions_placeholder.write(f"Prediction: severity level {predictions[0]}.")

    # Display probabilities for all classes
    probabilities_placeholders[0].write(f"Probability for severity level 0 (material damage only): {100 * probabilities[0,0]:.2f}%")
    probabilities_placeholders[1].write(f"Probability for severity level 1 (minor injuries): {100 * probabilities[0,1]:.2f}%")
    probabilities_placeholders[2].write(f"Probability for severity level 2 (serious/fatal injuries): {100 * probabilities[0,2]:.2f}%")


with tab2:
    st.write("...")

    
        


    

    
