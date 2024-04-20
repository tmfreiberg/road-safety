# ROAD ACCIDENT SEVERITY

# Save this as foo.py and run it with 'streamlit run foo.py [-- script args]'.
# Alternatively, run 'python -m streamlit run foo.py'.

import sys
from pathlib import Path

# Determine the root directory (where streamlit.py is located)
root_directory = Path(__file__).resolve().parent

# Append the root directory to sys.path if it's not already there
if str(root_directory) not in sys.path:
    sys.path.append(str(root_directory))
    
import streamlit as st  
import numpy as np 
import pandas as pd 
import joblib
# import random

import dictionaries

from dictionaries import selection_dictionary, FR_EN, shorthand, selection_dictionary_shorthand, inverse_shorthand, explain 
from io import StringIO
# Redirect stdout to a StringIO object
stdout_orig = sys.stdout
sys.stdout = StringIO()

model = joblib.load(Path("./mtl_3sev_xgb.joblib"))

st.markdown("<h1 style='text-align: center; text-transform: uppercase;'>Road accident severity</h1>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Predictions", "Dictionary"])

with tab1:
    
    text_to_display = "Many feature combinations are incompatible, e.g. the speed limit in a school zone cannot exceed 50. In such cases, our model will still offer a prediction, but it will be meaningless."
    s = f"<p style='font-size:15px;'>{text_to_display}</p>"
    st.markdown(s, unsafe_allow_html=True)  
    
    st.divider()

    # Make predictions and get probabilities
    predictions_placeholder = st.empty()
    probabilities_placeholders = [st.empty() for _ in range(3)]  # One placeholder for each class


    # Display predictions and probabilities at the top of the page
    predictions_placeholder.text("Predicted severity level: ")
    probabilities_placeholders[0].text(f"Probability of severity level 0: ")
    probabilities_placeholders[1].text(f"Probability of severity level 1: ")
    probabilities_placeholders[2].text(f"Probability of severity level 2: ")  

    # Initialize a dictionary to store selected options
    unencoded_selection = { }
    selected_options_dict = {}

    num_rows = 3

    col1, col2, col3 = st.columns(3)  # Create three columns
    for idx, (feature, options) in enumerate(selection_dictionary_shorthand.items()):
        if idx%num_rows == 0:
            selected_option = col1.selectbox(f"{feature}", list(options.values()))
        elif idx%num_rows == 1:
            selected_option = col2.selectbox(f"{feature}", list(options.values()))
        else:
            selected_option = col3.selectbox(f"{feature}", list(options.values()))

        # Store the selected option in the dictionary
        unencoded_selection[feature] = inverse_shorthand[feature][selected_option]
        selected_options_dict[feature] = selection_dictionary[feature][unencoded_selection[feature]]

    # Create a DataFrame from selected options
    selected_options_df = pd.DataFrame(selected_options_dict, columns=selection_dictionary_shorthand.keys(), index=[0])

    # Make predictions and get probabilities
    predictions = model.predict(selected_options_df)
    probabilities = model.predict_proba(selected_options_df)

    # Display predictions and probabilities
    predictions_placeholder.text(f"Predicted severity level: {predictions[0]}")

    # Display probabilities for all classes
    probabilities_placeholders[0].text(f"Probability of severity level 0: {100 * probabilities[0,0]:.2f}%")
    probabilities_placeholders[1].text(f"Probability of severity level 1: {100 * probabilities[0,1]:.2f}%")
    probabilities_placeholders[2].text(f"Probability of severity level 2: {100 * probabilities[0,2]:.2f}%")
  
with tab2:
    
    # Dropdown menu for selecting keys
    options = list(FR_EN.values())
    default_index = options.index("SEVERITY") if "SEVERITY" in options else 0
    selected_key = st.selectbox("", options, index=default_index)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Placeholder for dynamic content
    explanation_placeholder = st.empty()

    # Call the explain function and capture the output
    sys.stdout = StringIO()  # Redirect stdout to a StringIO object
    dictionaries.explain(terms=[selected_key])  # Call explain function
    generated_text = sys.stdout.getvalue()  # Get the generated text

    # Restore stdout to its original value
    sys.stdout = stdout_orig
    
    # Display the generated text in Streamlit
    st.text(generated_text)    
    
    if selected_key == "SEVERITY":
        st.text("*NB: for our purposes:")
        st.text("Severity class 0: Material damage only/Material damage below the reporting threshold.")
        st.text("Severity class 1: Minor.")
        st.text("Severity class 2: Fatal or serious.")
    
    

    
        


    

    
