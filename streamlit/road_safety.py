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

import dictionaries

from dictionaries import selection_dictionary, shorthand, selection_dictionary_shorthand, inverse_shorthand, selection_dictionary_fr, shorthand_fr, selection_dictionary_shorthand_fr, inverse_shorthand_fr, FR_EN, explain

from io import StringIO
# Redirect stdout to a StringIO object
stdout_orig = sys.stdout
sys.stdout = StringIO()

model = joblib.load(Path("./mtl_3sev_xgb.joblib"))

if 'selected_language' not in st.session_state:
    st.session_state.selected_language = "English"  
    
def english_content():
    
    st.title("Road safety")

    tab1, tab2, tab3  = st.tabs(["Predictions", "Dictionary", "Citation"])

    with tab1:

        text_to_display = "Many feature combinations are incompatible, e.g. the speed limit in a school zone cannot exceed 50. In such cases, our model will still offer a prediction, which may or may not be useful for planning purposes, or offer insight on hypothetical questions."
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
        selected_key = st.selectbox("", options, index=default_index, key = "dict_en")

        st.divider()

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
            
    with tab3:
        
        st.write("QUEBEC AUTOMOBILE INSURANCE SOCIETY (SAAQ). Accident reports, [Dataset], in Data Quebec, 2017, updated December 18, 2023. [https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident](https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident) (accessed March 13, 2024).")
                 
        st.write("_Data from accident reports completed by police officers, including the time, severity of the accident as well as the type of vehicles involved._")
        
        st.divider()
        ''' [![Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/tmfreiberg/road-safety.git) ''' 
        st.markdown("<br>",unsafe_allow_html=True)
             
def french_content():      
             
    st.title("Sécurité routière")
    
    tab4, tab5, tab6 = st.tabs(["Prédictions", "Dictionnaire", "Citation"])
        
    with tab4:

        text_to_display_fr = "De nombreuses combinaisons de fonctionnalités sont incompatibles, par exemple, la limite de vitesse dans une zone scolaire ne peut pas dépasser 50km/hre. Dans de tels cas, notre modèle offrira toujours une prédiction, qui peut ou non être utile à des fins de planification, ou offrira un aperçu de questions hypothétiques."
        s = f"<p style='font-size:15px;'>{text_to_display_fr}</p>"
        st.markdown(s, unsafe_allow_html=True)  

        st.divider()

        # Make predictions and get probabilities
        predictions_placeholder_fr = st.empty()
        probabilities_placeholders_fr = [st.empty() for _ in range(3)]  # One placeholder for each class


        # Display predictions and probabilities at the top of the page
        predictions_placeholder_fr.text("Niveau de gravité prédit : ")
        probabilities_placeholders_fr[0].text(f"Probabilité du niveau de gravité 0 : ")
        probabilities_placeholders_fr[1].text(f"Probabilité du niveau de gravité 1 : ")
        probabilities_placeholders_fr[2].text(f"Probabilité du niveau de gravité 2 : ")  

        # Initialize a dictionary to store selected options
        unencoded_selection_fr = { }
        selected_options_dict_fr = {}

        num_rows_fr = 3

        col1_fr, col2_fr, col3_fr = st.columns(3)  # Create three columns
        for idx, (feature_fr, options_fr) in enumerate(selection_dictionary_shorthand_fr.items()):
            if idx%num_rows_fr == 0:
                selected_option_fr = col1_fr.selectbox(f"{feature_fr}", list(options_fr.values()), key=f"col1_fr_{idx}")
            elif idx%num_rows_fr == 1:
                selected_option_fr = col2_fr.selectbox(f"{feature_fr}", list(options_fr.values()), key=f"col2_fr_{idx}")
            else:
                selected_option_fr = col3_fr.selectbox(f"{feature_fr}", list(options_fr.values()), key=f"col3_fr_{idx}")

            # Store the selected option in the dictionary
            unencoded_selection_fr[feature_fr] = inverse_shorthand_fr[feature_fr][selected_option_fr]
            selected_options_dict_fr[feature_fr] = selection_dictionary_fr[feature_fr][unencoded_selection_fr[feature_fr]]
            selected_options_dict = { FR_EN[feature_fr] : selected_options_dict_fr[feature_fr] for feature_fr in selected_options_dict_fr.keys()}
        # Create a DataFrame from selected options
        selected_options_df = pd.DataFrame(selected_options_dict, columns=selection_dictionary_shorthand.keys(), index=[0])

        # Make predictions and get probabilities
        predictions_fr = model.predict(selected_options_df)
        probabilities_fr = model.predict_proba(selected_options_df)

        # Display predictions and probabilities
        predictions_placeholder_fr.text(f"Niveau de gravité prédit :  {predictions_fr[0]}")

        # Display probabilities for all classes
        probabilities_placeholders_fr[0].text(f"Probabilité du niveau de gravité 0 : {100 * probabilities_fr[0,0]:.2f}%")
        probabilities_placeholders_fr[1].text(f"Probabilité du niveau de gravité 1 : {100 * probabilities_fr[0,1]:.2f}%")
        probabilities_placeholders_fr[2].text(f"Probabilité du niveau de gravité 2 : {100 * probabilities_fr[0,2]:.2f}%")

    with tab5:

        # Dropdown menu for selecting keys
        options_fr = list(FR_EN.keys())
        default_index_fr = options_fr.index("GRAVITE") if "GRAVITE" in options_fr else 0
        selected_key_fr = st.selectbox("", options_fr, index=default_index_fr, key="dict_fr")

        st.divider()

        # Placeholder for dynamic content
        explanation_placeholder_fr = st.empty()

        # Call the explain function and capture the output
        sys.stdout = StringIO()  # Redirect stdout to a StringIO object
        dictionaries.explain("FR", terms=[selected_key_fr])  # Call explain function
        generated_text_fr = sys.stdout.getvalue()  # Get the generated text

        # Restore stdout to its original value
        sys.stdout = stdout_orig

        # Display the generated text in Streamlit
        st.text(generated_text_fr)    

        if selected_key_fr == "GRAVITE":
            st.text("*NB : pour nos besoins :")
            st.text("Classe de gravité 0 : Dommages matériels inférieurs au seuil de rapportage/Dommages matériels seulement.")
            st.text("Classe de gravité 1 : Léger.")
            st.text("Classe de gravité 2 : Mortel ou grave.")
            
    with tab6:
        
        st.write("SOCIÉTÉ DE L'ASSURANCE AUTOMOBILE DU QUÉBEC (SAAQ). Rapports d'accident, [Jeu de données], dans Données Québec, 2017, mis à jour le 18 decembre 2023. [https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident](https://www.donneesquebec.ca/recherche/dataset/rapports-d-accident).")
                 
        st.write("_Données issues des rapports d’accident remplis par les policiers, incluant notamment le moment, la gravité de l’accident de même que le type des véhicules impliqués._")
        
        st.divider()       
        
        ''' [![Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/tmfreiberg/road-safety.git) ''' 
        st.markdown("<br>",unsafe_allow_html=True)

if st.button("English/Français"):
    if st.session_state.selected_language == "English":
        st.session_state.selected_language = "French"
    else:
        st.session_state.selected_language = "English"

if st.session_state.selected_language == "English":
    english_content()
else:
    french_content()    

    
        


    

    
