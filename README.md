# Road Safety

## Overview

This project focuses on analyzing data related to traffic accidents based on police reports filed in Quebec, with a particular emphasis on Montreal, spanning from 2011 to 2022. The goal is to develop a robust understanding of accident patterns and severity levels.

## Classification Model

We employ the XGBoostClassifier to create a ternary classification model. This model categorizes accidents into three classes:

- **Material Damage Only**
- **At Least One Minor Injury**
- **At Least One Serious Injury or Fatality**

## Deployment

The trained classification model is deployed using a Streamlit app. This app serves as a valuable tool for various stakeholders, including:

- **Municipal Authorities and Town Planners**: Informing urban planning/infrastructure decisions, as well as traffic control logistics.
- **Vision Zero Campaigners**: Supporting initiatives aimed at reducing traffic accidents and fatalities.

## How to Use

To explore the insights derived from our analysis and classification model, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies listed in `requirements.txt`.
3. Run the Streamlit app using the command `streamlit run road_safety.py`.
4. Interact with the app to view accident classifications and related insights.

## Contributions and Feedback

We welcome contributions and feedback from the community to enhance the accuracy and applicability of our analysis and model. Please feel free to submit issues, pull requests, or reach out with suggestions.

 