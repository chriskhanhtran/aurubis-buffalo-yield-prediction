import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import re
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def main_page():
    st.markdown("# Yield Prediction with Machine Learning")
    
    ############################
    ###        Set-up        ###
    ############################
    # Load model
    model = joblib.load("lgb.joblib")
    
    # Load data
    final_data = pd.read_csv("final_data.csv")
    
    # Load example
    with open("example.json", "r+") as f:
        example = json.load(f)
        
    operation_list = [feat[7:] for feat in example.keys() if re.match(r"^SDOPCD_", feat)]
    machine_list = [feat[7:] for feat in example.keys() if re.match(r"^SDNUMB_", feat)]
    
    ############################
    ###     Input Values     ###
    ############################
    # Operations
    operations = st.sidebar.text_input("Operations: ", "CR-AN-CR-SL-CN-SL-CN-SL")
    operations = pd.Series(operations.split("-")).value_counts()
    operations.index = ['SDOPCD_' + oper for oper in operations.index]
    for k, v in operations.to_dict().items():
        k = k.upper()
        if k[7:] not in operation_list:
            st.write(f"Operation {k[7:]} not in Operation List: {', '.join(operation_list)}")
            continue
        example[k] = v
        
    # Machines  
    machines = st.sidebar.text_input("Machine: ", "44-143-46-1-24-77-24-77")
    machines = pd.Series(machines.split("-")).value_counts()
    machines.index = ['SDNUMB_' + machine for machine in machines.index]
    for k, v in machines.to_dict().items():
        k = k.upper()
        if k[7:] not in machine_list:
            st.write(f"Machine {k[7:]} not in Machine List: {', '.join(machine_list)}")
            continue
        example[k] = v
        
    st.sidebar.markdown("\**Input values in the above format*")
    
    # Numeric features
    example['total_slits'] = st.sidebar.number_input("Total Slits: ", 0, 5, 1)
    example['start_weight'] = st.sidebar.number_input("Starting Weight: ", 0, None, 18869)
    example['INSGAG'] = st.sidebar.number_input("Starting Gauge: ", 0.0, None, 0.25)
    example['INSWID'] = st.sidebar.number_input("Starting Width: ", 0.0, None, 38.21)
    example['SDHRS'] = st.sidebar.number_input("Expected Hours: ", 0.0, None, 7.18)
    
    # Show an example
    if st.checkbox("Display an example"): 
        example_data = pd.read_csv("example_data.csv")
        st.dataframe(example_data.fillna(" "))
 
    ############################
    ###       Predict        ###
    ############################
    # Convert example to appropriate format and shape
    example = pd.Series(example).values.reshape(1, -1)
    
    # Make prediction
    pred = model.predict(example)[0]
    st.markdown(f"# Predicted Yield: {pred:.2f} %")
    
    # Plot Yield
    plt.figure(figsize=(7, 4))
    sns.distplot(final_data['yield'])
    plt.axvline(pred/100, color='r')
    plt.title("Histogram of Yield")
    st.pyplot()    
    
    ############################
    ###     Visualization    ###
    ############################
    st.markdown("# Feature Visualization")
    feat = st.selectbox("Feature: ", ["Weight", "Gauge", "Width", "Hours"])
    feat_dict = {"Weight": "start_weight",
                 "Gauge": "INSGAG",
                 "Width": "INSWID",
                 "Hours": "SDHRS"}
                 
    plt.figure(figsize=(7, 4))
    sns.distplot(final_data[feat_dict[feat]], kde=False)
    plt.title(f"Histogram of {feat}")
    st.pyplot()


if __name__ == "__main__":
    main_page()
