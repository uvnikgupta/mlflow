import os
import pandas as pd
import numpy as np
import json
import requests
import subprocess
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error

URL = os.environ.get('PREDICTION_URL', "http://127.0.0.1:31236/invocations")
HEADERS = ""

prediction_types = None
registered_models = pd.DataFrame(columns=['source', 'version', 'stage'])
    
mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', 
                                    'postgresql+psycopg2://postgres:password@localhost:32345'))
client = MlflowClient()

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def activate_model(model, name):
    cond = 'stage == "Staging"'
    active_version = registered_models.query(cond)['version'].iloc[0]
    client.transition_model_version_stage(
                name = name,
                version = int(active_version),
                stage="Archived"
        )

    client.transition_model_version_stage(
                name = name,
                version = int(model),
                stage="Staging"
        )
    
    subprocess.run(["kubectl", "delete", "-f", "..\\k8s\\mlflow_linreg_inference_dep.yml"], 
                    stdout=subprocess.PIPE, encoding='utf-8')
    subprocess.run(["kubectl", "apply", "-f", "..\\k8s\\mlflow_linreg_inference_dep.yml"], 
                    stdout=subprocess.PIPE, encoding='utf-8')
    subprocess.run(["timeout", "50"], 
                    stdout=subprocess.PIPE, encoding='utf-8')


def init_prediction_types():
    global prediction_types
    prediction_types = [
        { 'type': 'Regression', 'registered_as': 'linreg'},
        { 'type': 'Classification', 'registered_as': 'clf'}
    ]

    prediction_types = pd.DataFrame(prediction_types)
    prediction_types.index += 1


def get_pred_str(selection):
    return prediction_types.loc[selection]['type']


def get_model_str(selection):
    return registered_models.loc[selection]['source'] + " : ver. " + str(registered_models.loc[selection]['version'])


def init_registered_models(name):
    global registered_models
    registered_models = registered_models[0:0]
    for mv in client.search_model_versions(f"name='{name}'"):
        mv = dict(mv)
        data = {'source': mv['source'].split('/')[-1], 
                'version': mv['version'], 
                'stage': mv['current_stage']
            }
        registered_models = registered_models.append(data, ignore_index = True)
        
    registered_models.index += 1


def build_sidebar():
    ptype = st.sidebar.selectbox("Prediction Type:", prediction_types.reset_index(), format_func=get_pred_str)
    return prediction_types.loc[ptype]['registered_as']


def build_available_models_display(ptype):
    cond = 'stage != "Staging"'
    text, cb, bt = st.beta_columns([3, 6, 4])
    with text:
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("**Available Models** : ")
    with cb:
        model = st.selectbox("", registered_models.query(cond).reset_index(), format_func=get_model_str)
    with bt:
        # st.markdown("  ")
        st.markdown("  ")
        activate = st.button("Activate", key="1")
    if activate:
        activate_model(model, ptype)


def build_active_model_display():
    cond = 'stage == "Staging"'
    try:
        version = registered_models.query(cond)['version'].iloc[0]
        source = registered_models.query(cond)['source'].iloc[0]
        text, model, bt = st.beta_columns([1.8, 4, 2.6])
        with text:
            st.markdown("  ")
            st.markdown("  ")
            text.markdown("**Active Model**:")
        with model:
            st.markdown("  ")
            st.markdown("  ")
            model.markdown(source + " : ver. " + str(version))
        with bt:
            st.write("  ")
            predict = bt.button("Predict")
        return predict
    except:
        msg = '<font color="red">Not implemented yet! Come back later.</font>'
        _, text = st.beta_columns([1.1, 4])
        text.write(msg, unsafe_allow_html=True)
        return False


def build_display():
    ptype = build_sidebar()
    init_registered_models(ptype)
    st.markdown("## Prediction Testing Client")
    build_available_models_display(ptype)
    return ptype


@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def do_regression(file_path):
    housing = load_data(file_path)
    targetCol = "median_house_value"
    catCols = ["ocean_proximity"]
    numCols = ["housing_median_age", "total_rooms", "total_bedrooms", "population", 
               "households", "median_income"]
    x = housing[numCols + catCols]
    y = housing[targetCol]
    batch_size = 80
    input_json = x[:batch_size].to_json(orient="split")
    input_json = json.loads(input_json)

    r = requests.post(URL, json = input_json, headers=HEADERS)
    output = r.text
    predictions = np.array(json.loads(output)).reshape(-1,1)
    rmse = np.sqrt(mean_squared_error(y[:batch_size], predictions))
    return np.round(rmse, 2)
    

def do_classification(file_path):
    return '<font color="red">Not implemented yet! Come back later.</font>'


if __name__ == "__main__":
    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
    init_prediction_types()

    file_path = "dataset/housing.csv"
    ptype = build_display()
    predict = build_active_model_display()
    if predict:
        text, _, rmse, _ = st.beta_columns([1, 1, 6, 2])
        if ptype == 'linreg':
            text.markdown('**RMSE :** ')   
            rmse.write(do_regression(file_path))
        elif ptype == 'clf':
            text.markdown(" ")   
            rmse.write(do_classification(file_path), unsafe_allow_html=True)
        else:
            text.markdown(" ")   
            rmse.write(" ")