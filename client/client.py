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

slide = None
prediction_types = None
registered_models = pd.DataFrame(columns=['source', 'version', 'stage', 'name'])
    
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


def is_canary_on(ptype):
    svc = "mlflow-" + ptype + "-infer"
    cmd = "kubectl describe svc " + svc + " | findstr Endpoints"
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, encoding='utf-8')
    output = ps.communicate()[0]
    if len(output.split(',')) > 1:
        return True
    return False


def change_canary_status(status, ptype):
    dep_yaml = "..\\k8s\\mlflow_" + ptype + "_v2_inference_dep.yml"
    if status == "Off":
        subprocess.run(["kubectl", "delete", "-f", dep_yaml], 
                    stdout=subprocess.PIPE, encoding='utf-8')
    else:
        out = subprocess.run(["kubectl", "apply", "-f", dep_yaml], 
                             stdout=subprocess.PIPE, encoding='utf-8')


def init_prediction_types():
    global prediction_types
    prediction_types = [
        { 'type': 'Regression', 'registered_as': 'regres'},
        { 'type': 'Classification', 'registered_as': 'clf'}
    ]

    prediction_types = pd.DataFrame(prediction_types)
    prediction_types.index += 1


def get_pred_str(selection):
    return prediction_types.loc[selection]['type']


def get_model_str(selection):
    return registered_models.loc[selection]['source'] + " : ver. " + str(registered_models.loc[selection]['version'])


def build_sidebar():
    ptype = st.sidebar.selectbox("Prediction Type:", prediction_types.reset_index(), format_func=get_pred_str)
    return prediction_types.loc[ptype]['registered_as']


def init_registered_models(basename):
    global registered_models
    registered_models = registered_models[0:0]
    names = []

    for rm in client.list_registered_models():
        reg_name = dict(dict(rm)['latest_versions'][0])['name']
        if basename in reg_name:
            names.append(reg_name)

    for name in names:        
        for mv in client.search_model_versions(f"name='{name}'"):
            mv = dict(mv)
            data = {'source': mv['source'].split('/')[-1], 
                    'version': mv['version'], 
                    'stage': mv['current_stage'],
                    'name': name
                }
            registered_models = registered_models.append(data, ignore_index = True)
            
        registered_models.index += 1


def get_deployment_yaml(name):
    prefix = '..\\k8s\\mlflow_'
    postfix = '_inference_dep.yml'
    parts = name.split('-')
    if len(parts) == 1:
        dep_name = prefix + parts[0] + postfix
    else:
        dep_name = prefix + parts[0] + "_" + parts[1][:2] + postfix
    return dep_name


def activate_model(model):
    name = registered_models.loc[model]['name']
    cond = '(stage == "Staging") and (name == @name)'
    current_version = registered_models.query(cond)['version'].iloc[0]
    new_version = registered_models.loc[model]['version']
    client.transition_model_version_stage(
                name = name,
                version = int(current_version),
                stage="Archived"
        )

    client.transition_model_version_stage(
                name = name,
                version = int(new_version),
                stage="Staging"
        )
    
    dep_yml = get_deployment_yaml(name)
    subprocess.run(["kubectl", "delete", "-f", dep_yml], 
                    stdout=subprocess.PIPE, encoding='utf-8')
    subprocess.run(["kubectl", "apply", "-f", dep_yml], 
                    stdout=subprocess.PIPE, encoding='utf-8')
    subprocess.run(["timeout", "50"], 
                    stdout=subprocess.PIPE, encoding='utf-8')


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
        activate_model(model)


def build_display():
    ptype = build_sidebar()
    init_registered_models(ptype)
    st.markdown("## Prediction Testing Client")
    build_available_models_display(ptype)
    return ptype


def build_canary_display(ptype):
    global slide
    text, slide, _ = st.beta_columns([1, 2, 1])
    with text:
        st.markdown(" ")
        st.markdown(" ")
        text.markdown("**Canary:**")
    
    status = "On" if is_canary_on(ptype) else "Off"
    new_status = slide.select_slider("", ["On", "Off"], value = status)
    print(status, new_status)
    if new_status != status:
        change_canary_status(new_status, ptype)


def build_active_model_display(ptype):
    active_models = ''
    cond = 'stage == "Staging"'
    if len(registered_models) > 0:
        for _, row in registered_models.query(cond).iterrows():
            version = row['version']
            source = row['source']
            active_models += source + " : ver. " + str(version) + ", "
        active_models = active_models[:-2]
        
        text, model, bt = st.beta_columns([1.8, 4, 2.6])
        with text:
            st.markdown("  ")
            st.markdown("  ")
            text.markdown("**Active Models**:")
        with model:
            st.markdown("  ")
            model.markdown(active_models)
        with bt:
            st.write("  ")
            predict = bt.button("Predict")
        build_canary_display(ptype)
        return predict
    else:
        msg = '<font color="red">Not implemented yet! Come back later.</font>'
        _, text = st.beta_columns([1.1, 4])
        text.write(msg, unsafe_allow_html=True)
        return False


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
    predict = build_active_model_display(ptype)
    if predict:
        text, _, rmse, _ = st.beta_columns([1, 1, 6, 2])
        if ptype == 'regres':
            text.markdown('**RMSE :** ')   
            rmse.write(do_regression(file_path))
        elif ptype == 'clf':
            text.markdown(" ")   
            rmse.write(do_classification(file_path), unsafe_allow_html=True)
        else:
            text.markdown(" ")   
            rmse.write(" ")