import os
import pandas as pd
import numpy as np
import json
from enum import Enum
import itertools
import requests
import streamlit as st
from sklearn.metrics import mean_squared_error

URL = os.environ.get('PREDICTION_URL', "http://127.0.0.1:31236/invocations")
HEADERS = ""

_PREDICTIONTYPE = {
    1: ['Linear Regression', 'LinReg'],
}


PredictionType = Enum(
    value='PredictionTypes',
    names=itertools.chain.from_iterable(
            itertools.product(v, [k]) for k, v in _PREDICTIONTYPE.items()
        )
)

@st.cache
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def do_prediction(file_path):
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
    

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    


def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)


def get_pred_str(selection):
    return PredictionType(selection).name


def build_display():
    st.markdown("## Prediction Testing Client")
    ptype = tuple([ptype.value for ptype in PredictionType])
    
    text, cb, _, bt = st.beta_columns([3, 6, 1, 4])
    with text:
        st.markdown("  ")
        st.markdown("  ")
        st.markdown("**Prediction Type** : ")
    with cb:
        ptype = st.selectbox("", ptype, format_func=get_pred_str, key=1)
    with bt:
        st.markdown("  ")
        predict = st.button("Predict",)
    if predict:
        return PredictionType(ptype)


if __name__ == "__main__":
    local_css("style.css")
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

    file_path = "../dataset/housing.csv"
    predict = build_display()
    text, rmse, _ = st.beta_columns([1, 1, 5])
    with text:
        st.markdown('**RMSE :** ')

    if predict:
        rmse.write(do_prediction(file_path))
    else:
        rmse.write(" ")