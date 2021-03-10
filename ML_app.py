import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow.keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
import pickle
import streamlit as st 
from eda_app import run_eda_app
import joblib 

def run_ML_app():
    st.subheader('Maching Learnig')

    model = tensorflow.keras.models.load_model('data/car_ai.h5')
    
    new_data = np.array([0,38,90000,2000,500000])

    new_data = new_data.reshape(1,-1)
    
    sc_X = joblib.load('data/sc_X.pkl')
    
    new_data = sc_X.transform(new_data)

    y_pred=model.predict(new_data)

    #st.write(y_pred[0][0])

    sc_y = joblib.load('data/sc_y.pkl')

    orginal = sc_y.inverse_transform(y_pred)

    st.write(orginal[0,0])













