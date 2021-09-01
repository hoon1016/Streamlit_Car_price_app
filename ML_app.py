import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint,CSVLogger
import pickle
import streamlit as st 
from eda_app import run_eda_app
import joblib 

def run_ML_app():
    st.subheader('Maching Learnig')

    

    #1.유저한테 입력을 받는다.
    #성별
    gendar = st.radio("성별을 선택하세요",['남자','여자'])
    if gendar == '남자':
        gendar = 1
    else:
        gendar = 0
    
    age = st.number_input("나이 입력",min_value=0,max_value=120)
    
    salary = st.number_input('연봉 입력',min_value=0)

    debt = st.number_input('빚 입력',min_value=0)

    worth = st.number_input('자산 입력',min_value=0)

    # 2. 예측한다.
    # 2- 1. 모델 불러오기
    model = tensorflow.keras.models.load_model('data/car_ai.h5')
    #2-2. 넘파이 어레이 만든다.

    new_data = np.array([gendar,age,salary,debt,worth])
    #2-3. 피쳐스케일링하자.
    new_data = new_data.reshape(1,-1)
    sc_X = joblib.load('data/sc_X.pkl')
    
    new_data = sc_X.transform(new_data)
    #2-4. 예측한다.
    y_pred=model.predict(new_data)
    # 예측 결과는, 스케일링 된 결과이므로, 다시 돌려야한다.
    #st.write(y_pred[0][0])
    sc_y = joblib.load('data/sc_y.pkl')
    orginal = sc_y.inverse_transform(y_pred)
    
    # 3. 결과를 화면에 보여준다.
    bth = st.button('결과 보기')
    if bth :
        st.write('예측 결과입니다. {:,.1f} 달러의 차를 살수 있습니다'.format(orginal[0,0],))
    













