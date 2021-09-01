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
import os 

def run_eda_app():
    st.subheader('EDA 화면입니다.')
    car_df = pd.read_csv('data/Car_Purchasing_Data.csv',encoding='ISO-8859-1')
    
    radio_menu =['데이터프레임','통계치']
    selected_radio = st.radio('선택하세요',radio_menu)

    if selected_radio == '데이터프레임':
        st.dataframe(car_df)
    elif selected_radio == '통계치':
        st.dataframe(car_df.describe())
    
    columns = car_df.columns
    columns = list(columns)
    
    selected_columns = st.multiselect('컬럼을 선택하시오',columns)
    if len(selected_columns) != 0 :
        st.dataframe(car_df[selected_columns])
    else:
        st.write('선택한 컬럼이 없습니다')
    # 상관 계수를 화면에 보여주도록 만듭니다.
    # 멀티셀럭트에 컬럼명을 보여주고.
    #해당 컬럼들에 대한 상관계수를 보여주세요
    #단,컬럼들은 숫자 컬럼들만 멀티셀럭트에 나타나야합니다

    print(car_df.dtypes == object)

    corr_columns = car_df.columns[car_df.dtypes != object]
    corr_list = st.multiselect('상관 계수를 볼 컬럼을 선택하세요',corr_columns)
    
    if len(corr_list) != 0 :
        st.dataframe(car_df[corr_list].corr())
    #위에서 선택한 컬럼들을 이용해서, 시본의 페어플롯을
    #그린다
        st.pyplot(sns.pairplot(car_df[corr_list]))
    
    else:
        st.write('선택한 컬럼이 없습니다')
    
    #컬럼을 하나만 선택하면, 해당 컬럼의 min과 max에 
    #해당하는 사람의 데이터를 화면에 보여주는 기능
    menu = corr_columns
    choice = st.selectbox('Min & Max',menu)
    
    min_data = car_df[choice].min() == car_df[choice]
    st.write('최소값 데이터')
    st.dataframe(car_df.loc[min_data,])
    
    max_data = car_df[choice].max() == car_df[choice]
    st.write('최대값 데이터')
    st.dataframe(car_df.loc[max_data,])
    #고객이름을 검색할 수 있는 기능 개발    
    name = st.text_input('고객의 이름을 입력하세요')
    search_name=car_df.loc[car_df['Customer Name'].str.contains(name,case = False),]
    if len(search_name) != 0:
        st.dataframe(search_name)
    else:
        st.write('해당 고객에 정보가 없습니다')