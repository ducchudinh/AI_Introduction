import streamlit as st
import pickle
import data_prepare as dp
import numpy as np

st.title('Airlines Customer Satisfaction')

st.header('INPUT')

attribute_values = []

col1, col2, col3, col4 = st.columns(4)
with col1:
    option = st.selectbox('Gender', ('Male', 'Female'))
    attribute_values.append(option)
with col2:
    option = st.selectbox('Customer Type', ('Loyal Customer', 'disloyal Customer'))
    attribute_values.append(option)
with col3:
    age = st.slider('Age', 0, 130, 30)
    attribute_values.append(age)
with col4:
    option = st.selectbox('Type of Travel', ('Personal Travel', 'Business travel'))
    attribute_values.append(option)

col5, col6, col7, col8 = st.columns(4)
with col5:
    option = st.selectbox('Class', ('Eco', 'Business', 'Eco Plus'))
    attribute_values.append(option)
with col6:
    number = st.number_input('Flight Distance', step=500)
    attribute_values.append(number)
with col7:
    option = st.selectbox('Seat comfort', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col8:
    option = st.selectbox('De/Ar time convenient', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)

col9, col10, col11, col12 = st.columns(4)
with col9:
    option = st.selectbox('Food and drink', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col10:
    option = st.selectbox('Gate location', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col11:
    option = st.selectbox('Inflight wifi service', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col12:
    option = st.selectbox('Inflight entertainment', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)

col13, col14, col15, col16 = st.columns(4)
with col13:
    option = st.selectbox('Online support', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col14:
    option = st.selectbox('Ease of Online booking', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col15:
    option = st.selectbox('On-board service', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col16:
    option = st.selectbox('Leg room service', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)

col17, col18, col19, col20 = st.columns(4)
with col17:
    option = st.selectbox('Baggage handling', ('1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col18:
    option = st.selectbox('Checkin service', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col19:
    option = st.selectbox('Cleanliness', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)
with col20:
    option = st.selectbox('Online boarding', ('0', '1', '2', '3', '4', '5'))
    attribute_values.append(option)

col21, col22, col23, col24 = st.columns(4)
with col21:
    number = st.number_input('Departure Delay in Minutes', step=60)
    attribute_values.append(number)
with col22:
    number = st.number_input('Arrival Delay in Minutes', step=60)
    attribute_values.append(number)

attribute_values = dp.prepare_infer_data(attribute_values)


st.header('OUTPUT')

LinearSVC_model = pickle.load(open('LinearSVC_trained_model', 'rb'))
linearsvc_result = LinearSVC_model.predict(np.array(attribute_values).reshape(1,-1))
CatBoost_model = pickle.load(open('CatBoost_trained_model', 'rb'))
catboost_result = CatBoost_model.predict(np.array(attribute_values).reshape(1,-1))

col1, col2 = st.columns(2)
with col1:
    st.subheader('LinearSVC')
    if linearsvc_result == 1:
        st.write('Satisfied')
        st.image('img/satisfied.jpg')
    else:
        st.write('Dissatisfied')
        st.image('img/disatisfied.jpg')
with col2:
    st.subheader('CatBoost')
    if catboost_result == 1:
        st.write('Satisfied')
        st.image('img/satisfied.jpg')
    else:
        st.write('Dissatisfied')
        st.image('img/disatisfied.jpg')

st.caption('Chu Đình Đức - Đỗ Minh Hiệp - Đinh Ngọc Huân')
st.caption('Khoa học máy tính - HUST')
st.caption('Nhập môn Trí tuệ nhân tạo')