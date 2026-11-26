import streamlit as st
import requests

st.title('Распознование')
file = st.file_uploader('Загрузи картинку', ['jpg', 'png', 'jpeg'])

if file:
    st.image(file, width=250)
    if st.button('Распознать'):
        response = requests.post('http://127.0.0.1:8000/predict', files={'file': (file.name, file.getvalue(), file.type)})
        st.success(response.json()['prediction'])