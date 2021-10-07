import streamlit as st
from PIL import Image
 

 

def app():  
    st.title("IntelliShow: Solución en Inteligencia Artificial")
    imagen1 = Image.open('imagenes/IA1.jpg')
    st.image(imagen1, caption='Inteligencia Artificial')
