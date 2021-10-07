import streamlit as st
from PIL import Image
 

 

def app():  
    st.title("IntelliShow")
    st.header("Solución en Inteligencia Artificial")
    imagen1 = Image.open('imagenes/IA3.png')
    st.image(imagen1)
    imagen2 = Image.open('imagenes/IA2.jpg')
    st.image(imagen2)
    imagen3 = Image.open('imagenes/IA1.jpg')
    st.image(imagen3)

    st.subheader("La presente aplicación tiene por objetivo presentar algunos algoritmos de inteligencia artificial")
