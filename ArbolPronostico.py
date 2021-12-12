from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def app():
    colorFondo = '#FEFBF3'

    st.title("Árbol de Decisión (pronóstico)")

    archivo = st.file_uploader("Ingrese el archivo deseado")
    if archivo != None:
        st.write("Usted seleccionó el archivo " + "*"+archivo.name+"*" + "   de tipo   " + "*"+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        datos_opcional = st.expander("Datos Raw")
        datos_opcional.write(DataFrameArchivo)

        #datos_opcional2 = st.expander("Pairplot")
        #datos_opcional2.pyplot(sns.pairplot(DataFrameArchivo))

        