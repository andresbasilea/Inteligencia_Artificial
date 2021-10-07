import streamlit as st
from streamlit.type_util import data_frame_to_bytes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance

def app():
    
    st.title("Métricas de Distancia")
    st.sidebar.title('Parámetros')
    selection = st.sidebar.radio("Ir a", [1,2,3])
   

    archivo = st.file_uploader("Ingrese el archivo deseado")
    if archivo != None:
        st.write("Usted seleccionó el archivo " + "*"+archivo.name+"*" + "   de tipo   " + "*"+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        datos_opcional = st.expander("Datos Raw")
        datos_opcional.write(DataFrameArchivo)
        
        
        seleccionMetrica = st.radio("Seleccionar la métrica de distancia a utilizar: ", ["Euclidiana", "Chebyshev", "Manhattan (City Block)", "Minkowski"])
        if seleccionMetrica == "Euclidiana":
            DstEuclideana = cdist(DataFrameArchivo, DataFrameArchivo, metric='euclidean')
            MEuclideana = pd.DataFrame(DstEuclideana)
            st.write(MEuclideana)
        if seleccionMetrica =="Chebyshev":
            DstChebyshev = cdist(DataFrameArchivo, DataFrameArchivo, metric='chebyshev')
            MChebyshev = pd.DataFrame(DstChebyshev)
            st.write(MChebyshev)
        if seleccionMetrica =="Manhattan (City Block)":
            DstManhattan = cdist(DataFrameArchivo,DataFrameArchivo, metric='cityblock')
            MManhattan = pd.DataFrame(DstManhattan)
            st.write(MManhattan)
        if seleccionMetrica == "Minkowski":
            paramLambda = st.text_input("λ:", 1.5)
            DstMinkowski = cdist(DataFrameArchivo, DataFrameArchivo, metric='minkowski', p=float(paramLambda))
            MMinkowski = pd.DataFrame(DstMinkowski)
            st.write(MMinkowski)

