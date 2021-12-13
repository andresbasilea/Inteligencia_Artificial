import streamlit as st
from streamlit.type_util import data_frame_to_bytes
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import distance

def app():
    
    st.title("Métricas de Distancia")


   

    archivo = st.file_uploader("Ingrese el archivo deseado")
    if archivo != None:
        st.write("Usted seleccionó el archivo " + "*"+archivo.name+"*" + "   de tipo   " + "*"+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        datos_opcional = st.expander("Datos Raw")
        datos_opcional.write(DataFrameArchivo)
        
        
        seleccionMetrica = st.radio("Seleccionar la métrica de distancia a utilizar: ", ["Euclidiana", "Chebyshev", "Manhattan (City Block)", "Minkowski"])
        opcion = 0
        if seleccionMetrica == "Euclidiana":
            opcion = 1
            DstEuclideana = cdist(DataFrameArchivo, DataFrameArchivo, metric='euclidean')
            MEuclideana = pd.DataFrame(DstEuclideana)
            st.write(MEuclideana)
        if seleccionMetrica =="Chebyshev":
            opcion = 2
            DstChebyshev = cdist(DataFrameArchivo, DataFrameArchivo, metric='chebyshev')
            MChebyshev = pd.DataFrame(DstChebyshev)
            st.write(MChebyshev)
        if seleccionMetrica =="Manhattan (City Block)":
            opcion = 3
            DstManhattan = cdist(DataFrameArchivo,DataFrameArchivo, metric='cityblock')
            MManhattan = pd.DataFrame(DstManhattan)
            st.write(MManhattan)
        if seleccionMetrica == "Minkowski":
            opcion = 4
            paramLambda = st.text_input("λ:", 1.5)
            DstMinkowski = cdist(DataFrameArchivo, DataFrameArchivo, metric='minkowski', p=float(paramLambda))
            MMinkowski = pd.DataFrame(DstMinkowski)
            st.write(MMinkowski)

        
        
        if opcion == 1:
            st.download_button('¡A hoja de cálculo!', file_name='matrizDistanciaEuclideana' + '.csv', data =  MEuclideana.to_csv().encode('utf-8'), mime = 'text/csv')
        if opcion == 2:
            st.download_button('¡A hoja de cálculo!', file_name='matrizDistanciaChebyshev' + '.csv', data =  MChebyshev.to_csv().encode('utf-8'), mime = 'text/csv')
        if opcion == 3:
            st.download_button('¡A hoja de cálculo!', file_name='matrizDistanciaManhattan' + '.csv', data =  MManhattan.to_csv().encode('utf-8'), mime = 'text/csv')
        if opcion == 4:
            st.download_button('¡A hoja de cálculo!', file_name='matrizDistanciaMinkowski' + '.csv', data =  MMinkowski.to_csv().encode('utf-8'), mime = 'text/csv')