from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



def app():
    colorFondo = '#FEFBF3'

    st.title("Clasificación R. Logística")

    archivo = st.file_uploader("Ingrese el archivo deseado")
    if archivo != None:
        st.write("Usted seleccionó el archivo " + "*"+archivo.name+"*" + "   de tipo   " + "*"+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        datos_opcional = st.expander("Datos Raw")
        datos_opcional.write(DataFrameArchivo)

        #datos_opcional2 = st.expander("Pairplot")
        #datos_opcional2.pyplot(sns.pairplot(DataFrameArchivo))

        CorrDataFrame = DataFrameArchivo.corr(method='pearson')
        MatrizCorr = np.triu(CorrDataFrame)
        datos_opcional2 = st.expander("Matriz de correlaciones")
        datos_opcional2.write(MatrizCorr)
        fig, ax = plt.subplots()
        #plt.figure(figsize=(14,7))
        fig.patch.set_facecolor(colorFondo)
        sns.heatmap(CorrDataFrame, cmap='RdBu_r', annot=True, mask=MatrizCorr)
        st.pyplot(fig)

        st.write("#")
        #SELECCION DE VARIABLES
        #st.caption("Seleccione las variables con las cuales se aplicará el algoritmo (es conveniente eliminar las variables que tienen gran dependencia entre ellas)")
        opciones = []
        for columna in DataFrameArchivo.columns:
            opciones.append(columna)
        opcionesVariables = st.multiselect("Seleccione las variables con las cuales se aplicará el algoritmo (es conveniente eliminar las variables que tienen gran dependencia entre ellas)", opciones, default=opciones)
        MatrizSeleccion = np.array(DataFrameArchivo[opcionesVariables])

