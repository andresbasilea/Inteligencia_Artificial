import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori



def app():
       
    colorFondo = '#FEFBF3'
    colorSecundarioFondo = "#F8F0DF"
    colorPrimario = '#79B4B7'
    
    st.title("Reglas de Asociación")

    archivo = st.file_uploader("Ingrese el archivo deseado")
    if archivo != None:
        st.write("Usted seleccionó el archivo " + "*"+archivo.name+"*" + "   de tipo   " + "*"+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        datos_opcional = st.expander("Datos Raw")
        datos_opcional.write(DataFrameArchivo)



        Transacciones = DataFrameArchivo.values.reshape(-1).tolist()
        

        Lista = pd.DataFrame(Transacciones)
        Lista['Frecuencia'] = 1  # valor que después se reemplazará, es nada más para agregar la columna. 
        Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
        Lista['Porcentaje'] = (Lista['Frecuencia']/Lista['Frecuencia'].sum())
        Lista = Lista.rename(columns={0:'Item'})
        lista_opcional = st.expander("Lista de frecuencia:", False)
        lista_opcional.write(Lista)


        
        font = {'family' : 'normal',
                'weight' : 'normal',
                'size'   : 7}

        plt.rc('font', **font)
        fig = plt.figure(figsize=(16,16), dpi = 300)
        plt.ylabel('Item')
        plt.xlabel('Frecuencia')
        fig.patch.set_facecolor(colorFondo)
        ax = plt.axes()
        ax.set(facecolor = colorSecundarioFondo)
        plt.barh(Lista['Item'], width=Lista['Frecuencia'], color=colorPrimario)
        st.pyplot(fig)

        Lista = DataFrameArchivo.stack().groupby(level=0).apply(list).tolist()
        ReglasC1 = apriori(Lista, 
                   min_support=0.0045, 
                   min_confidence=0.2, 
                   min_lift=3)
        ResultadoC1 = list(ReglasC1)
        st.write(len(ResultadoC1))
        df_resultado_c1 = pd.DataFrame(ResultadoC1)
        
