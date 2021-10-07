import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori

def app():
    st.title("Reglas de Asociacion")
    

    st.title('IntelliShow')

    DatosPeliculas = pd.read_csv('movies.csv', header=None)
    st.write(DatosPeliculas.head(10))

    Transacciones = DatosPeliculas.values.reshape(-1).tolist()
    #Transacciones

    Lista = pd.DataFrame(Transacciones)
    Lista['Frecuencia'] = 1  # valor que después se reemplazará, es nada más para agregar la columna. 

    #Agrupamos los elementos
    #Conteo
    # By=[0] empieza en cero.
    # as_index para que no aparezca el nombre 
    Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True)
    Lista['Porcentaje'] = (Lista['Frecuencia']/Lista['Frecuencia'].sum())
    Lista = Lista.rename(columns={0:'Item'})

    Lista #vemos que el máximo de apariciones en porc

    fig = plt.figure(figsize=(16,20), dpi = 300)
    plt.ylabel('Item')
    plt.xlabel('Frecuencia')
    plt.barh(Lista['Item'], width=Lista['Frecuencia'], color='#FF2442')
    st.pyplot(fig)

