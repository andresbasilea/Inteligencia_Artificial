import streamlit as st
import MetricasDistancia
import ReglasAsociacion
import Bienvenida

TABS = {"Bienvenida": Bienvenida, "MetricasDistancia": MetricasDistancia, "ReglasAsociacion": ReglasAsociacion}

st.sidebar.title('Algoritmos')
selection = st.sidebar.radio("Ir a", list(TABS.keys()))
page = TABS[selection]
page.app()





