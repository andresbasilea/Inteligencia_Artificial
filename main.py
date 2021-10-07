import streamlit as st
import MetricasDistancia
import ReglasAsociacion

TABS = {"MetricasDistancia": MetricasDistancia, "ReglasAsociacion": ReglasAsociacion}

st.sidebar.title('Algoritmos')
selection = st.sidebar.radio("Ir a", list(TABS.keys()))
page = TABS[selection]
page.app()





