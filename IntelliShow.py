from PIL.Image import init
import streamlit as st
import MetricasDistancia
import ReglasAsociacion
import Bienvenida
import Clustering
import ClasificacionLogistica
import ArbolClasificacionPronostico
import re

TABS = {"Inicio": Bienvenida, "Métricas Distancia": MetricasDistancia, "Reglas de Asociación (A priori)": ReglasAsociacion, "Clústering": Clustering, "Clasificación R. Logística": ClasificacionLogistica, "Árbol de Decisión (pronóstico y clasificación)": ArbolClasificacionPronostico}



st.set_page_config(page_icon=":brain:", page_title="IntelliShow", initial_sidebar_state="collapsed",)

st.sidebar.title('Algoritmos')
selection = st.sidebar.radio("Ir a", list(TABS.keys()))
page = TABS[selection]
page.app()




