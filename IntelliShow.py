from PIL.Image import init
import streamlit as st
import MetricasDistancia
import ReglasAsociacion
import Bienvenida
import ClusteringJerarquico
import re

TABS = {"Inicio": Bienvenida, "Métricas Distancia": MetricasDistancia, "Reglas de Asociación (A priori)": ReglasAsociacion, "Clústering Jerárquico": ClusteringJerarquico}



st.set_page_config(page_icon=":brain:", page_title="IntelliShow", initial_sidebar_state="collapsed",)

st.sidebar.title('Algoritmos')
selection = st.sidebar.radio("Ir a", list(TABS.keys()))
page = TABS[selection]
page.app()

# colores = []
# with open(".streamlit/config.toml","r") as config:
#         lines = config.readlines()
#         st.write(lines)
#         j = 0
#         for i in lines:
#             color = re.search("#[a-zA-Z0-9]{6}", i)
#             if color != None:
#                 colores[j] = color
#             j+=1





