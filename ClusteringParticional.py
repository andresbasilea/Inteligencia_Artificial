from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit.elements.arrow import Data
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator


def app():
    colorFondo = '#FEFBF3'

    st.title("Clústering Particional")

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

        #APLICACIÓN DEL ALGORITMO

        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
        MEstandarizada = estandarizar.fit_transform(MatrizSeleccion)
        st.write(pd.DataFrame(MEstandarizada))


        #DEFINICIÓN DE K CLUSTERS PARA K MEANS
        SSE = []
        for i in range(2, 12):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(MEstandarizada)
            SSE.append(km.inertia_)

        #Se grafica SSE en función de k
        # METODO DEL CODO

        # fig,ax = plt.subplots()
        # fig.patch.set_facecolor(colorFondo)
        # plt.plot(range(2, 12), SSE, marker='o')
        # plt.xlabel('Cantidad de clusters')
        # plt.ylabel('SSE')
        # plt.title('Elbow Method')
        # st.pyplot(fig)

        st.write("#")
        st.caption("Definición automática del número de clústers a utilizar.")
        kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
        k = kl.knee
        kl.elbow
        plt.style.use('ggplot')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(kl.plot_knee())
        
        nclusters = st.text_input('¿Cuántos clusters desea utilizar?', k)
        if nclusters:
            k = nclusters

        MParticional = KMeans(n_clusters=int(k), random_state=0).fit(MEstandarizada)
        MParticional.predict(MEstandarizada)
        # st.write(MParticional.labels_)

        DataFrameArchivo['cluster'] = MParticional.labels_
        st.write("#")
        st.write(DataFrameArchivo)

        st.write("#")
        st.caption('Número de elementos en cada cluster')
        st.write(DataFrameArchivo.groupby(['cluster'])['cluster'].count())

        st.write("#")
        st.caption('Centroides para cada cluster')
        CentroidesP = DataFrameArchivo.groupby('cluster').mean()
        st.write(CentroidesP)

        st.write("#")
        st.caption("Gráfica de los elementos y los centros de los clusters")
        from mpl_toolkits.mplot3d import Axes3D
        plt.rcParams['figure.figsize'] = (10, 7)
        plt.style.use('ggplot')
        colores=['red', 'blue', 'green', 'yellow']#,'brown','black','violet', 'orange', 'cyan','darkred','lime']
        asignar=[]
        for row in MParticional.labels_:
            print(row)
            asignar.append(colores[row])

        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(MEstandarizada[:, 0], 
                MEstandarizada[:, 1], 
                MEstandarizada[:, 2], marker='o', c=asignar, s=60)
        ax.scatter(MParticional.cluster_centers_[:, 0], 
                MParticional.cluster_centers_[:, 1], 
                MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
        st.pyplot(fig)