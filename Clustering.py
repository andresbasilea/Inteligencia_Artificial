from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
from datetime import datetime

def app():
    colorFondo = '#FEFBF3'
    st.title("Clústering")
    archivo = st.file_uploader("Ingrese el archivo deseado")
    if archivo != None:
        st.write("Usted seleccionó el archivo " + "*"+archivo.name+"*" + "   de tipo   " + "*"+archivo.type+"*")

        DataFrameArchivo = pd.read_csv(archivo, error_bad_lines=False)

        datos_opcional = st.expander("Datos Raw")
        datos_opcional.write(DataFrameArchivo)


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
        st.write("#")
        st.caption("Matriz estandarizada")
        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
        MEstandarizada = estandarizar.fit_transform(MatrizSeleccion)
        st.write(pd.DataFrame(MEstandarizada))

        opcion = st.selectbox("Elija el tipo de clústering a utilizar", ("Particional", "Jerárquico"))
        
        
        if opcion == "Particional":
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
            #cambiado = 0
            k2 = kl.knee
            
            if nclusters:
                k = nclusters
                #cambiado = 1

            MParticional = KMeans(n_clusters=int(k), random_state=0).fit(MEstandarizada)
            MParticional.predict(MEstandarizada)
            # st.write(MParticional.labels_)

            DataFrameArchivo['cluster'] = MParticional.labels_
            st.write("#")
            st.write(DataFrameArchivo)

            st.write("#")
            st.caption('Número de elementos en cada cluster')
            st.write(DataFrameArchivo.groupby(['cluster'])['cluster'].count())

            # st.write("#")
            # st.caption('Centroides para cada cluster')
            # CentroidesP = DataFrameArchivo.groupby('cluster').mean()
            # st.write(CentroidesP)

            if int(k2) == int(k):
                st.write("#")
                st.caption("Gráfica de los elementos y los centros de los clusters")
                from mpl_toolkits.mplot3d import Axes3D
                plt.rcParams['figure.figsize'] = (10, 7)
                plt.style.use('ggplot')
                colores=['red', 'blue', 'green', 'yellow']#,'brown','black','violet', 'orange', 'cyan','darkred','lime']
                asignar=[]
                for row in MParticional.labels_:
                    #print(row)
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


        if opcion == "Jerárquico":
            fig = plt.figure(figsize=(10, 7))
            plt.title("Casos de hipoteca")
            plt.xlabel('Hipoteca')
            plt.ylabel('Distancia')
            Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))
            plt.axhline(y=5.4, color='orange', linestyle='--')
            st.caption("Visualización de los clústers creados")
            st.pyplot(fig)

            st.subheader("Número de clústers para análisis")
            nclusters = st.text_input("Ingrese el número de clústers a utilizar", 9)
            MJerarquico = AgglomerativeClustering(n_clusters=int(nclusters), linkage='complete', affinity='euclidean')
            MJerarquico.fit_predict(MEstandarizada)
            st.write(MJerarquico.labels_)

            st.caption("Los elementos y sus respectivos clústers")
            DataFrameArchivo = DataFrameArchivo.drop(columns=['comprar'])
            DataFrameArchivo['cluster'] = MJerarquico.labels_
            st.write(DataFrameArchivo)

        nElemCluster  = DataFrameArchivo.groupby(['cluster'])['cluster'].count()
        st.write(nElemCluster)
        
        numClusterSeleccion = st.slider("Seleccione de qué clúster quiere visualizar sus elementos",min_value = 0, max_value = max(DataFrameArchivo['cluster']), value = 0)
        st.write(DataFrameArchivo[DataFrameArchivo.cluster == int(numClusterSeleccion)])

        st.write("#")
        st.caption("Centroides")
        CentroidesH = DataFrameArchivo.groupby('cluster').mean()
        st.write(CentroidesH)


        ObservacionesClusters = CentroidesH.values.tolist()
        
        Observaciones = []
        DatosCluster = "Fecha: " + str(datetime.now())
        DatosCluster += "Análisis clusters. Fuente de datos: " + archivo.name + "\n"
        DatosCluster += "Método utilizado: Clustering " + opcion + "\n\n"
        for contador in range (0, int(nclusters)):
            st.caption("Cluster " + str(contador) + ":")
            DatosCluster += "Cluster "+ str(contador) + ":\n"
            contador2 = 0
            st.write("Número de elementos en el cluster: " + str(nElemCluster[contador]))
            DatosCluster += "Número de elementos en el cluster: " + str(nElemCluster[contador]) + "\n"
            for i in CentroidesH:
                st.write(i + ' :', ObservacionesClusters[contador][contador2])
                DatosCluster += i + ' :' + str(ObservacionesClusters[contador][contador2]) + "\n"
                contador2+=1
            
            Observaciones.append(st.text_input('Observaciones cluster ' + str(contador) + ' :', ""))
            DatosCluster += "\nObservaciones:\n"
            DatosCluster += Observaciones[contador] + "\n\n"
        st.download_button('Descargar análisis de clusters', file_name='clustering' + opcion +'.txt', data = DatosCluster)


        # st.caption("Analicemos el cluster # ")
        # numcluster = st.text_input("Ingresa el número de clúster a analizar", 0)
    