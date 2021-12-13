from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

import graphviz
from sklearn.tree import export_graphviz
from sklearn.tree import plot_tree
from sklearn.tree import export_text
from datetime import datetime


def app():
    colorFondo = '#FEFBF3'

    st.title("Árbol de Decisión (clasificación)")

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

        st.write("#")
        st.caption("Selección de variables predictoras y variable a predecir")

        variableClase = st.selectbox("Seleccione la variable a predecir", opcionesVariables)
        variableClaseList = []
        variableClaseList.append(variableClase)

        variablesPredictoras = list(set(opcionesVariables) - set(variableClaseList)) + list(set(variableClaseList) - set(opcionesVariables))

        st.caption("Matriz con las variables predictoras")
        st.write(variablesPredictoras)
        MatrizSeleccionPredictoras = np.array(DataFrameArchivo[variablesPredictoras])
        MatrizSeleccionClase = np.array(DataFrameArchivo[variableClase])
        # st.write(MatrizSeleccionPredictoras)

        



        st.caption("A continuación, puede modificar los valores para la división de datos:")
        col1, col2 = st.columns(2)
        test_size_input = col1.text_input("variable 'test_size'", 0.2)
        random_state_input = col2.text_input("variable random_state", 1234)


        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(MatrizSeleccionPredictoras, MatrizSeleccionClase, 
                                                                                test_size = float(test_size_input), 
                                                                                random_state = int(random_state_input),
                                                                                shuffle = True)


        opcion = st.selectbox("Elija el tipo de árbol a utilizar", ("Pronóstico", "Clasificación"))
                                                                                

        st.write("#")
        st.write("#")
        st.caption("Aplicación del algoritmo")

        col1, col2, col3 = st.columns(3)
        profundidad_ = col1.text_input("Profundidad máxima", 7)
        min_sample_split_ = col2.text_input("Mínimo de separación muestras", 4)
        min_sample_leaf_ = col3.text_input("Mínimo elementos hoja", 2)
        if opcion == 'Pronóstico':
            ArbolDecision = DecisionTreeRegressor(max_depth=int(profundidad_), min_samples_split=int(min_sample_split_), min_samples_leaf=int(min_sample_leaf_))
        if opcion == 'Clasificación':
            ArbolDecision = DecisionTreeClassifier(max_depth=int(profundidad_), min_samples_split=int(min_sample_split_), min_samples_leaf=int(min_sample_leaf_))
        ArbolDecision.fit(X_train, Y_train)
        Y_Pronostico = ArbolDecision.predict(X_train)


        st.write("#")
        st.write("#")
        st.write("Validación del Algoritmo")
        if opcion == 'Pronóstico':
            st.write('Criterio: \n', ArbolDecision.criterion)
            st.write("MAE - Error Absoluto Medio: %.4f" % mean_absolute_error(Y_train, Y_Pronostico))
            st.write("MSE - Error Cuadrático Medio: %.4f" % mean_squared_error(Y_train, Y_Pronostico))
            st.write("RMSE - Raíz Cuadrada del Error Cuadrático Medio: %.4f" % mean_squared_error(Y_train, Y_Pronostico, squared=False))   #True devuelve MSE, False devuelve RMSE
            st.write('Score: %.4f' % r2_score(Y_train, Y_Pronostico))
        if opcion == 'Clasificación':
            Y_Clasificacion = ArbolDecision.predict(X_validation)
            Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación']) 
            st.write(Matriz_Clasificacion)
            st.write('Criterio: \n', ArbolDecision.criterion)
            
        st.write("Exactitud promedio de la validación: ", ArbolDecision.score(X_validation, Y_validation))
        st.caption("Importancia de las variables")
        Importancia = pd.DataFrame({'Variable': list(DataFrameArchivo[variablesPredictoras]), 'Importancia': + ArbolDecision.feature_importances_}).sort_values('Importancia', ascending=False)         
        st.write(Importancia)

        st.write("#")
        st.caption("Conformación del modelo")
       

        st.write("Árbol de Decisión")
       
        ArbolDecision2 = plt.figure(figsize=(16,16))  
        plot_tree(ArbolDecision, feature_names = variablesPredictoras)
        st.pyplot(ArbolDecision2)
        
        informe = export_text(ArbolDecision, feature_names = variablesPredictoras)
        informe = informe.split("\n")
        
        
        st.caption("Esquema del árbol generado")
        with st.expander("Esquema del árbol de decisión"):
            datos = "\nFecha de generación: " + str(datetime.now())
            datos += "\n\nEsquema generado de: " + archivo.name + "\n"
            datos += "Árbol de decisión: " +  opcion + "\n"
            datos += "Variables Predictoras:\n"
            for i in variablesPredictoras:
                datos += i + ", "
            datos += "\nVariable Clase:\n"
            
            datos += str(variableClase)
            datos += "\n\n\n"
            for i in informe:
                st.text(i)
                datos += i + "\n"
        
        
        col1, col2 = st.columns(2)
        plt.savefig('ArbolDecision' + opcion +'.png')
        with open('ArbolDecision' + opcion +'.png',"rb") as file:
            col1.download_button("Descargar árbol de decisión (.png)", data = file, file_name = 'ArbolDecision' + opcion +'.png', mime ="image/png")
        
        col2.download_button('Descargar informe del modelo generado (.txt)', file_name='InformeArbol' + opcion+ '.txt', data = datos)    


        with st.expander("Nuevas predicciones: "): 
            NPrediccion = "\n Nueva Predicción \n"
            lista = []
            contador = 0
            for i in variablesPredictoras:
                contador+=1
                lista.append(st.text_input(i,0))
                NPrediccion += str(i) + ": " + str(lista[contador-1]) + "\n"
           
            predicciontxt = ""
            for j in range (0, len(variablesPredictoras)) :
                if j == len(variablesPredictoras)-1: 
                    predicciontxt += "" + str(lista[j]) +""
                else:
                    predicciontxt += "" + str(lista[j]) +", "
            NuevaPrediccionPrint = pd.DataFrame(x.split(',') for x in predicciontxt.split('\n'))
            arr = ArbolDecision.predict(NuevaPrediccionPrint)
            st.write(arr[0])
            NPrediccion += "\nValor de la nueva predicción: " + str(arr[0]) + "\n\n"



            st.download_button('Descargar nueva predicción', file_name='ArbolDecision_NuevaPrediccion.txt', data = datos + NPrediccion)
            
            
