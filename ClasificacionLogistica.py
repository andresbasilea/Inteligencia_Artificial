from datetime import datetime
from numpy.lib.shape_base import column_stack
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


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

        st.write("#")
        st.caption("Selección de variables predictoras y variable clase")

        variableClase = st.selectbox("Seleccione la variable clase", opcionesVariables)
        variableClaseList = []
        variableClaseList.append(variableClase)

        variablesPredictoras = list(set(opcionesVariables) - set(variableClaseList)) + list(set(variableClaseList) - set(opcionesVariables))



        # opcionesPredictoras = []
        # for columna in opcionesVariables:
        #     opcionesPredictoras.append(columna)
        # opcionesPredictorasSeleccionadas = st.multiselect("Seleccione las variables predictoras del algoritmo", opcionesPredictoras, default=opcionesPredictoras)
        

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


        pd.DataFrame(X_train)
        pd.DataFrame(Y_train)
        Clasificacion = linear_model.LogisticRegression()
        Clasificacion.fit(X_train, Y_train) 
        
        st.caption("Predicciones probabilísticas")
        Probabilidad = Clasificacion.predict_proba(X_validation)
        st.write(pd.DataFrame(Probabilidad))

        st.caption("Predicciones con clasificación final")
        Predicciones = Clasificacion.predict(X_validation)
        st.write(pd.DataFrame(Predicciones))
    

        #validacion

        st.markdown("__Validación del modelo__")
        Y_Clasificacion = Clasificacion.predict(X_validation)
        Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
        st.caption("Matriz de clasificación") 
        st.write(Matriz_Clasificacion)
        st.write("Exactitud", Clasificacion.score(X_validation, Y_validation))

        st.markdown("__Ecuación del modelo de clasificación__") 
        
        st.write("Intercept:", str(Clasificacion.intercept_))
        st.write('Coeficientes: \n', (Clasificacion.coef_))
        

########################################################### 

        st.write("La ecuación del modelo de clasificacion tiene la forma: Prob = 1/1+𝑒^−(𝑎+𝑏𝑋))\n")
        
        ecuacion = "Ecuación del modelo de clasificación"
        ecuacion += "1/1+𝑒^−("
        ecuacion += ""

        n = len(variablesPredictoras)
    
        CoeficientesEcuacion= Clasificacion.coef_.tolist()
        Intercept = Clasificacion.intercept_.tolist()

        st.write("a+bX = " + str(Intercept[0]))
        ecuacion += "a+bX = " + str(Intercept[0]) + "\n"
        for i in range (0, n) :
            if float(CoeficientesEcuacion[0][i]) < 0:
                st.write( str(CoeficientesEcuacion[0][i]) + "[" + str(variablesPredictoras[i]) + "]")
                ecuacion += str(CoeficientesEcuacion[0][i]) + "[" + str(variablesPredictoras[i]) + "]\n"
            else:
                st.write( "+" + str(CoeficientesEcuacion[0][i]) + "[" + str(variablesPredictoras[i]) + "]")
                ecuacion +=  "+" + str(CoeficientesEcuacion[0][i]) + "[" + str(variablesPredictoras[i]) + "]\n"
        
        Descarga = "Fecha: " + str(datetime.now())
        Descarga += "\nAlgoritmo: Regresión Logística"
        Descarga += "\nFuente de datos: " + archivo.name + "\n\n"
        Descarga += "Variables Predictoras:\n"
        for elemento in variablesPredictoras:
            Descarga += elemento + ", "
        Descarga += "\n\nVariable Clase:\n"
        Descarga += variableClase
        Descarga += "Test size: " + test_size_input + "\n"
        Descarga += "Random state: " + random_state_input + "\n\n"
        st.download_button('Descargar modelo generado', file_name='modeloRL.txt', data = Descarga + ecuacion)


        #nuevas predicciones
        with st.expander("Nuevas predicciones"):
            st.write("#")
            st.write("#")
            st.write("#")
            st.caption("Nuevas Predicciones") 

            NPrediccion = ""
            NPrediccion += "Nueva Predicción: \n\n"
            lista = []
            contador = 0
            for j in variablesPredictoras :
                contador += 1
                lista.append(st.text_input(j, 0))
                NPrediccion += str(j) + ": " + lista[contador-1] + "\n"
            st.caption("Clasificación de la predicción:")
            TextoPrediccion = ""
            for j in range (0, len(variablesPredictoras)) :
                if j == len(variablesPredictoras)-1: 
                    TextoPrediccion += "" + str(lista[j]) +""
                else:
                    TextoPrediccion += "" + str(lista[j]) +", "
            NuevaPrediccion = pd.DataFrame(x.split(',') for x in TextoPrediccion.split('\n'))
            #st.write(NuevaPrediccion)
            arreglo = Clasificacion.predict(NuevaPrediccion)
            st.write(arreglo[0])
            NPrediccion += "Clasificación de la nueva predicción: " + str(arreglo[0]) + "\n\n"
            st.download_button('Descargar nueva predicción', file_name='nuevaPrediccionRL.txt', data = Descarga + NPrediccion)