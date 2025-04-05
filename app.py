import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from module_ml import train_model
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Examen outil versioning", layout="wide")
st.title("EXAMEN OUTIL VERSIONING")
st.divider()

col1, col2 = st.columns(2)
file = col1.file_uploader(label="Veuillez charger un fichier csv contenant les données: ", type="csv")

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.title("Matrice de confusion")
    plt.show()

if file:
    data = pd.read_csv(file)
    col2.subheader("APERÇU DU FICHIER")
    col2.write(data.head())
    st.divider()

    col3, col4 = st.columns(2)
    target = col3.selectbox("Sélectionnez la variable cible: ", data.columns)

    if data[f"{target}"].dtype == "object":
        models_list = ["Classification svm", "Classification Randomforest"]
    else:
        models_list = ["Regression linéaire", "Regression Randomforest"]

    model = col4.selectbox("Sélectionnez le modèle à utiliser: ", models_list)

    btn = st.button("Lancer l'entrainement", type="primary")

    if btn:
        y_test, preds = train_model(data, target, model)
        if model in ["Classification svm", "Classification Randomforest"]:
            plot_confusion_matrix(y_true=y_test, y_pred=preds)

    st.divider()
