import streamlit as st
import pandas as pd

st.set_page_config(page_title="Examen outil versioning", layout="wide")
st.title("EXAMEN OUTIL VERSIONING")
st.divider()

col1, col2 = st.columns(2)
file = col1.file_uploader(label="Veuillez charger un fichier csv contenant les données: ", type="csv")

if file:
    data = pd.read_csv(file)
    col2.subheader("APERÇU DU FICHIER")
    col2.write(data.head())
    st.divider()

    col3, col4 = st.columns(2)
    col3.selectbox("Sélectionnez la variable cible: ", data.columns)
    col4.selectbox("Sélectionnez le modèle à utiliser: ",
                   ["Régression linéraire", "Classification avec SVM", "Classification avec Random Forest",
                    "Régression linéaire avec Random Forest"])

    btn = st.button("Lancer l'entrainement", type="primary")
    st.divider()
