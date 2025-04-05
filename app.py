import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from module_ml import train_model
from sklearn.metrics import confusion_matrix, mean_squared_error

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
    target = col3.selectbox("Sélectionnez la variable cible: ", data.columns)

    if data[f"{target}"].dtype == "object":
        models_list = ["Classification svm", "Classification Randomforest"]
    else:
        models_list = ["Regression linéaire", "Regression Randomforest"]

    model = col4.selectbox("Sélectionnez le modèle à utiliser: ", models_list)

    btn = st.button("Lancer l'entrainement", type="primary")

    if btn:
        chemin_model, y_test, preds = train_model(data, target, model)

        # AFFICHAGE DES RESULTATS
        if model in ["Regression linéaire", "Regression Randomforest"]:
            # Calculer l'erreur quadratique moyenne (MSE)
            mse = mean_squared_error(y_test, preds)

            # Afficher l'erreur dans Streamlit
            st.write(f"Erreur quadratique moyenne (MSE): {mse:.2f}")

            # Créer un graphique de comparaison entre vraies valeurs et prédictions
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, color='blue')
            ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red',
                    linestyle='--')  # Ligne d'égalité
            ax.set_xlabel("Vraies valeurs")
            ax.set_ylabel("Prédictions")
            ax.set_title("Régression : Vraies vs Prédictions")

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)

        elif model in ["Classification svm", "Classification Randomforest"]:
            # Calculer la matrice de confusion
            cm = confusion_matrix(y_test, preds)

            # Affichage de la matrice de confusion avec Seaborn
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Prédictions")
            ax.set_ylabel("Réel")
            ax.set_title("Matrice de Confusion")

            # Afficher la matrice dans Streamlit
            st.pyplot(fig)

        with open(chemin_model, "rb") as f:
            model_file = f.read()

        # Bouton pour télécharger le modèle
        st.download_button(
            label="Télécharger le modèle entrainé",
            data=model_file,
            file_name="model.pkl",
            mime="application/octet-stream"
        )

    st.divider()
