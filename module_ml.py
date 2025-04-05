from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
import seaborn as sns

def preprocess_data(df, target_column):

    object_columns = df.drop([target_column], axis=1).select_dtypes('object')
    if not object_columns.empty :
        encoder = LabelEncoder()
        encoder.fit(object_columns)
        object_columns_encoded = pd.DataFrame(encoder.transform(object_columns), columns=object_columns.columns)
        df.drop(object_columns, axis=1, inplace=True)
        cleaned_df = df.join(object_columns_encoded)
    else:
        cleaned_df = df.drop([target_column], axis=1)

    x = cleaned_df
    y = df[target_column]

    if y.dtype == 'object':
        y = pd.factorize(y)[0]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    return train_test_split(x_scaled, y, test_size=0.2, random_state=42)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Prédictions")
    plt.ylabel("Vraies valeurs")
    plt.title("Matrice de confusion")
    plt.show()

def train_model(df, target_column, model_name):
    x_train, x_test, y_train, y_test = preprocess_data(df, target_column)

    if model_name == 'Regression linéaire':
        model = LinearRegression()
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        # mse = mean_squared_error(y_test, preds)
        # plt.scatter(y_test, preds)
        # plt.xlabel("Vraies valeurs")
        # plt.ylabel("Prédictions")
        # plt.title("Régression : Vraies vs Prédictions")
        # plt.show()
        return y_test, preds

    elif model_name == "Classification svm":
        model = SVC()
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        # print("Rapport de classification :")
        # print(classification_report(y_test, preds))
        # plot_confusion_matrix(y_test, preds)
        return y_test, preds

    elif model_name == "Classification Randomforest":
        model = RandomForestClassifier(n_estimators=100)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        # print("Rapport de classification :")
        # print(classification_report(y_test, preds))
        # plot_confusion_matrix(y_test, preds)
        return y_test, preds

    elif model_name == "Regression Randomforest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        # mse = mean_squared_error(y_test, preds)
        # print(f"Mean Squared Error: {mse:.4f}")
        # plt.scatter(y_test, preds)
        # plt.xlabel("Vraies valeurs")
        # plt.ylabel("Prédictions")
        # plt.title("Régression : Vraies vs Prédictions")
        # plt.show()
        return y_test, preds

    else:
        print("Modèle non reconnu.", model_name)
