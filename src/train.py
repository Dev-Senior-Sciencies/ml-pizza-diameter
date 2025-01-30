import os
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from omegaconf import OmegaConf
import mlflow

file_path = os.getcwd()
conf = OmegaConf.load(os.path.join(file_path, "..", "src", "config.yml"))
mlflow.set_experiment(conf["tracking_uri"]["experiment_name"])

data_path = os.path.join(file_path, "..", "data", "pizzas-diameter.csv")
df = pd.read_csv(data_path)

modelo, scaler = None, None

def train(df, params):
    global modelo, scaler
    if modelo is not None and scaler is not None:
        return modelo, scaler

    with mlflow.start_run():
        X = df[["diametro"]]
        y = df["preco"]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        modelo = LinearRegression()
        modelo.fit(X, y)

        y_pred_train = modelo.predict(X)

        main_x_train = mean_absolute_error(y, y_pred_train)
        main_y_train = mean_squared_error(y, y_pred_train)
        r2_train = r2_score(y, y_pred_train)

        mlflow.log_params(params)
        mlflow.log_metrics({
            "main_x_train": main_x_train,
            "main_y_train": main_y_train,
            "r2_train": r2_train
        })
        mlflow.sklearn.log_model(modelo, "modelo_pizza")

        return modelo, scaler

modelo, scaler = train(df, conf["parameters"])

def main():
    st.set_page_config(page_title="Previsão de Preço da Pizza", layout="wide")
    st.title("🍕 Previsão de Preço da Pizza")
    st.markdown("---")

    st.sidebar.title("Configuração do Modelo")
    st.sidebar.subheader("Escolha o Diâmetro da Pizza")

    diametros_predefinidos = list(range(40, 401, 2))
    
    diametro_selecionado = st.sidebar.selectbox("Selecione um diâmetro (cm)", diametros_predefinidos, key='selectbox_diametro')

    diametro_personalizado = st.sidebar.number_input("Ou digite um diâmetro (cm)", min_value=10, max_value=400, value=diametro_selecionado, key='input_diametro')

    if diametro_personalizado != diametro_selecionado:
        diametro_selecionado = diametro_personalizado

    def prever_preco(diametro):
        diametro_transformado = scaler.transform([[diametro]])
        preco_previsto = modelo.predict(diametro_transformado)[0]
        
        st.write(f"O valor da pizza com diâmetro {diametro:.2f} cm é de **R$ {preco_previsto:.2f}**.")
        st.subheader("💰 Resultado da Previsão")
        st.balloons()

    if st.session_state.get('selectbox_diametro') or st.session_state.get('input_diametro'):
        prever_preco(diametro_selecionado)

if __name__ == "__main__":
    main()