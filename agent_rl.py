import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
import plotly.express as px

def run_agent():
    st.info("Iniciando execução do agente com os dados selecionados...")

    if "agent_data" not in st.session_state:
        st.error("Nenhum dataset disponível no session_state!")
        return

    X = st.session_state["agent_data"]["X"]
    y = st.session_state["agent_data"]["y"]

    # --- Definir se é classificação ou regressão
    classification_task = False
    if y is not None:
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() <= 20:  # simplificação: target discreto = classificação
                classification_task = True
        else:
            classification_task = True

    # --- Separar treino/teste
    if y is None:
        X_train, X_test = X, X
        y_train, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Selecionar estimadores
    estimators = []
    if classification_task:
        estimators = [
            ("LogisticRegression1", LogisticRegression(max_iter=1)),
            ("LogisticRegression2", LogisticRegression(max_iter=2)),
            ("LogisticRegression4", LogisticRegression(max_iter=4)),
            ("LogisticRegression8", LogisticRegression(max_iter=8)),
            ("LogisticRegression16", LogisticRegression(max_iter=16)),
            ("LogisticRegression32", LogisticRegression(max_iter=32)),
            ("LogisticRegression64", LogisticRegression(max_iter=64)),
            ("LogisticRegression128", LogisticRegression(max_iter=128)),
            ("LogisticRegression256", LogisticRegression(max_iter=256)),
            ("LogisticRegression512", LogisticRegression(max_iter=512)),
            ("LogisticRegression1024", LogisticRegression(max_iter=1024)),
            ("RandomForestClassifier", RandomForestClassifier()),
            ("SVC", SVC())
        ]
    else:
        estimators = [
            ("LinearRegression", LinearRegression()),
            ("RandomForestRegressor", RandomForestRegressor()),
            ("SVR", SVR())
        ]

    # --- Testar cada estimador e coletar resultados
    results = []
    for name, model in estimators:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        if y is None:
            # Não há target, apenas armazenar variáveis
            score = None
        else:
            scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2" if not classification_task else "accuracy")
            score = scores.mean()
        results.append({"Estimator": name, "Score": score})

    df_results = pd.DataFrame(results)

    # --- Mostrar tabela de resultados
    st.subheader("Resultados do agente")
    st.dataframe(df_results)

    # --- Gráfico de comparação
    fig = px.bar(df_results, x="Estimator", y="Score", text="Score")
    st.plotly_chart(fig, use_container_width=True)

    # --- Para datasets 2D ou 3D, mostrar scatterplot com cores (se y existir)
    if y is not None and X.shape[1] >= 2:
        fig2d = px.scatter(X, x=X.columns[0], y=X.columns[1], color=y if len(y.shape)==1 else y.columns[0])
        fig2d.update_traces(marker=dict(size=4))
        st.plotly_chart(fig2d, use_container_width=True)

    if y is not None and X.shape[1] >= 3:
        fig3d = px.scatter_3d(X, x=X.columns[0], y=X.columns[1], z=X.columns[2],
                              color=y if len(y.shape)==1 else y.columns[0])
        fig3d.update_traces(marker=dict(size=3))
        st.plotly_chart(fig3d, use_container_width=True)