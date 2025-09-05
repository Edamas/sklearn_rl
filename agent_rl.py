import streamlit as st
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import plotly.express as px
import numpy as np

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
    from sklearn.linear_model import (
        LogisticRegression, RidgeClassifier, Perceptron, LinearRegression, 
        LogisticRegressionCV, SGDClassifier, PassiveAggressiveRegressor, 
        PassiveAggressiveClassifier, GammaRegressor, RidgeClassifierCV
    )
    from sklearn.ensemble import (
        RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, 
        AdaBoostClassifier, HistGradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
    )
    from sklearn.svm import SVC, LinearSVC, NuSVC, SVR, LinearSVR, NuSVR
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering, FeatureAgglomeration
    from sklearn.multioutput import ClassifierChain, RegressorChain
    from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
    from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.semi_supervised import LabelPropagation, LabelSpreading
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import CategoricalNB, ComplementNB, BernoulliNB
    from sklearn.neural_network import MLPClassifier

    classifiers = [
        ('sklearn.ensemble.GradientBoostingClassifier', GradientBoostingClassifier(random_state=42)),
        ('sklearn.ensemble.RandomForestClassifier', RandomForestClassifier(random_state=42)),
        ('sklearn.ensemble.HistGradientBoostingClassifier', HistGradientBoostingClassifier(random_state=42)),
        ('sklearn.ensemble.BaggingClassifier', BaggingClassifier(random_state=42)),
        ('sklearn.ensemble.ExtraTreesClassifier', ExtraTreesClassifier(random_state=42)),
        ("sklearn.linear_model.LogisticRegression", LogisticRegression(solver='lbfgs', max_iter=200, random_state=42)),
        ('sklearn.ensemble.AdaBoostClassifier', AdaBoostClassifier(random_state=42)),
        ('sklearn.svm.LinearSVC', LinearSVC(max_iter=1000, random_state=42)),
        ("sklearn.linear_model.LogisticRegressionCV", LogisticRegressionCV(solver='lbfgs', max_iter=500, random_state=42)),
        ("sklearn.svm.SVC", SVC(kernel='rbf', probability=True, random_state=42)),
        ('sklearn.linear_model.RidgeClassifier', RidgeClassifier(random_state=42)),
        ('sklearn.linear_model.RidgeClassifierCV', RidgeClassifierCV()),
        ("sklearn.linear_model.SGDClassifier", SGDClassifier(max_iter=1000, random_state=42)),
        ('sklearn.neural_network.MLPClassifier', MLPClassifier(max_iter=300, random_state=42)),
        ('sklearn.linear_model.PassiveAggressiveClassifier', PassiveAggressiveClassifier(max_iter=1000, random_state=42)),
        ("sklearn.cluster.KMeans", KMeans(n_init='auto', random_state=42)),
        ('sklearn.cluster.DBSCAN', DBSCAN()),
        ('sklearn.multioutput.ClassifierChain', ClassifierChain(base_estimator=LogisticRegression(), random_state=42)),
        ('sklearn.multiclass.OneVsOneClassifier', OneVsOneClassifier(estimator=LogisticRegression())),
        ('sklearn.multiclass.OneVsRestClassifier', OneVsRestClassifier(estimator=LogisticRegression())),
        ('sklearn.calibration.CalibratedClassifierCV', CalibratedClassifierCV(estimator=LogisticRegression(), cv=5)),
        ('sklearn.gaussian_process.GaussianProcessClassifier', GaussianProcessClassifier(random_state=42)),
        ("sklearn.semi_supervised.LabelPropagation", LabelPropagation(kernel='rbf')),
        ("sklearn.semi_supervised.LabelSpreading", LabelSpreading(kernel='rbf')),
        ("sklearn.cluster.SpectralClustering", SpectralClustering(affinity='rbf', random_state=42)),
        ('sklearn.tree.ExtraTreeClassifier', ExtraTreeClassifier(random_state=42)),
        ('sklearn.naive_bayes.CategoricalNB', CategoricalNB()),
        ('sklearn.naive_bayes.ComplementNB', ComplementNB()),
        ('sklearn.cluster.FeatureAgglomeration', FeatureAgglomeration()),
        ('sklearn.naive_bayes.BernoulliNB', BernoulliNB())
    ]

    regressors = [
        ("LinearRegression", LinearRegression()),
        ("RandomForestRegressor", RandomForestRegressor(random_state=42)),
        ("SVR", SVR()),
        ("LinearSVR", LinearSVR(random_state=42)),
        ("PassiveAggressiveRegressor", PassiveAggressiveRegressor(max_iter=1000, random_state=42)),
        ("NuSVR", NuSVR()),
        ("GammaRegressor", GammaRegressor()),
        ("GaussianProcessRegressor", GaussianProcessRegressor(random_state=42)),
        ("RadiusNeighborsRegressor", RadiusNeighborsRegressor()),
        ("RegressorChain", RegressorChain(base_estimator=LinearRegression()))
    ]

    if classification_task:
        estimators = classifiers
    else:
        estimators = regressors

    # --- Testar cada estimador e coletar resultados
    results = []
    for name, model in estimators:
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        if y is None:
            # Não há target, apenas armazenar variáveis
            score = None
        else:
            try:
                scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2" if not classification_task else "accuracy")
                score = scores.mean()
            except ValueError as e:
                score = f"Error: {e}"
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


