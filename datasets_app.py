# datasets_app.py
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

DATA_DIR = Path("data")
DATASETS_TSV = DATA_DIR / "datasets_metadata.tsv"

def datasets_scikit_learn():
    st.header("Datasets Scikit-Learn")

    # ---------------------------
    # Ler TSV corretamente e corrigir deslocamento
    # ---------------------------
    df = pd.read_csv(DATASETS_TSV, sep="\t", dtype=str, index_col=False, engine="python")
    df.fillna("False", inplace=True)

    # Realinha caso o nome do dataset esteja deslocado
    if not df.iloc[0,0].startswith("iris"):
        df = df.shift(axis=1)

    # Colunas esperadas
    cols_order = [
        "Nome do dataset","Descrição do dataset","Número de registros","Colunas (parâmetros)","Fonte",
        "Modelos","Ensemble","Regressão","Linear","Redes Neurais","SVM","Clustering",
        "Seleção de Features","Redução de Dimensionalidade","Pré-Processamento","Seleção",
        "Textos","Vizinhos","Semi-Supervisionado","Probabilístico","Covariância",
        "Estatística","Kernel","Otimização","Detecção de Outliers","Tópicos",
        "Multi-output","Naive Bayes","Transformação"
    ]
    df = df[cols_order]

    # ---------------------------
    # Corrige "Número de registros"
    # ---------------------------
    df["Número de registros"] = df["Número de registros"].str.replace(".", "", regex=False)
    df["Número de registros"] = pd.to_numeric(df["Número de registros"], errors="coerce")

    # ---------------------------
    # Colunas booleanas
    # ---------------------------
    bool_cols = [
        "Modelos","Ensemble","Regressão","Linear","Redes Neurais","SVM","Clustering",
        "Seleção de Features","Redução de Dimensionalidade","Pré-Processamento","Seleção",
        "Textos","Vizinhos","Semi-Supervisionado","Probabilístico","Covariância",
        "Estatística","Kernel","Otimização","Detecção de Outliers","Tópicos",
        "Multi-output","Naive Bayes","Transformação"
    ]
    for col in bool_cols:
        df[col] = df[col].map(lambda x: True if str(x).strip().lower() == "true" else False)

    # ---------------------------
    # Adicionar coluna "Baixado"
    # ---------------------------
    df["Baixado"] = df["Nome do dataset"].apply(lambda x: (DATA_DIR / f"{x}.csv").exists())

    # ---------------------------
    # Configurar colunas para st.dataframe
    # ---------------------------
    column_config = {
        "Nome do dataset": st.column_config.Column("Nome do dataset", width="large"),
        "Descrição do dataset": st.column_config.Column("Descrição", width="large"),
        "Número de registros": st.column_config.NumberColumn("Registros", min_value=0),
        "Colunas (parâmetros)": st.column_config.Column("Colunas", width="large"),
        "Fonte": st.column_config.Column("Fonte", width="medium"),
        "Baixado": st.column_config.CheckboxColumn("Baixado", help="Indica se o dataset já foi baixado")
    }
    for col in bool_cols:
        column_config[col] = st.column_config.CheckboxColumn(col, help=f"Compatível com {col}")

    # ---------------------------
    # Inicializa session_state
    # ---------------------------
    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = None

    # ---------------------------
    # Primeiro dataframe com seleção
    # ---------------------------
    event = st.dataframe(
        df.set_index("Nome do dataset"),
        column_config=column_config,
        hide_index=False,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key="datasets_editor"
    )

    # Captura seleção
    if event and hasattr(event, "selection") and event.selection.rows:
        dataset = df.iloc[event.selection.rows[0]]["Nome do dataset"]
        st.session_state["selected_dataset"] = dataset
        st.markdown(f"`{dataset}`")

    # ---------------------------
    # Botão resetar seleção
    # ---------------------------
    if st.session_state.get("selected_dataset"):
        st.write(f"Dataset selecionado: **{st.session_state['selected_dataset']}**")
        if st.button("Resetar seleção"):
            st.session_state["selected_dataset"] = None

    # ---------------------------
    # Baixar dataset se não existir
    # ---------------------------
    if st.session_state.get("selected_dataset"):
        dataset_name = st.session_state["selected_dataset"]
        file_path = DATA_DIR / f"{dataset_name}.csv"
        if not file_path.exists():
            download_dataset(dataset_name, file_path)
            st.success(f"{dataset_name} baixado com sucesso!")

        # ---------------------------
        # Mostrar dataset baixado
        # ---------------------------
        df_dataset = pd.read_csv(file_path)
        numeric_cols = df_dataset.select_dtypes(include="number").columns.tolist()

        # Estatísticas por coluna
        stats = []
        for col in numeric_cols:
            series = df_dataset[col]
            stats.append({
                "Coluna": col,
                "Mínimo": series.min(),
                "Máximo": series.max(),
                "Média": series.mean(),
                "Mediana": series.median(),
                "Desvio Padrão": series.std(),
                "Nulos": series.isna().sum(),
                "Valores": series.tolist()
            })
        df_stats = pd.DataFrame(stats).set_index("Coluna")

        # Checkbox para normalização
        normalize = st.checkbox("Normalizar valores para gráfico", value=False, key=f"normalize_{dataset_name}")
        if normalize:
            for idx, row in df_stats.iterrows():
                values = row["Valores"]
                min_val = min(values)
                max_val = max(values)
                if max_val != min_val:
                    normalized = [(v - min_val) / (max_val - min_val) for v in values]
                else:
                    normalized = [0.5 for v in values]
                df_stats.at[idx, "Valores"] = normalized

        # Configura o gráfico
        column_config_stats = {
            "Valores": st.column_config.LineChartColumn(
                "Valores",
                y_min=df_stats["Valores"].apply(lambda x: min(x)).min(),
                y_max=df_stats["Valores"].apply(lambda x: max(x)).max()
            )
        }

        # Exibe dataframe com estatísticas + gráfico
        st.dataframe(
            df_stats,
            column_config=column_config_stats,
            hide_index=False,
            use_container_width=True
        )


# ---------------------------
# Função para baixar datasets do sklearn
# ---------------------------
def download_dataset(name: str, path: Path):
    dataset_map = {
        "iris": datasets.load_iris,
        "wine": datasets.load_wine,
        "breast_cancer": datasets.load_breast_cancer,
        "diabetes": datasets.load_diabetes,
        "california_housing": datasets.fetch_california_housing,
        "digits": datasets.load_digits,
        "moons": lambda: datasets.make_moons(n_samples=200, noise=0.1),
        "circles": lambda: datasets.make_circles(n_samples=200, noise=0.1, factor=0.5),
        "blobs": lambda: datasets.make_blobs(n_samples=200, centers=3, n_features=2),
        "outliers": lambda: datasets.make_blobs(n_samples=200, centers=1, n_features=2, cluster_std=10)
    }

    if name not in dataset_map:
        st.error(f"Dataset {name} não encontrado na lista de download automático.")
        return

    loader = dataset_map[name]
    data = loader()
    if isinstance(data, tuple):
        X, y = data
        df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
        df["target"] = y
    else:
        df = pd.DataFrame(data=data.data, columns=data.feature_names)
        if hasattr(data, "target"):
            df["target"] = data.target
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
