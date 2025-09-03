# datasets_app.py
import pandas as pd
import streamlit as st
from pathlib import Path

DATA_DIR = Path("data")
DATASETS_TSV = DATA_DIR / "datasets_metadata.tsv"

def datasets_scikit_learn():
    st.header("Datasets Scikit-Learn")

    # ---------------------------
    # Ler TSV com a primeira coluna como índice
    # ---------------------------
    df = pd.read_csv(
        DATASETS_TSV,
        sep="\t",
        index_col=0,       # primeira coluna como índice
        header=0,          # primeira linha como cabeçalho
        encoding="utf-8",
        engine="python"
    )

    # ---------------------------
    # Adicionar coluna "Baixado"
    # ---------------------------
    df["Baixado"] = df.index.to_series().apply(lambda x: (DATA_DIR / f"{x}.csv").exists() if pd.notna(x) else False)

    # ---------------------------
    # Configurar colunas Streamlit
    # ---------------------------
    column_config = {}
    for col in df.columns:
        if col in ["Baixado"]:
            column_config[col] = st.column_config.CheckboxColumn(col)
        else:
            column_config[col] = st.column_config.Column(col, width="large")

    # ---------------------------
    # Inicializar session_state
    # ---------------------------
    if "selected_dataset" not in st.session_state:
        st.session_state["selected_dataset"] = None

    # ---------------------------
    # Dataframe com seleção
    # ---------------------------
    event = st.dataframe(
        df,
        column_config=column_config,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
        key="datasets_editor"
    )

    # Captura seleção
    selection = getattr(event, "selection", None)
    if selection and getattr(selection, "rows", None):
        dataset = df.index[selection.rows[0]]
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
            dataset_data = eval(str(df.loc[dataset_name, "DownloadCommand"]))
            if hasattr(dataset_data, "data"):
                df_dataset = pd.DataFrame(dataset_data.data, columns=dataset_data.feature_names)
                if hasattr(dataset_data, "target"):
                    df_dataset["target"] = dataset_data.target
            else:  # Tuplas sintéticas (X, y)
                X, y = dataset_data
                df_dataset = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
                df_dataset["target"] = y

            file_path.parent.mkdir(parents=True, exist_ok=True)
            df_dataset.to_csv(file_path, index=False)
            st.success(f"{dataset_name} baixado com sucesso!")

        # ---------------------------
        # Mostrar dataset baixado
        # ---------------------------
        df_dataset = pd.read_csv(file_path)
        numeric_cols = df_dataset.select_dtypes(include="number").columns.tolist()

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
                "Únicos": series.nunique(),
                "Valores": series.tolist()
            })
        df_stats = pd.DataFrame(stats).set_index("Coluna")

        normalize = st.checkbox("Normalizar valores para gráfico", value=False, key=f"normalize_{dataset_name}")
        if normalize:
            for idx, row in df_stats.iterrows():
                values = row["Valores"]
                min_val = min(values)
                max_val = max(values)
                df_stats.at[idx, "Valores"] = [(v - min_val) / (max_val - min_val) if max_val != min_val else 0.5 for v in values]

        column_config_stats = {
            "Valores": st.column_config.LineChartColumn(
                "Valores",
                y_min=df_stats["Valores"].apply(lambda x: min(x)).min(),
                y_max=df_stats["Valores"].apply(lambda x: max(x)).max()
            )
        }

        st.dataframe(
            df_stats,
            column_config=column_config_stats,
            width="stretch"
        )
