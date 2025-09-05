import streamlit as st
import pandas as pd
import graphviz as gv
from pathlib import Path
import sklearn.datasets as sk_datasets
from functions import df_select_single_row
from agent_rl import run_agent
import plotly.express as px

DATA_DIR = Path("data")
DATASETS_TSV = DATA_DIR / "datasets_metadata.tsv"
COR_DE_FUNDO = 'rgba(128,128,128,1)'
# --- 1️⃣ Função que verifica se existe e carrega
def load_dataset(dataset_name: str):
    filename = DATA_DIR / f"{dataset_name}.csv"
    if filename.exists():
        df = pd.read_csv(filename)
        return df
    return None

# --- 2️⃣ Função que gera/baixa dataset via eval
def download_dataset(dataset_name: str, download_command: str):
    st.toast(f"Dataset '{dataset_name}' não encontrado. Gerando...")
    try:
        result = eval(download_command, {"datasets": sk_datasets, "pd": pd})
    except Exception as e:
        st.error(f"Erro ao executar comando: {e}")
        return None

    # Converter resultado em DataFrame fiel
    if isinstance(result, tuple):
        X, y = result
        df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(X.shape[1])])
        if y is not None:
            df["target"] = y
    else:
        df = pd.DataFrame(result, columns=[f"feature{i+1}" for i in range(result.shape[1])])

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / f"{dataset_name}.csv", index=False, encoding="utf-8")
    st.toast(f"Dataset '{dataset_name}' gerado e salvo em {DATA_DIR / f'{dataset_name}.csv'}")
    return df

# --- 3️⃣ Função para construir stats para data_editor
def build_feature_table(df: pd.DataFrame):
    stats = pd.DataFrame(index=df.columns)
    stats["Feature Role"] = ["y" if str(c).lower() in ["target", "y", "destino"] else "X" for c in df.columns]
    stats["mean"] = df.mean(numeric_only=True)
    stats["std"] = df.std(numeric_only=True)
    stats["median"] = df.median(numeric_only=True)
    stats["min"] = df.min(numeric_only=True)
    stats["max"] = df.max(numeric_only=True)
    stats["n_nulls"] = df.isnull().sum()
    stats["n_non_nulls"] = df.notnull().sum()
    # Gráfico normalizado
    norm_df = (df - df.min()) / (df.max() - df.min())
    stats["Gráfico"] = [norm_df[col].tolist() if col in norm_df.columns else None for col in df.columns]
    return stats

# --- 4️⃣ Função principal
def datasets():
    st.subheader("📚 Datasets")
    if not DATASETS_TSV.exists():
        st.error(f"{DATASETS_TSV} não encontrado")
        st.stop()

    df_meta = pd.read_csv(DATASETS_TSV, sep="\t", index_col=0)

    # --- Seleciona dataset via df_select_single_row
    dataset_name = df_select_single_row(df_meta)
    if not dataset_name:
        st.stop()

    # --- Tentar carregar dataset
    df = load_dataset(dataset_name)
    if df is None:
        download_command = df_meta.loc[dataset_name, "DownloadCommand"]
        if not isinstance(download_command, str):
            download_command = str(download_command)
        df = download_dataset(dataset_name, download_command)
        if df is None:
            st.stop()
    st.success(f"Dataset '{dataset_name}' carregado com sucesso!")

    # --- Salvar dataset no session_state para o agente
    st.session_state["dataset"] = df

    # --- Construir tabela para data_editor
    stats = build_feature_table(df)
    edited_stats = st.data_editor(
        stats,
        column_config={
            "Feature Role": st.column_config.SelectboxColumn("Feature Role", options=["X", "y", "desativado"]),
            "Gráfico": st.column_config.LineChartColumn("Gráfico"),
        },
        hide_index=False,
        num_rows="fixed",
    )

    # --- Grafo
    def draw_graph(feature_table, has_target: bool):
        dot = gv.Digraph()
        dot.attr(rankdir="LR")
        for col, role in zip(feature_table.index, feature_table["Feature Role"]):
            if role == "X":
                dot.node(col, shape="box", color="lightblue")
                dot.edge(col, "Agente")
        if has_target:
            for col, role in zip(feature_table.index, feature_table["Feature Role"]):
                if role == "y":
                    dot.node(col, shape="ellipse", color="lightgreen")
                    dot.edge(col, "Rótulo")
        dot.node("Agente", shape="box", style="filled", color="lightgray")
        dot.node("Rótulo", shape="oval", style="filled", color="lightyellow")
        return dot

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Rede')
        dot = draw_graph(edited_stats, has_target="y" in edited_stats["Feature Role"].values)
        st.graphviz_chart(dot)

    # --- Scatterplots 2D e 3D
    X_cols = edited_stats.loc[edited_stats["Feature Role"]=="X"].index.tolist()
    y_cols = edited_stats.loc[edited_stats["Feature Role"]=="y"].index.tolist()
    target_col = y_cols[0] if y_cols else None

    with col2:
        st.subheader('2D')
        if len(X_cols) >= 2:
            fig2d = px.scatter(df, x=X_cols[0], y=X_cols[1], color=target_col)
            fig2d.update_traces(marker=dict(size=4))
            fig2d.update_layout(
                plot_bgcolor='rgba(128,128,128,0.5)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig2d, use_container_width=True)

    with col3:
        st.subheader('3D')
        if len(X_cols) >= 3:
            fig3d = px.scatter_3d(df, x=X_cols[0], y=X_cols[1], z=X_cols[2], color=target_col)
            fig3d.update_traces(marker=dict(size=3))
            fig3d.update_layout(
                scene=dict(
                    xaxis=dict(backgroundcolor='rgba(128,128,128,0.5)'),
                    yaxis=dict(backgroundcolor='rgba(128,128,128,0.5)'),
                    zaxis=dict(backgroundcolor='rgba(128,128,128,0.5)')
                ),
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig3d, use_container_width=True)

    # --- Treinamento do agente
    st.info("🚀 Iniciando agente...")

    # --- Determinar X e y a partir do data editor
    X_cols = edited_stats.loc[edited_stats["Feature Role"]=="X"].index.tolist()
    y_cols = edited_stats.loc[edited_stats["Feature Role"]=="y"].index.tolist()

    if len(X_cols) == 0:
        st.error("Não há colunas X selecionadas. O agente não pode treinar.")
        st.stop()

    X_df = df[X_cols]

    if len(y_cols) == 0:
        y_df = None
        st.warning("Nenhuma coluna y encontrada. Agente fará testes sem target (apenas inspeciona features).")
    else:
        if len(y_cols) == 1:
            y_df = df[y_cols[0]]
        else:
            y_df = df[y_cols]  # múltiplos targets possíveis

    # --- Armazenar no session_state
    st.session_state["agent_data"] = {"X": X_df, "y": y_df}

    # --- Executar agente
    from agent_rl import run_agent
    run_agent()

    st.success("✅ Treinamento do agente finalizado")

