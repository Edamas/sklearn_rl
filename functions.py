import streamlit as st
import pandas as pd
import numpy as np
import graphviz as gv
import os
from datetime import datetime
import traceback
from typing import Optional

LOG_FILE = "log.tsv"

def log_message(level: str, message: str, exception: Optional[Exception] = None, display_streamlit: bool = True):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp}\t{level}\t{message}"

    if exception:
        log_entry += f"\t{traceback.format_exc()}"
    else:
        log_entry += "\t" # Add empty column for consistency

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_entry + "\n")

    if display_streamlit:
        if level == "ERROR" or level == "EXCEPTION":
            st.error(f"Um erro ocorreu. Veja {LOG_FILE} para detalhes: {message}")
        elif level == "WARNING":
            st.warning(f"Um aviso ocorreu. Veja {LOG_FILE} para detalhes: {message}")
        else: # INFO
            st.info(message)


def df_select_rows(df, selection_mode: Optional[str] = 'multi-row', prompt: Optional[str] = None, key: str = "dataframe_selection"):
    """
    NÃO MODIFICAR ESSA FUNÇÃO, POIS ELA É IMPORTANTE PARA TODOS OS DATAFRAMES SELECIONÁVEIS DO PROJETO.
    Retorna um DataFrame interativo para seleção de linhas.
    O modo de seleção pode ser 'single-row' ou 'multi-row'.
    A seleção é armazenada em st.session_state[key].
    """
    event = st.dataframe(
        df,
        width="stretch",
        height=300,
        hide_index=False,
        on_select="rerun",
        selection_mode=selection_mode,
        key=key # Add a key to the dataframe
    )
    
    # Get selection from session state
    selection = event.get('selection')

    if selection and selection.get('rows'):
        row_indices = selection['rows']
        if not row_indices:
            return None if selection_mode == 'single-row' else []

        if selection_mode == 'single-row':
            return df.index[row_indices[0]]
        else: # multi-row
            return df.index[row_indices].tolist()

    if prompt:
        st.info(prompt)
    
    return None if selection_mode == 'single-row' else []


def build_feature_table(X: pd.DataFrame):
    """Cria tabela de resumo das features, com roles e estatísticas básicas."""

    desc = X.describe(include="all").transpose()

    # Contagem de nulos e não-nulos
    nulls = X.isnull().sum()
    non_nulls = X.notnull().sum()

    # Mediana
    medians = X.median(numeric_only=True)

    # Tipo da coluna
    dtypes = X.dtypes.astype(str)

    # Normalização para o gráfico
    normalized = (X - X.min()) / (X.max() - X.min())

    # Monta dataframe final
    summary = pd.DataFrame({
        "Feature Role": ["X"] * len(X.columns),
        "Coluna": X.columns,
        "Mínimo": desc["min"].fillna(""),
        "Média": desc["mean"].fillna(""),
        "Mediana": medians.reindex(X.columns).fillna(""),
        "Máximo": desc["max"].fillna(""),
        "Desvio Padrão": desc["std"].fillna(""),
        "Nulos": nulls,
        "Não-Nulos": non_nulls,
        "Tipo": dtypes,
        "Gráfico": [normalized[col].tolist() for col in X.columns],
    })

    return summary


def draw_graph(feature_table: pd.DataFrame, has_target: bool):
    """Desenha grafo com Graphviz baseado nos papéis das features."""
    dot = gv.Digraph()
    dot.attr(rankdir="LR")

    for col, role in zip(feature_table["Coluna"], feature_table["Feature Role"]):
        if role == "X":
            dot.node(col, shape="box", color="lightblue")
            dot.edge(col, "Agente")
        elif role == "y":
            dot.node(col, shape="ellipse", color="lightgreen")
            dot.edge(col, "Rótulo")

    dot.node("Agente", shape="box", style="filled", color="lightgray")
    dot.node("Rótulo", shape="oval", style="filled", color="lightyellow")

    return dot


def show_feature_table(X: pd.DataFrame):
    """Mostra tabela interativa com Feature Role, estatísticas + gráfico normalizado."""
    st.subheader("Resumo do Dataset Selecionado")
    col1, col2 = st.columns([3, 2])
    with col2:
        st.info("Selecione o papel (Feature role) de cada coluna do dataset escolhido")

    feature_table = build_feature_table(X)

    edited = st.data_editor(
        feature_table,
        column_config={
            "Feature Role": st.column_config.SelectboxColumn(
                "Feature Role", options=["X", "y", "[Desativado]"], default="X"
            ),
            "gráfico": st.column_config.LineChartColumn("Distribuição Normalizada"),
        },
        hide_index=True,
        use_container_width=True,
        disabled=[
            "Coluna", "min", "mean", "median",
            "max", "std", "nulos", "nao_nulos", "dtype"
        ],
    )

    return edited

def show_feature_editor(X: pd.DataFrame):
    """Exibe editor de features com selectbox, estatísticas e gráfico normalizado."""
    st.subheader("📊 Resumo do dataset selecionado")

    st.write("➡️ Selecione o papel (**Feature role**) de cada coluna do dataset escolhido")

    edited = st.data_editor(
        build_feature_table(X),
        column_config={
            "Feature Role": st.column_config.SelectboxColumn(
                "Feature Role",
                help="Defina se a coluna é entrada (X), alvo (y) ou deve ser ignorada.",
                options=["X", "y", "[Desativado]"],
                default="X",
            ),
            "Coluna": st.column_config.TextColumn("Coluna", disabled=True),
            "Mínimo": st.column_config.NumberColumn("Mínimo", disabled=True),
            "Média": st.column_config.NumberColumn("Média", disabled=True),
            "Mediana": st.column_config.NumberColumn("Mediana", disabled=True),
            "Máximo": st.column_config.NumberColumn("Máximo", disabled=True),
            "Desvio Padrão": st.column_config.NumberColumn("Desvio Padrão", disabled=True),
            "Nulos": st.column_config.NumberColumn("Nulos", disabled=True),
            "Não-Nulos": st.column_config.NumberColumn("Não-Nulos", disabled=True),
            "Tipo": st.column_config.TextColumn("Tipo", disabled=True),
            "Gráfico": st.column_config.LineChartColumn("Gráfico", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
        num_rows="dynamic",
    )

    return edited

def show_log_page():
    st.subheader("📝 Log do Aplicativo")

    # Use st.session_state.files to get the log file path
    log_file_path = st.session_state.files.get('log')

    if not log_file_path or not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        st.info("O arquivo de log está vazio ou não existe.")
        return

    try:
        df_log = pd.read_csv(log_file_path, sep='\t', header=None, names=["Timestamp", "Level", "Message", "Traceback"])
        st.dataframe(df_log, width='stretch')
    except Exception as e:
        st.error(f"Erro ao ler o arquivo de log: {e}")
        return

    if st.button("Limpar Log", help=f"Limpa todo o conteúdo do arquivo {st.session_state.files.get('log')} recarrega a página"):
        try:
            with open(log_file_path, "w") as f:
                f.truncate(0)
            st.success("Log limpo com sucesso!")
            st.rerun() # Rerun to show empty log
        except Exception as e:
            st.error(f"Erro ao limpar o log: {e}")

def manage_files_df():
    st.header("Gerenciador de Arquivos do Projeto")
    # Get the path to files.tsv from session_state
    files_tsv_path = st.session_state.files.get('files')

    if files_tsv_path:
        try:
            # Read the files.tsv into a DataFrame
            df = pd.read_csv(files_tsv_path, sep='\t', engine='python')
            st.info("Edite os nomes ou caminhos dos arquivos. As alterações serão salvas automaticamente.")
            
            # Display the DataFrame as an editable data_editor
            edited_df = st.data_editor(
                df,
                num_rows="dynamic", # Allows adding/deleting rows
                use_container_width=True,
                key="files_data_editor"
            )
            
            # Check if the DataFrame has changed
            if not edited_df.equals(df):
                # Save the changes back to files.tsv
                edited_df.to_csv(files_tsv_path, sep='\t', index=False)
                st.success("Arquivo 'files.tsv' atualizado com sucesso!")
                # Update st.session_state.files immediately
                st.session_state.files = dict(zip(edited_df['file_name'], edited_df['file_path']))
                st.rerun() # Rerun to reflect changes immediately
        except FileNotFoundError:
            st.error(f"Arquivo de configuração 'files.tsv' não encontrado em: {files_tsv_path}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao carregar ou salvar o arquivo de arquivos: {e}")
            st.exception(e)
    else:
        st.warning("O caminho para 'files.tsv' não está definido em st.session_state.files. Por favor, verifique a configuração inicial.")
