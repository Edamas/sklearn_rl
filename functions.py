import streamlit as st
import pandas as pd
import graphviz as gv
from typing import Optional, Literal
from pathlib import Path
import io # Mover importação para o topo

# importações não usadas:
import numpy as np
import re
import os

def get_card_style():
    return """
    <style>
        .card {
            border: 1px solid #e6e6e6;
            border-radius: 0.25rem;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
        }
        .card-title {
            font-weight: bold;
        }
    </style>
    """

def df_select_rows(df, selection_mode: Literal['single-row', 'multi-row'] = 'multi-row', prompt: Optional[str] = None, key: str = "dataframe_selection"):
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

def analyze_and_group_columns(df: pd.DataFrame):
    """
    Analyzes a DataFrame, groups columns by data type, and calculates summary statistics for each group.

    Returns:
        A DataFrame summarizing the column groups.
    """
    # Attempt to convert object columns to datetime
    df_temp = df.copy()
    for col in df_temp.select_dtypes(include=['object']).columns:
        try:
            df_temp[col] = pd.to_datetime(df_temp[col], errors='coerce')
        except (ValueError, TypeError):
            continue # Ignore columns that can't be converted

    groups = {
        'Numeric': [],
        'Binary': [],
        'Categorical': [],
        'Date': [],
        'Text': [] # For high cardinality strings
    }

    for col in df_temp.columns:
        dtype = df_temp[col].dtype
        nunique = df_temp[col].nunique()

        if pd.api.types.is_numeric_dtype(dtype):
            if nunique == 2:
                groups['Binary'].append(col)
            else:
                groups['Numeric'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            groups['Date'].append(col)
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            # Heuristic for categorical vs. text
            if nunique < 50 and nunique > 0:
                groups['Categorical'].append(col)
            else: 
                groups['Text'].append(col)

    summary_list = []
    for group_name, cols in groups.items():
        if not cols:
            continue

        n_cols = len(cols)
        subset = df_temp[cols]
        n_rows = len(subset)
        nulls = subset.isnull().sum().sum()
        pct_nulls = nulls / (n_rows * n_cols) if (n_rows * n_cols) > 0 else 0

        stats = {
            'group_type': group_name,
            'column_count': n_cols,
            'columns': cols,
            'null_percentage': pct_nulls
        }

        if group_name == 'Numeric':
            stats['mean_of_means'] = subset.mean().mean()
            stats['mean_of_stds'] = subset.std().mean()
            stats['min_of_mins'] = subset.min().min()
            stats['max_of_maxs'] = subset.max().max()
        elif group_name == 'Binary':
            if n_cols > 0:
                counts = subset[cols[0]].value_counts(normalize=True)
                stats['value1_proportion'] = counts.iloc[0] if len(counts) > 0 else 0.0
                stats['value2_proportion'] = counts.iloc[1] if len(counts) > 1 else 0.0
        elif group_name == 'Categorical':
            stats['mean_unique_values'] = subset.nunique().mean()
        elif group_name == 'Text':
            stats['mean_unique_values'] = subset.nunique().mean()
            stats['mean_string_length'] = subset.astype(str).apply(lambda x: x.str.len()).mean().mean()
        elif group_name == 'Date':
            min_date = subset.min().min() if not subset.min().empty else pd.NaT
            max_date = subset.max().max() if not subset.max().empty else pd.NaT
            stats['min_date'] = min_date
            stats['max_date'] = max_date

        summary_list.append(stats)

    if not summary_list:
        return pd.DataFrame()

    summary_df = pd.DataFrame(summary_list).set_index('group_type')
    return summary_df.fillna(0)

def build_feature_table(X: pd.DataFrame):
    """Cria tabela de resumo das features, com roles e estatísticas básicas."""
    desc = X.describe(include="all").transpose()
    nulls = X.isnull().sum()
    non_nulls = X.notnull().sum()
    medians = X.median(numeric_only=True)
    dtypes = X.dtypes.astype(str)
    normalized = (X - X.min()) / (X.max() - X.min())
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
    """
    Desenha grafo com Graphviz baseado nos papéis das features."""
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
    """
    Mostra tabela interativa com Feature Role, estatísticas + gráfico normalizado.
    """
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
        width='stretch',
        disabled=[
            "Coluna", "min", "mean", "median",
            "max", "std", "nulos", "nao_nulos", "dtype"
        ],
    )
    return edited

def show_feature_editor(X: pd.DataFrame):
    """
    Exibe editor de features com selectbox, estatísticas e gráfico normalizado.
    """
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
        width='stretch',
        num_rows="dynamic",
    )
    return edited


def create_optimized_estimators_tsv(original_tsv_path: str, output_tsv_path: str):
    """
    Reads the original estimators.tsv, selects only the actively used columns,
    and saves them to a new optimized TSV file.
    """
    st.toast(f"Criando arquivo otimizado de estimadores: {output_tsv_path}")

    actively_used_columns = [
        'estimator_name', 'class_path', 'estimator_type', 'input_X_structure',
        'input_X_types', 'input_y_structure', 'input_y_types', 'output_X_structure',
        'output_X_types', 'output_y_structure', 'output_y_types', 'compatible_scores'
    ]

    try:
        # Read the original TSV file
        df_original = pd.read_csv(original_tsv_path, sep='\t', low_memory=False)

        # Select only the actively used columns
        df_optimized = df_original[actively_used_columns]

        # Ensure the output directory exists
        output_dir = Path(output_tsv_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the optimized DataFrame to a new TSV file
        df_optimized.to_csv(output_tsv_path, sep='\t', index=False)
        st.toast(f"Arquivo otimizado de estimadores criado com sucesso em: {output_tsv_path}")
        return True
    except FileNotFoundError:
        st.error(f"Erro: O arquivo original de estimadores '{original_tsv_path}' não foi encontrado.")
        return False
    except KeyError as e:
        st.error(f"Erro: Coluna essencial faltando no arquivo original de estimadores: {e}. Verifique se '{original_tsv_path}' está completo.")
        return False
    except Exception as e:
        st.error(f"Ocorreu um erro ao criar o arquivo otimizado de estimadores: {e}")
    import io

def get_rubricas_by_function_score_8(function_name: str) -> pd.DataFrame:
    """
    Retorna as rubricas com nota 8 para uma função específica como um DataFrame.

    Args:
        function_name (str): O nome da função (e.g., "Gestão do TCC", "Documentação de Software").

    Returns:
        pd.DataFrame: Um DataFrame contendo as rubricas filtradas. Retorna um DataFrame vazio em caso de erro ou se nenhuma rubrica for encontrada.
    """
    rubricas_file_path = st.session_state['files']['rubricas']

    try:
        df = pd.read_csv(rubricas_file_path, sep='\t', encoding='utf-8', engine='python')
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de rubricas '{rubricas_file_path}' não foi encontrado.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao ler o arquivo de rubricas: {e}")
        return pd.DataFrame()

    # Renomear as colunas do DataFrame para facilitar o acesso e corresponder ao TSV
    df.rename(columns={
        'Entrega': 'entrega',
        'Subitem': 'subitem',
        'Competência': 'competencia',
        'Item_rubrica': 'item_rubrica',
        'Rubrica de Avaliação': 'rubrica',
        'Aplicação no projeto': 'aplicacao_no_projeto'
    }, inplace=True)


    # Filtrar as linhas onde a coluna da função tem valor '8'
    if function_name not in df.columns:
        st.error(f"Erro: A função '{function_name}' não foi encontrada nas rubricas.")
        return pd.DataFrame()

    filtered_df = df[df[function_name].astype(str) == '8'].copy() # Adicionar .copy() para evitar SettingWithCopyWarning

    return filtered_df

def show_data_by_function(function_name, data_type):
    config = {
        'cronograma': {
            'file_key': 'cronograma',
            'date_col': None,
            'sort_by': None,
            'ascending': True,
            'display_cols': ['Quinzena', 'Início', 'Fim', 'Tarefa', 'Responsável'],
            'rename_cols': {'Quinzena': 'Quinzena', 'Início': 'Início', 'Fim': 'Fim', 'Tarefa': 'Tarefa', 'Responsável': 'Responsável'},
            'details_title': 'Ficha da Tarefa',
            'details_fields': {
                'Quinzena': 'Quinzena',
                'Período': lambda r: f"{r['Início']} a {r['Fim']}",
                'Função': 'Função',
                'Tarefa': 'Tarefa',
                'Responsável': lambda r: r.get('Responsável', 'A definir')
            }
        },
        'registro_de_atividades': {
            'file_key': 'registro_de_atividades',
            'date_col': 'Data',
            'sort_by': 'Data',
            'ascending': False,
            'display_cols': ['Data', 'Evento', 'Responsável'],
            'rename_cols': {'Data': 'Data', 'Evento': 'Atividade', 'Responsável': 'Responsável'},
            'details_title': 'Ficha da Atividade',
            'details_fields': {
                'Data': lambda r: r['Data'].strftime('%d/%m/%Y %H:%M'),
                'Canal': 'Canal',
                'Evento': 'Evento',
                'Responsável': 'Responsável',
                'Observações': 'Observações'
            }
        }
    }

    cfg = config.get(data_type)
    if not cfg:
        st.error(f"Tipo de dado '{data_type}' não configurado.")
        return

    file_path = st.session_state['files'][cfg['file_key']]
    try:
        df = pd.read_csv(file_path, sep='\t', engine='python')
        if cfg['date_col']:
            df[cfg['date_col']] = pd.to_datetime(df[cfg['date_col']], format='%d/%m/%Y %H:%M', errors='coerce')
            df.dropna(subset=[cfg['date_col']], inplace=True)
        
        if cfg['sort_by']:
            df = df.sort_values(by=cfg['sort_by'], ascending=cfg['ascending'])

        df_filtered = df[df['Função'] == function_name].copy() if function_name else df.copy()

        if not df_filtered.empty:
            df_display = df_filtered[cfg['display_cols']].copy()
            if cfg['date_col'] and not df_display.empty:
                df_display[cfg['date_col']] = df_display[cfg['date_col']].dt.strftime('%d/%m/%Y %H:%M')
            df_display.rename(columns=cfg['rename_cols'], inplace=True)

            selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"{data_type}_{function_name.replace(' ', '_') if function_name else 'all'}_selector")

            if selected_index is not None and selected_index in df_filtered.index:
                selected_item = df_filtered.loc[selected_index]
                st.subheader(cfg['details_title'])
                
                with st.container(border=True):
                    for label, field in cfg['details_fields'].items():
                        value = None
                        if callable(field):
                            value = field(selected_item)
                        else:
                            value = selected_item.get(field)
                        
                        if pd.notna(value) and value != '':
                            # Usando um esquema de cores simples para diferenciar os campos
                            color = "#FFD700" # Dourado para o primeiro campo
                            if label == 'Período' or label == 'Canal': color = '#ADD8E6' # Azul claro
                            elif label == 'Função' or label == 'Evento': color = '#90EE90' # Verde claro
                            elif label == 'Tarefa' or label == 'Responsável': color = '#FF6347' # Tomate
                            else: color = '#4682B4' # SteelBlue para os demais
                            
                            st.markdown(f"**<font color='{color}'>{label}:</font>** {value}", unsafe_allow_html=True)

        else:
            st.info(f"Nenhum dado encontrado para a função '{function_name}' em {data_type.replace('_', ' ')}.")
    except FileNotFoundError:
        st.error(f"Arquivo de {data_type.replace('_', ' ')} não encontrado em: {file_path}")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar ou processar {data_type.replace('_', ' ')}: {e}")

def show_cronograma_by_function(function_name):
    show_data_by_function(function_name, 'cronograma')

def show_registro_atividades_by_function(function_name):
    show_data_by_function(function_name, 'registro_de_atividades')

def show_disciplinas_relacionadas_vri(function_name: Optional[str] = None):
    disciplinas_path = st.session_state['files']['disciplinas_relacionadas']
    try:
        df_disciplinas = pd.read_csv(disciplinas_path, sep='\t')
        df_disciplinas.columns = df_disciplinas.columns.str.strip().str.replace(':', '')

        df_filtered_disciplinas = df_disciplinas.copy()
        if function_name:
            df_filtered_disciplinas = df_filtered_disciplinas[
                df_filtered_disciplinas['Funções Relacionadas'].astype(str).str.contains(function_name, na=False)
            ].copy()

        df_filtered_disciplinas = df_filtered_disciplinas.sort_values(by='Relação com Projeto', ascending=False)

        if not df_filtered_disciplinas.empty:
            df_display = df_filtered_disciplinas[['Bimestre', 'Disciplina', 'Relação com Projeto', 'Funções Relacionadas']].copy()

            selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"disciplinas_relacionadas_{function_name.replace(' ', '_') if function_name else 'all'}_selector")

            if selected_index is not None and selected_index in df_filtered_disciplinas.index:
                selected_discipline = df_filtered_disciplinas.loc[selected_index]
                st.subheader("Ficha da Disciplina")
                
                with st.container(border=True):
                    st.markdown(f"**<font color='#FFD700'>Bimestre:</font>** {selected_discipline['Bimestre']}", unsafe_allow_html=True)
                    st.markdown(f"**<font color='#ADD8E6'>Disciplina:</font>** {selected_discipline['Disciplina']}", unsafe_allow_html=True)
                    st.markdown(f"**<font color='#90EE90'>Relação com Projeto:</font>** {selected_discipline['Relação com Projeto']}", unsafe_allow_html=True)
                    st.markdown(f"**<font color='#FF6347'>Objetivo:</font>** {selected_discipline['Objetivo']}", unsafe_allow_html=True)
                    st.markdown(f"**<font color='#4682B4'>Ementa:</font>** {selected_discipline['Ementa']}", unsafe_allow_html=True)
                    st.markdown(f"**<font color='#FFD700'>Conteúdo Programático:</font>** {selected_discipline['Conteúdo programático']}", unsafe_allow_html=True)
                    st.markdown(f"**<font color='#FF6347'>Funções Relacionadas:</font>** {selected_discipline['Funções Relacionadas']}", unsafe_allow_html=True)
            else:
                pass # Removido: st.info("Nenhuma disciplina selecionada ou nenhuma disciplina relacionada à função atual.")
        else:
            st.info(f"Nenhuma disciplina encontrada para a função '{function_name}' nas disciplinas relacionadas.")
    except FileNotFoundError:
        st.error(f"Arquivo de disciplinas relacionadas não encontrado em: {disciplinas_path}")
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar ou processar as disciplinas relacionadas: {e}")

def show_referencias_by_function(function_name: str):
    referencias_path = st.session_state['files']['referencias']
    try:
        df_referencias = pd.read_csv(referencias_path, sep='\t')
        df_filtered = df_referencias[
            df_referencias['função_relacionada'].astype(str).str.contains(function_name, na=False)
        ].copy()

        if not df_filtered.empty:
            st.header("Referências e Fontes")

            df_display = df_filtered[['referência', 'utilizado_em']].copy()
            df_display.rename(columns={'referência': 'Referência', 'utilizado_em': 'Utilizado em'}, inplace=True)

            selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"referencias_{function_name.replace(' ', '_')}_selector")

            if selected_index is not None and selected_index in df_filtered.index:
                selected_ref = df_filtered.loc[selected_index]
                st.subheader("Ficha da Referência")
                with st.container(border=True):
                    st.markdown(f"**<font color='#FFD700'>Referência:</font>** {selected_ref['referência']}", unsafe_allow_html=True)
                    if pd.notna(selected_ref['função_relacionada']):
                        st.markdown(f"**<font color='#ADD8E6'>Função Relacionada:</font>** {selected_ref['função_relacionada']}", unsafe_allow_html=True)
                    if pd.notna(selected_ref['utilizado_em']):
                        st.markdown(f"**<font color='#90EE90'>Utilizado em:</font>** {selected_ref['utilizado_em']}", unsafe_allow_html=True)
            else:
                pass # Removido: st.info("Nenhuma referência selecionada.")
        else:
            st.header("Referências e Fontes")
            st.info(f"Nenhuma referência encontrada para a função '{function_name}'.")

    except FileNotFoundError:
        st.header("Referências e Fontes")
        st.error(f"Arquivo de referências não encontrado em: {referencias_path}")
    except Exception as e:
        st.header("Referências e Fontes")
        st.error(f"Ocorreu um erro ao carregar ou processar as referências: {e}")
