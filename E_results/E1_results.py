import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydoc
import ast
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import estimator_html_repr

from functions import df_select_rows

from datetime import datetime  # não está sendo usado

# ------------------------------------------------------------------
# Gráfico de Dispersão de Resultados
# ------------------------------------------------------------------

def render_scatterplot(df_data, x_col, y_col, color_col, selected_orig_idx=None):
    """
    Renderiza um gráfico de dispersão interativo dos resultados dos episódios.
    Recebe as seleções de eixos e cor como argumentos.
    """
    if 'score' not in df_data.columns or 'duration_seconds' not in df_data.columns:
        st.info("Dados insuficientes para gerar o gráfico de desempenho (scores ou durações ausentes).")
        return

    df_plot = df_data.copy()
    if 'timestamp' in df_plot.columns:
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')

    # Adiciona uma coluna de ranking baseada no score e na duração
    df_plot = df_plot.sort_values(by=['score', 'duration_seconds'], ascending=[False, True])
    df_plot['ranking'] = range(1, len(df_plot) + 1)
    df_plot['orig_idx'] = df_plot.index

    plot_args = {
        "data_frame": df_plot,
        "x": x_col,
        "y": y_col,
        "color": color_col,
        "hover_name": "estimator_name",
        "hover_data": {c: True for c in df_plot.columns if c not in ['orig_idx', 'fitted_pipeline_obj', 'pipeline_steps', 'training_processed_df']},
        "title": f"{y_col.replace('_', ' ').title()} vs. {x_col.replace('_', ' ').title()} por {color_col.replace('_', ' ').title()}"
    }

    # Mapeamento de cores para status e escalas de cores para valores numéricos
    if color_col == 'status':
        plot_args["color_discrete_map"] = {'Erro': 'red', 'Sucesso': 'blue'}
    elif pd.api.types.is_numeric_dtype(df_plot.get(color_col)):
        plot_args["color_continuous_scale"] = px.colors.sequential.Viridis

    fig = px.scatter(**plot_args)

    # Adiciona um marcador especial para o ponto selecionado na tabela
    if selected_orig_idx is not None:
        marker_row = df_plot[df_plot['orig_idx'] == selected_orig_idx]
        if not marker_row.empty:
            sel = marker_row.iloc[0]
            fig.add_trace(
                go.Scatter(
                    x=[sel[x_col]],
                    y=[sel[y_col]],
                    mode='markers',
                    marker=dict(
                        symbol='star',
                        color='lime',
                        size=15,
                        line=dict(
                            color='black',
                            width=2
                        )
                    ),
                    name='Selecionado',
                    hoverinfo='none'
                )
            )

    st.plotly_chart(fig, width='stretch')


# ------------------------------------------------------------------
# Helpers: reconstrução de pipeline a partir da representação salva
# ------------------------------------------------------------------
def _reconstruct_pipeline_object(pipeline_repr):
    """
    Reconstructs a scikit-learn pipeline object from a dictionary representation.
    """
    if not isinstance(pipeline_repr, dict):
        raise ValueError("Representação do pipeline inválida: não é um dicionário.")

    # 1. Reconstruct Preprocessor
    reconstructed_transformers = []
    for group_step in pipeline_repr.get('preprocessor', []):
        group_name = group_step.get('group')
        columns = group_step.get('columns', []) or []
        
        inner_steps = []
        for step_name, step_info in group_step.get('steps', []):
            class_path = step_info.get('class_path')
            params = step_info.get('params', {}) or {}
            
            estimator_class = pydoc.locate(class_path)
            if estimator_class:
                try:
                    inner_steps.append((step_name, estimator_class(**params)))
                except Exception as e:
                    raise RuntimeError(f"Falha ao instanciar {class_path} com params {params}: {e}")
            else:
                raise ImportError(f"Não foi possível localizar a classe: {class_path}")
        
        if inner_steps:
            group_pipeline = Pipeline(inner_steps)
            reconstructed_transformers.append((group_name, group_pipeline, columns))
        else:
            # Se não há steps instanciáveis, usar passthrough para esse grupo
            reconstructed_transformers.append((group_name, "passthrough", columns))
    
    reconstructed_preprocessor = ColumnTransformer(reconstructed_transformers, remainder='drop')

    # 2. Reconstruct Estimator
    estimator_info = pipeline_repr.get('estimator', {}) or {}
    est_class_path = estimator_info.get('class_path')
    est_params = estimator_info.get('params', {}) or {}
    
    estimator_class = pydoc.locate(est_class_path)
    if not estimator_class:
        raise ImportError(f"Não foi possível localizar a classe do estimador: {est_class_path}")
        
    reconstructed_estimator = estimator_class(**est_params)

    # 3. Assemble final pipeline
    final_pipeline = Pipeline([
        ('preprocessor', reconstructed_preprocessor),
        ('estimator', reconstructed_estimator)
    ])
    return final_pipeline

def display_pipeline_graphically(pipeline_repr):
    """
    Reconstructs a scikit-learn pipeline object from a dictionary representation
    and displays its HTML diagram.
    """
    if not isinstance(pipeline_repr, dict):
        st.warning("Não é possível exibir o gráfico do pipeline: a representação não é válida.")
        return

    try:
        final_pipeline = _reconstruct_pipeline_object(pipeline_repr)
        st.markdown("###### 5.5.4 Diagrama do Pipeline")
        html_repr = estimator_html_repr(final_pipeline)
        components.html(html_repr, height=500, scrolling=True)  # height=1000,

    except Exception as e:
        error_message = f"Falha ao gerar o diagrama do pipeline: {e}"
        

# ------------------------------------------------------------------
# Main results view
# ------------------------------------------------------------------
def results():
    st.header("2. Simulações")

    if 'agent_results' not in st.session_state or not st.session_state['agent_results']:
        st.warning("Nenhum resultado de agente encontrado. Execute o agente primeiro.")
        st.stop()

    results_data = st.session_state['agent_results']
    dataset_name = results_data.get("name", "N/A")
    df_results = results_data.get("results_df")
    all_predictions_df = results_data.get("all_predictions_df")
    X_cols_transformed = results_data.get("X_cols_transformed")
    y_cols_transformed = results_data.get("y_cols_transformed")

    if df_results is None or df_results.empty:
        st.warning("O agente não produziu nenhum resultado para exibir.")
        st.stop()

    if all_predictions_df is None or X_cols_transformed is None or y_cols_transformed is None:
        st.error("Dados de previsão completos (all_predictions_df) ou informações de colunas estão faltando. O treinamento pode não ter sido concluído corretamente.")
        st.stop()

    # Info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**2.1 Nome do Dataset:** `{dataset_name}`")
    with col_info2:
        st.markdown(f"**2.2 Formato do Dataset:** `Pandas DataFrame`")

    st.markdown("### 2.3 Episódios de Treinamento")

    # Prepare dataframe for display — keep original index as 'orig_idx'
    df_display = df_results.copy()
    # Ensure timestamp column exists and format safely
    if 'timestamp' in df_display.columns:
        df_display['timestamp'] = pd.to_datetime(df_display['timestamp'], errors='coerce').dt.strftime('%d/%m/%Y %H:%M:%S')

    # Select columns to show (keep pipeline_steps stringified)
    columns_to_show = ['estimator_name', 'status', 'score', 'timestamp', 'duration_seconds', 'error', 'pipeline_steps']
    # If some columns missing, adapt
    cols_present = [c for c in columns_to_show if c in df_display.columns]
    df_display = df_display[cols_present]

    df_display = df_display.rename(columns={
        'estimator_name': 'Estimador',
        'status': 'Status',
        'score': 'Score',
        'timestamp': 'Timestamp',
        'duration_seconds': 'Duração (s)',
        'error': 'Erro',
        'pipeline_steps': 'Pipeline'
    })

    # Preserve original index to map selection back to df_results
    df_display['orig_idx'] = df_results.index

    # Convert object columns to string for display (Pipeline)
    if 'Pipeline' in df_display.columns:
        df_display['Pipeline'] = df_display['Pipeline'].apply(lambda x: str(x) if not pd.isna(x) else "")

    # Sort and rank
    sort_cols = []
    if 'Score' in df_display.columns:
        sort_cols.append('Score')
    if 'Duração (s)' in df_display.columns:
        sort_cols.append('Duração (s)')
    if sort_cols:
        ascending = [False if c == 'Score' else True for c in sort_cols]
        df_display.sort_values(by=sort_cols, ascending=ascending, inplace=True)

    # Keep original index, add Ranking as a column
    df_display['Ranking'] = range(1, len(df_display) + 1) # Simple sequential ranking
    # Ensure 'Ranking' is the first column for display
    df_display = df_display[['Ranking'] + [col for col in df_display.columns if col != 'Ranking']]

    # Use the provided df_select_rows widget to select a single row; it should return the orig_idx
    selected_row_return = df_select_rows(df_display, selection_mode='single-row',
                                        prompt="Selecione um episódio na tabela para ver os detalhes.",
                                        key="results_dataframe_selection")

    # Map selection to original index in df_results
    selected_trial_index = None
    if selected_row_return is not None:
        # df_select_rows now returns the original index directly
        selected_trial_index = selected_row_return

    st.markdown("### 2.4 Desempenho Geral dos Modelos")

    # Move widget configuration outside the cached function
    with st.expander("Configuração do Gráfico de Desempenho", expanded=True):
        col1, col2, col3, col4 = st.columns([1, 1, 1, 2]) # Added col4 for the plot

        axis_options = {
            'Estimador': 'estimator_name',
            'Status': 'status',
            'Score': 'score',
            'Timestamp': 'timestamp',
            'Duração (s)': 'duration_seconds',
            'Ranking': 'ranking'
        }
        option_keys = list(axis_options.keys())

        with col1:
            x_axis_selection = st.radio("Eixo X", option_keys, index=option_keys.index('Ranking'), key="x_axis_selection_main")
        with col2:
            y_axis_selection = st.radio("Eixo Y", option_keys, index=option_keys.index('Score'), key="y_axis_selection_main")
        with col3:
            color_selection = st.radio("Legenda (Cor)", option_keys, index=option_keys.index('Estimador'), key="color_selection_main")

        x_col = axis_options[x_axis_selection]
        y_col = axis_options[y_axis_selection]
        color_col = axis_options[color_selection]
        
        # Plot the scatterplot outside the expander to avoid unnecessary re-renders
    render_scatterplot(df_results, x_col, y_col, color_col, selected_trial_index)

    st.markdown("### 2.5 Detalhes do Episódio Selecionado")

    if selected_trial_index is not None and selected_trial_index in df_results.index:
        selected_trial = df_results.loc[selected_trial_index]
        st.toast("Trial selecionado: " + str(selected_trial.get('estimator_name', 'N/A')))
        



        pipeline_repr = selected_trial.get('pipeline_steps')
        # pipeline_obj_saved = selected_trial.get('fitted_pipeline_obj')  # No longer needed for re-fitting
        # target_column = selected_trial['target_column'] # No longer needed directly from trial
        episode_index = selected_trial['episode_index']

        st.markdown("### 2.5.1 Previsões Detalhadas do Pipeline Selecionado")

        # Construct results_df for display
        cols_to_display = X_cols_transformed.copy()
        if y_cols_transformed and len(y_cols_transformed) == 1:
            target_col_name = y_cols_transformed[0]
            cols_to_display.append(target_col_name)
        else:
            target_col_name = None

        prediction_col_name = str(episode_index)
        cols_to_display.append(prediction_col_name)

        # Ensure all columns exist in all_predictions_df before selecting
        missing_cols = [col for col in cols_to_display if col not in all_predictions_df.columns]
        if missing_cols:
            st.error(f"Colunas essenciais para exibição de previsões detalhadas estão faltando no DataFrame de previsões: {missing_cols}")
            return

        results_df = all_predictions_df[cols_to_display].copy()

        # Calculate error column dynamically
        if target_col_name and prediction_col_name in results_df.columns:
            results_df['error'] = results_df[prediction_col_name] - results_df[target_col_name]
        else:
            results_df['error'] = np.nan

        st.dataframe(results_df, width='stretch')

        # Display pipeline (graphical)
        if isinstance(pipeline_repr, dict):
            display_pipeline_graphically(pipeline_repr)
        else:
            st.warning("Formato de pipeline desconhecido ou não disponível.")
            st.text(str(pipeline_repr))

        # --- New: Display predictions and errors ---
        st.markdown("### Gráficos de Resultados")

        # Opções de ordenação para o gráfico de predição
        order_option = st.radio(
            "Ordenar registros do gráfico por:",
            ("Ordem Normal", "Por Target", "Por Predição", "Por Erro"),
            index=0, # Padrão: Ordem Normal
            horizontal=True, # Adicionado para layout horizontal
            key="prediction_chart_order"
        )
        # Adicionar gráfico de predição e erro
        st.markdown("##### Gráfico de Predição e Erro")
        if prediction_col_name in results_df.columns and target_col_name:
            if target_col_name in results_df.columns:
                # Criar uma cópia para o gráfico e garantir o índice original
                df_plot_preds = results_df[[target_col_name, prediction_col_name, 'error']].copy()
                df_plot_preds.index.name = 'Original_Index' # Renomear para evitar conflito
                df_plot_preds = df_plot_preds.reset_index()

                # Aplicar ordenação com base na seleção do rádio
                if order_option == "Por Target":
                    df_plot_preds.sort_values(by=target_col_name, ascending=True, inplace=True)
                elif order_option == "Por Predição":
                    df_plot_preds.sort_values(by=prediction_col_name, ascending=True, inplace=True)
                elif order_option == "Por Erro":
                    df_plot_preds.sort_values(by='error', ascending=True, inplace=True)

                # O eixo X do gráfico será o índice atual do DataFrame ordenado
                df_plot_preds['Plot_Index'] = range(len(df_plot_preds))

                fig_preds = px.scatter(df_plot_preds, x='Plot_Index', y=[target_col_name, prediction_col_name],
                                    title=f"Target vs Predição para {target_col_name}")
                # Definir estilos específicos para cada trace
                fig_preds.for_each_trace(
                    lambda trace: trace.update(mode='markers', marker=dict(symbol='circle-open', size=3, opacity=0.5, color='red')) if trace.name == target_col_name
                    else trace.update(mode='markers', marker=dict(symbol='circle', opacity=0.5, size=2, color='blue'))
                )
                fig_preds.update_layout(hovermode="x unified")
                st.plotly_chart(fig_preds, width='stretch')
            else:
                st.warning(f"Coluna target '{target_col_name}' não encontrada no DataFrame processado para o gráfico.")
        else:
            st.info("Não foi possível gerar o gráfico de predição e erro (coluna de previsão ou target ausente/múltiplo).")

        # Allow download
        csv_bytes = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV do dataframe usado",
            data=csv_bytes,
            file_name=f"processed_df_episode_{episode_index}.csv",
            mime="text/csv"
        )

        # Optionally, store the processed_df back into session_state or selected_trial for future
        st.session_state.setdefault('last_selected_processed_df', results_df)
        st.session_state.setdefault('last_selected_processed_df_index', episode_index)

