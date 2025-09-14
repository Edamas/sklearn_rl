import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
from datetime import datetime
from functions import df_select_rows
import plotly.express as px
import plotly.graph_objects as go
import pydoc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.utils import estimator_html_repr
from functions import _append_to_erros_txt


def display_pipeline_graphically(pipeline_repr):
    """
    Reconstructs a scikit-learn pipeline object from a dictionary representation
    and displays its HTML diagram.
    """
    if not isinstance(pipeline_repr, dict):
        st.warning("Não é possível exibir o gráfico do pipeline: a representação não é válida.")
        return

    try:
        # 1. Reconstruct Preprocessor
        reconstructed_transformers = []
        for group_step in pipeline_repr.get('preprocessor', []):
            group_name = group_step.get('group')
            columns = group_step.get('columns', [])
            
            inner_steps = []
            for step_name, step_info in group_step.get('steps', []):
                class_path = step_info.get('class_path')
                params = step_info.get('params', {})
                
                estimator_class = pydoc.locate(class_path)
                if estimator_class:
                    inner_steps.append((step_name, estimator_class(**params)))
                else:
                    raise ImportError(f"Não foi possível localizar a classe: {class_path}")
            
            group_pipeline = Pipeline(inner_steps)
            reconstructed_transformers.append((group_name, group_pipeline, columns))
        
        reconstructed_preprocessor = ColumnTransformer(reconstructed_transformers, remainder='drop')

        # 2. Reconstruct Estimator
        estimator_info = pipeline_repr.get('estimator', {})
        est_class_path = estimator_info.get('class_path')
        est_params = estimator_info.get('params', {})
        
        estimator_class = pydoc.locate(est_class_path)
        if not estimator_class:
            raise ImportError(f"Não foi possível localizar a classe do estimador: {est_class_path}")
            
        reconstructed_estimator = estimator_class(**est_params)

        # 3. Assemble final pipeline
        final_pipeline = Pipeline([
            ('preprocessor', reconstructed_preprocessor),
            ('estimator', reconstructed_estimator)
        ])

        # 4. Display HTML representation
        st.markdown("###### 5.5.4 Diagrama do Pipeline")
        html_repr = estimator_html_repr(final_pipeline)
        components.html(html_repr, height=1000, scrolling=True)

    except Exception as e:
        error_message = f"Falha ao gerar o diagrama do pipeline: {e}"
        st.error(error_message)
        _append_to_erros_txt(error_message)
        error_message = f"Falha ao gerar o diagrama do pipeline: {e}"
        st.error(error_message)
        _append_to_erros_txt(error_message)
        _append_to_erros_txt(f"Exceção ao gerar o diagrama do pipeline: {e}")
        st.exception(e)

def render_scatterplot(df_data, selected_index=None):
    if 'score' in df_data.columns and 'duration_seconds' in df_data.columns:
        df_plot = df_data.copy()
        if 'timestamp' in df_plot.columns:
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])

        # --- New Ranking Logic ---
        df_plot = df_plot.sort_values(by=['score', 'duration_seconds'], ascending=[False, True])
        df_plot['ranking'] = range(1, len(df_plot) + 1)
        df_plot['original_index'] = df_plot.index # Preserve original index

        with st.expander("5.4.1 Configuração do Gráfico", expanded=True):
            # --- Plot Controls ---
            col1, col2, col3 = st.columns(3)

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
                st.markdown("###### 5.4.1.1 Eixo X")
                x_axis_selection = st.radio("Eixo X", option_keys, index=option_keys.index('Ranking'), key="x_axis_selection", label_visibility="collapsed")

            with col2:
                st.markdown("###### 5.4.1.2 Eixo Y")
                y_axis_selection = st.radio("Eixo Y", option_keys, index=option_keys.index('Score'), key="y_axis_selection", label_visibility="collapsed")

            with col3:
                st.markdown("###### 5.4.1.3 Legenda (Cor)")
                color_selection = st.radio("Legenda (Cor)", option_keys, index=option_keys.index('Estimador'), key="color_selection", label_visibility="collapsed")

        x_col = axis_options[x_axis_selection]
        y_col = axis_options[y_axis_selection]
        color_col = axis_options[color_selection]

        # --- Plotting ---
        plot_args = {
            "data_frame": df_plot,
            "x": x_col,
            "y": y_col,
            "color": color_col,
            "hover_name": "estimator_name",
            "hover_data": {c: True for c in df_plot.columns if c != 'original_index'},
            "title": f"{y_axis_selection} vs. {x_axis_selection} por {color_selection}"
        }

        # Set color scale
        if color_col == 'timestamp':
            # Use numeric values for a continuous color scale
            plot_args['color'] = df_plot['timestamp'].astype('int64')
            plot_args['color_continuous_scale'] = px.colors.sequential.Viridis
        elif pd.api.types.is_numeric_dtype(df_plot[color_col]):
            plot_args["color_continuous_scale"] = px.colors.sequential.Viridis
        
        if color_col == 'status':
            plot_args["color_discrete_map"] = {'Erro': 'red', 'Sucesso': 'blue'}
            
        fig = px.scatter(**plot_args)

        # Set color bar title for timestamp
        if color_col == 'timestamp':
            fig.update_layout(coloraxis_colorbar_title_text='Timestamp')

        if selected_index is not None:
            # Use original_index to find the selected row
            selected_row_df = df_plot[df_plot['original_index'] == selected_index]
            if not selected_row_df.empty:
                selected_row = selected_row_df.iloc[0]
                x_val = selected_row[x_col]
                y_val = selected_row[y_col]

                fig.add_trace(go.Scatter(x=[x_val], y=[y_val], mode='markers', marker=dict(symbol='star', color='lime', size=15, line=dict(color='black', width=2)), name='Selecionado', hoverinfo='none'))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o gráfico de desempenho geral (scores ou durações ausentes).")

def results():
    st.header("2. Configurações")

    if 'agent_results' not in st.session_state or not st.session_state['agent_results']:
        st.warning("Nenhum resultado de agente encontrado. Execute o agente primeiro.")
        st.info("Nenhum resultado de treinamento disponível. Por favor, inicie um treinamento na aba anterior.")
        st.stop()

    results_data = st.session_state['agent_results']
    dataset_name = results_data.get("name", "N/A")
    df_results = results_data.get("results_df")

    if df_results is None or df_results.empty:
        st.warning("O agente não produziu nenhum resultado para exibir.")
        st.info("Nenhum resultado de treinamento gerado para exibir.")
        st.stop()

    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**2.1 Nome do Dataset:** `{dataset_name}`")
    with col_info2:
        st.markdown(f"**2.2 Formato do Dataset:** `Pandas DataFrame`")

    st.markdown("### 2.3 Episódios de Treinamento")

    df_display = df_results.copy()
    df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
    
    columns_to_show = ['estimator_name', 'status', 'score', 'timestamp', 'duration_seconds', 'error', 'pipeline_steps']
    df_display = df_display[columns_to_show]

    df_display = df_display.rename(columns={
        'estimator_name': 'Estimador',
        'status': 'Status',
        'score': 'Score',
        'timestamp': 'Timestamp',
        'duration_seconds': 'Duração (s)',
        'error': 'Erro',
        'pipeline_steps': 'Pipeline'
    })

    # Convert object columns to string for display
    df_display['Pipeline'] = df_display['Pipeline'].astype(str)

    # Sort and Rank the dataframe
    df_display.sort_values(by=['Score', 'Duração (s)'], ascending=[False, True], inplace=True)
    df_display['Ranking'] = df_display.groupby(['Score', 'Duração (s)'], sort=False).ngroup() + 1
    
    # Reorder columns to show Ranking first
    cols = ['Ranking'] + [col for col in df_display.columns if col != 'Ranking']
    df_display = df_display[cols]

    selected_trial_index = df_select_rows(df_display, selection_mode='single-row', prompt="Selecione um episódio na tabela para ver os detalhes.", key="results_dataframe_selection")

    st.markdown("### 2.4 Desempenho Geral dos Modelos")
    render_scatterplot(df_results, selected_trial_index)

    st.markdown("### 2.5 Detalhes do Episódio Selecionado")

    if selected_trial_index is not None:
        selected_trial = df_results.loc[selected_trial_index]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 5.5.1 Métricas")
            st.metric(label="Status", value=selected_trial.get('status', 'N/A'))
            score_val = selected_trial.get('score', 0)
            st.metric(label="Score", value=f"{score_val:.4f}")
            timestamp_val = pd.to_datetime(selected_trial.get('timestamp', 'N/A'))
            if pd.notna(timestamp_val):
                st.metric(label="Timestamp", value=timestamp_val.strftime('%d/%m/%Y %H:%M:%S'))
            duration = selected_trial.get('duration_seconds', 0)
            st.metric(label="Duração", value=f"{duration:.4f} s")

        pipeline_repr = selected_trial.get('pipeline_steps')
        
        if isinstance(pipeline_repr, dict):
            with col2:
                display_pipeline_graphically(pipeline_repr)
        else:
            with col2:
                st.warning("Formato de pipeline desconhecido ou não disponível.")
                st.text(str(pipeline_repr))

    else:
        st.info("Selecione um episódio na tabela acima para ver seus detalhes.")

    st.markdown("### 2.6 Download dos Resultados")
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar resultados em CSV",
        data=csv,
        file_name=f'desempenho_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )
