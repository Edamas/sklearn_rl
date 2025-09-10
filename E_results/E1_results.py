import streamlit as st
import pandas as pd
from datetime import datetime
from functions import log_message, df_select_rows
from E_results.E2_graphs import show_pipeline_graph
import plotly.express as px
import plotly.graph_objects as go

def render_scatterplot(df_data, selected_index=None):
    if 'score' in df_data.columns and 'duration_seconds' in df_data.columns:
        df_plot = df_data.copy()
        if 'timestamp' in df_plot.columns:
            df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])

        # --- User selections for plot ---
        st.markdown("##### Controles do Gráfico")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("###### Tipo de Gráfico")
            plot_type = st.radio(
                "Tipo de Gráfico",
                ("Gráfico de Dispersão", "Gráfico de Linhas"),
                index=0,
                key="plot_type_selection",
                label_visibility="collapsed"
            )

        axis_options = {
            'Estimador': 'estimator_name',
            'Status': 'status',
            'Score': 'score',
            'Timestamp': 'timestamp',
            'Duração (s)': 'duration_seconds'
        }
        option_keys = list(axis_options.keys())

        with col2:
            st.markdown("###### Eixo X")
            x_axis_selection = st.radio("Eixo X", option_keys, index=option_keys.index('Estimador'), key="x_axis_selection", label_visibility="collapsed")

        with col3:
            st.markdown("###### Eixo Y")
            y_axis_selection = st.radio("Eixo Y", option_keys, index=option_keys.index('Score'), key="y_axis_selection", label_visibility="collapsed")

        # Disable color selection for line chart as it's always by estimator
        is_line_chart = plot_type == "Gráfico de Linhas"
        with col4:
            st.markdown("###### Legenda (Cor)")
            color_selection = st.radio(
                "Legenda (Cor)",
                option_keys,
                index=option_keys.index('Status'),
                key="color_selection",
                label_visibility="collapsed",
                disabled=is_line_chart,
                help="A legenda é sempre por Estimador no Gráfico de Linhas." if is_line_chart else ""
            )

        # Get actual column names from selections
        x_col = axis_options[x_axis_selection]
        y_col = axis_options[y_axis_selection]
        
        # --- Plotting ---
        color_discrete_map = {}
        category_orders = {}
        
        if is_line_chart:
            color_col = 'estimator_name'
            # Order estimators by mean score for a cleaner legend/plot
            if not df_plot.empty:
                mean_scores = df_plot.groupby('estimator_name')['score'].mean().sort_values(ascending=False)
                category_orders['estimator_name'] = mean_scores.index.tolist()
            title = f"{y_axis_selection} vs. {x_axis_selection} por Estimador"
        else: # Scatter plot
            color_col = axis_options[color_selection]
            if color_col == 'status':
                color_discrete_map = {'Erro': 'red', 'Sucesso': 'blue'}
            title = f"{y_axis_selection} vs. {x_axis_selection} por {color_selection}"

        plot_args = {
            "data_frame": df_plot,
            "x": x_col,
            "y": y_col,
            "color": color_col,
            "hover_name": "estimator_name",
            "hover_data": {"params": True, "status": True, "score": True, "duration_seconds": True, "error": True},
            "title": title,
            "color_discrete_map": color_discrete_map,
            "category_orders": category_orders
        }

        if is_line_chart:
            fig = px.line(**plot_args)
        else:
            plot_args["opacity"] = 0.7
            fig = px.scatter(**plot_args)

        # Add a standout marker for the selected point
        if selected_index is not None and selected_index in df_plot.index:
            selected_row = df_plot.loc[selected_index]
            fig.add_trace(go.Scatter(
                x=[selected_row[x_col]],
                y=[selected_row[y_col]],
                mode='markers',
                marker=dict(
                    symbol='star',
                    color='lime',
                    size=15,
                    line=dict(color='black', width=2)
                ),
                name='Selecionado',
                hoverinfo='none'
            ))

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o gráfico de desempenho geral (scores ou durações ausentes).")


def results():
    
    st.header("5. Resultados") # Changed from "Desempenho"

    if 'agent_results' not in st.session_state:
        log_message("WARNING", "Nenhum resultado de agente encontrado. Execute o agente primeiro.")
        st.stop()

    results_data = st.session_state['agent_results']
    dataset_name = results_data.get("name", "N/A")
    dataset_format = results_data.get("format", "N/A")
    df_results = results_data.get("results_df")

    if df_results is None or df_results.empty:
        log_message("WARNING", "O agente não produziu nenhum resultado para exibir.")
        st.info("Nenhum resultado de treinamento gerado para exibir. Por favor, verifique a configuração do agente ou os dados de entrada.")
        # Do not st.stop() here. Allow the rest of the function to render empty components.

    # Main layout for dataset info
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown(f"**5.1 Nome do Dataset:** `{dataset_name}`") # Changed numbering
    with col_info2:
        st.markdown(f"**5.2 Formato do Dataset:** `{dataset_format}`") # Changed numbering

    st.markdown("### 5.3 Dataframe de Desempenho") # Changed numbering

    # Prepare the dataframe for display
    df_display = df_results.copy()
    df_display['timestamp'] = pd.to_datetime(df_display['timestamp']).dt.strftime('%d/%m/%Y %H:%M:%S')
    
    columns_to_show_original = ['estimator_name', 'params', 'status', 'score', 'timestamp', 'duration_seconds', 'error', 'pipeline_steps']
    existing_columns = [col for col in columns_to_show_original if col in df_display.columns]
    df_display = df_display[existing_columns]

    df_display = df_display.rename(columns={
        'estimator_name': 'Estimador',
        'params': 'Parâmetros',
        'status': 'Status',
        'score': 'Score',
        'timestamp': 'Timestamp',
        'duration_seconds': 'Duração (s)',
        'error': 'Erro',
        'pipeline_steps': 'Pipeline'
    })

    df_display['Pipeline'] = df_display['Pipeline'].apply(lambda x: str(x))

    # Display the main dataframe (full width)
    selected_trial_index = df_select_rows(
        df_display,
        selection_mode='single-row',
        prompt="Selecione um episódio na tabela para ver os detalhes.",
        key="results_dataframe_selection" # Added key
    )

    # Scatterplot always visible, highlights selected point
    st.markdown("### 5.4 Desempenho Geral dos Modelos") # Renumbered
    render_scatterplot(df_results, selected_trial_index)

    st.markdown("### 5.5 Detalhes do Episódio Selecionado") # Renumbered

    if selected_trial_index is not None:
        selected_trial = df_results.loc[selected_trial_index]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 5.5.1 Métricas principais") # Renumbered
            
            # Status
            status = selected_trial.get('status', 'N/A')
            st.metric(label="Status", value=status)

            # Timestamp
            timestamp_val = pd.to_datetime(selected_trial.get('timestamp', 'N/A'))
            if pd.notna(timestamp_val):
                st.metric(label="Timestamp", value=timestamp_val.strftime('%d/%m/%Y %H:%M:%S'))
            else:
                st.metric(label="Timestamp", value="N/A")

            # Duração
            duration = selected_trial.get('duration_seconds', 0)
            st.metric(label="Duração", value=f"{duration:.4f} s")

        with col2:
            st.markdown("##### 5.5.2 Parâmetros") # Renumbered
            try:
                params_dict = eval(selected_trial['params'])
                num_params = len(params_dict)
                # Calculate dynamic height: 35px per row + 35px for header
                df_height = (num_params + 1) * 35
                # Clamp the height between a min and a max value
                df_height = max(100, min(df_height, 600))
                st.dataframe(pd.DataFrame.from_dict(params_dict, orient='index', columns=['Valor']), height=df_height)
            except (SyntaxError, NameError):
                st.write("Não foi possível exibir os parâmetros.")


        with col3:
            st.markdown("##### 5.5.3 Pipeline") # Renumbered
            pipeline_steps = selected_trial['pipeline_steps']
            # Ensure pipeline_steps is a list before getting its length
            if isinstance(pipeline_steps, list):
                num_steps = len(pipeline_steps)
                # Calculate dynamic height: 120px per step, clamped
                graph_height = max(500, min(num_steps * 120, 1000))
                show_pipeline_graph(pipeline_steps, height=graph_height, show_params=True)
            else:
                # Fallback for unexpected data format
                st.write("Não foi possível calcular a altura dinâmica do pipeline.")
                show_pipeline_graph(pipeline_steps, height=500, show_params=True)

    else:
        st.info("Selecione um episódio na tabela acima para ver seus detalhes.")

    # Download button (remains at the bottom)
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Baixar resultados em CSV",
        data=csv,
        file_name=f'desempenho_{dataset_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
        mime='text/csv',
    )