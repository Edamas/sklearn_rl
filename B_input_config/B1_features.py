import streamlit as st
import pandas as pd
from B_input_config.B2_feature_engineering import render_feature_engineering_suite, apply_transformations

def feature_definition():
    """
    Esta é a view para o Passo 2 e 3 do fluxo de trabalho: Feature Engineering.
    
    - Passo 2: O dataset é processado e apresentado no editor de features.
    - Passo 3: O usuário configura e atualiza as features, podendo aplicar transformações.
    """
    # Usa o dataframe processado se já existir, senão, o original.
    df_to_display = st.session_state.get("processed_df", st.session_state.get("original_df"))

    if df_to_display is None:
        st.warning("Por favor, carregue um dataset na Etapa 1.")
        return

    # Renderiza a suíte de engenharia de features, que retorna a configuração atualizada.
    config_df = render_feature_engineering_suite(df_to_display)

    # The config_df is only updated in session_state when the button is clicked.
    # We need to ensure the user has applied changes before proceeding.
    # if st.session_state.get("feature_config_df") is None or st.session_state.feature_config_df.empty or \
    #    'selected_type' not in st.session_state.feature_config_df.columns or \
    #    'is_feature' not in st.session_state.feature_config_df.columns or \
    #    'column_name' not in st.session_state.feature_config_df.columns:
    #     st.info("Por favor, configure as features e clique em 'Aplicar Alterações de Feature Engineering'.")
    #     return # Stop feature_definition if config_df is not valid

    # Use the config_df from session state, which is updated on button click
    config_df = st.session_state.feature_config_df
    
    # Aplica as transformações com base na configuração do usuário.
    processed_df, X_cols, y_cols = apply_transformations(df_to_display, config_df)

    # Garante que as colunas de target (y) realmente existem no dataframe processado.
    sanitized_y_cols = [col for col in y_cols if col in processed_df.columns]

    # --- Atualiza o Estado da Sessão ---
    # ATENÇÃO: É crucial que o estado da sessão seja atualizado corretamente para que
    # as etapas seguintes do fluxo de trabalho funcionem.
    st.session_state.processed_df = processed_df
    st.session_state.X_cols = X_cols
    st.session_state.y_cols = sanitized_y_cols
    
    st.divider()

    # Determina o tipo de tarefa (Regressão, Classificação, etc.) com base na natureza da(s) coluna(s) alvo.
    if sanitized_y_cols:
        is_regression = False
        if all(pd.api.types.is_numeric_dtype(processed_df[col]) for col in sanitized_y_cols):
            if any(pd.api.types.is_float_dtype(processed_df[col]) for col in sanitized_y_cols):
                is_regression = True
            elif all(processed_df[col].nunique() >= 10 for col in sanitized_y_cols):
                is_regression = True
        
        task_type = "Regression" if is_regression else "Classification"
    else:
        task_type = "Unsupervised"
    st.session_state.task_type = task_type

    # --- Opções de Exportação ---
    processed_df_for_export = st.session_state.get("processed_df")
    if processed_df_for_export is not None:
        with st.expander("Exportar Configurações e Dados"):
            col1, col2 = st.columns(2)
            with col1:
                config_to_export = st.session_state.get("feature_config_df", pd.DataFrame())
                if not config_to_export.empty:
                    config_json = config_to_export.to_json(orient="records", indent=4)
                    st.download_button("Download feature_config.json", config_json, "feature_config.json", "application/json")
            with col2:
                csv = processed_df_for_export.to_csv(index=False).encode('utf-8')
                st.download_button("Download Processed_Data.csv", csv, "processed_data.csv", "text/csv")
