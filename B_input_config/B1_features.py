import streamlit as st
import pandas as pd
import json
from B_input_config.B2_feature_engineering import render_feature_engineering_suite, apply_transformations

def feature_definition():
    # On first run, use original_df. On subsequent runs, use the processed_df for continuous feature engineering.
    df_to_display = st.session_state.get("processed_df", st.session_state.get("original_df"))

    if df_to_display is None:
        st.warning("Por favor, carregue um dataset na Etapa 1.")
        return

    # Render the interactive feature editor using the current state of the dataframe
    config_df = render_feature_engineering_suite(df_to_display)

    st.divider()
    if st.button("Aplicar Transformações", type="primary"):
        # Apply transformations to the current dataframe
        processed_df, X_cols, y_cols = apply_transformations(df_to_display, config_df)

        # --- Sanitize y_cols to prevent KeyError downstream ---
        sanitized_y_cols = [col for col in y_cols if col in processed_df.columns]

        # --- Update Session State ---
        st.session_state.processed_df = processed_df
        st.session_state.X_cols = X_cols
        st.session_state.y_cols = sanitized_y_cols

        if sanitized_y_cols:
            is_regression = all(pd.api.types.is_numeric_dtype(processed_df[col]) and processed_df[col].nunique() >= 20 for col in sanitized_y_cols)
            task_type = "Regression" if is_regression else "Classification"
        else:
            task_type = "Unsupervised"
        st.session_state.task_type = task_type

        # Force the data_editor to be rebuilt in the next run with the new processed_df
        st.session_state.feature_config_df = None
        st.rerun()

    # --- Exporting Options ---
    processed_df_for_export = st.session_state.get("processed_df")
    if processed_df_for_export is not None:
        st.subheader("2.3 Exportar Configurações")
        col1, col2 = st.columns(2)
        with col1:
            # We export the config of the df that is currently displayed in the editor
            config_to_export = st.session_state.get("feature_config_df", pd.DataFrame())
            if not config_to_export.empty:
                config_json = config_to_export.to_json(orient="records", indent=4)
                st.download_button("Download feature_config.json", config_json, "feature_config.json", "application/json")
        with col2:
            csv = processed_df_for_export.to_csv(index=False).encode('utf-8')
            st.download_button("Download Processed_Data.csv", csv, "processed_data.csv", "text/csv")
