import streamlit as st
import pandas as pd
import numpy as np

def get_chart_data(series):
    try:
        if pd.api.types.is_numeric_dtype(series):
            series = series.copy().fillna(series.mean())
            if series.nunique() > 1:
                normalized_series = (series - series.min()) / (series.max() - series.min())
                return normalized_series.tolist()
    except Exception:
        return None
    return None

def get_stats_string(series):
    if pd.api.types.is_numeric_dtype(series):
        stats = series.describe()
        return f"Média: {stats.get('mean', 0):.2f}, Min: {stats.get('min', 0):.2f}, Max: {stats.get('max', 0):.2f}, Nulos: {series.isnull().sum()}"
    else:
        stats = series.describe()
        return f"Únicos: {stats.get('unique', 0)}, Top: {stats.get('top', 'N/A')}, Nulos: {series.isnull().sum()}"

def render_feature_engineering_suite(df):
    st.subheader("2.1 Engenharia de Features")

    REQUIRED_COLS = {
        "Coluna", "X", "y", "Dummies / One-Hot", "Conversão Data",
        "Criar Média Móvel (Intervalo)", "Criar Lag (Passos Anteriores)",
        "Criar Diferença (n-passos)", "Criar Média Expansiva",
        "Estatísticas", "Gráfico"
    }
    config_df_from_state = st.session_state.get("feature_config_df")
    is_invalid = (
        config_df_from_state is None or
        st.session_state.get("last_df_columns") != df.columns.tolist() or
        not REQUIRED_COLS.issubset(config_df_from_state.columns)
    )

    if is_invalid:
        config_data = []
        for col in df.columns:
            is_target = "target" in col.lower()
            is_problematic = (df[col].isnull().sum() / len(df) > 0.9) or (df[col].nunique() <= 1)
            
            config_data.append({
                "Coluna": col,
                "X": not is_target and not is_problematic,
                "y": is_target and not is_problematic,
                "Dummies / One-Hot": False,
                "Conversão Data": "Nenhum",
                "Criar Média Móvel (Intervalo)": 0,
                "Criar Lag (Passos Anteriores)": 0,
                "Criar Diferença (n-passos)": 0,
                "Criar Média Expansiva": False,
                "Estatísticas": get_stats_string(df[col]),
                "Gráfico": get_chart_data(df[col]),
            })
        st.session_state.feature_config_df = pd.DataFrame(config_data)
        st.session_state.last_df_columns = df.columns.tolist()

    edited_df = st.data_editor(
        st.session_state.feature_config_df,
        column_config={
            "Coluna": st.column_config.TextColumn("Coluna", disabled=True, width="medium"),
            "X": st.column_config.CheckboxColumn("Feature (X)", default=True),
            "y": st.column_config.CheckboxColumn("Alvo (y)"),
            "Dummies / One-Hot": st.column_config.CheckboxColumn("Dummies / One-Hot", help="Converte colunas de texto/categoria em múltiplas colunas numéricas (0 ou 1)."),
            "Conversão Data": st.column_config.SelectboxColumn("Data/Tempo", help="Extrai informações de colunas de data/tempo.", options=["Nenhum", "Timestamp (Separado)", "Timestamp (Contínuo)"]),
            "Criar Média Móvel (Intervalo)": st.column_config.NumberColumn("Média Móvel", help="Cria uma coluna com a média dos 'n' valores anteriores. Ex: 7", min_value=0, format="%d"),
            "Criar Lag (Passos Anteriores)": st.column_config.NumberColumn("Lag", help="Cria uma coluna com o valor de 'n' passos anteriores. Útil para prever o próximo valor baseado no anterior.", min_value=0, format="%d"),
            "Criar Diferença (n-passos)": st.column_config.NumberColumn("Diferença", help="Cria uma coluna com a diferença para o valor de 'n' passos atrás. Ajuda a estabilizar a série.", min_value=0, format="%d"),
            "Criar Média Expansiva": st.column_config.CheckboxColumn("Média Expansiva", help="Cria uma coluna com a média de todos os valores desde o início até o ponto atual."),
            "Estatísticas": st.column_config.TextColumn("Estatísticas", disabled=True, width="large"),
            "Gráfico": st.column_config.LineChartColumn("Tendência"),
        },
        hide_index=True,
        use_container_width=True,
        key="feature_editor"
    )
    
    st.session_state.feature_config_df = edited_df
    return edited_df

def apply_transformations(df, config_df):
    df_transformed = df.copy()
    new_cols_mapping = {}
    config_df_copy = config_df.copy()

    for _, row in config_df_copy.iterrows():
        col_name = row["Coluna"]
        is_disabled = not row["X"] and not row["y"]

        if is_disabled:
            if col_name in df_transformed.columns: df_transformed.drop(columns=[col_name], inplace=True)
            continue

        if row["Dummies / One-Hot"] and col_name in df_transformed.columns:
            dummies = pd.get_dummies(df_transformed[col_name], prefix=col_name, dummy_na=False)
            df_transformed = pd.concat([df_transformed.drop(columns=[col_name]), dummies], axis=1)
            new_cols_mapping[col_name] = dummies.columns.tolist()

        date_conv = row["Conversão Data"]
        if date_conv != "Nenhum" and col_name in df_transformed.columns:
            dt_series = pd.to_datetime(df_transformed[col_name])
            if date_conv == "Timestamp (Separado)":
                df_transformed[f'{col_name}_year'] = dt_series.dt.year
                df_transformed[f'{col_name}_month'] = dt_series.dt.month
                df_transformed[f'{col_name}_day'] = dt_series.dt.day
                df_transformed[f'{col_name}_weekofyear'] = dt_series.dt.isocalendar().week.astype(int)
                df_transformed[f'{col_name}_dayofyear'] = dt_series.dt.dayofyear
                df_transformed[f'{col_name}_quarter'] = dt_series.dt.quarter
            elif date_conv == "Timestamp (Contínuo)":
                df_transformed[f'{col_name}_continuous'] = dt_series.astype(np.int64) // 10**9
            df_transformed.drop(columns=[col_name], inplace=True)

        ma_window = row["Criar Média Móvel (Intervalo)"]
        if ma_window > 0 and col_name in df_transformed.columns:
            df_transformed[f'{col_name}_ma{ma_window}'] = df_transformed[col_name].rolling(window=ma_window).mean()

        lag_steps = row["Criar Lag (Passos Anteriores)"]
        if lag_steps > 0 and col_name in df_transformed.columns:
            df_transformed[f'{col_name}_lag{lag_steps}'] = df_transformed[col_name].shift(lag_steps)

        diff_steps = row["Criar Diferença (n-passos)"]
        if diff_steps > 0 and col_name in df_transformed.columns:
            df_transformed[f'{col_name}_diff{diff_steps}'] = df_transformed[col_name].diff(periods=diff_steps)

        if row["Criar Média Expansiva"] and col_name in df_transformed.columns:
            df_transformed[f'{col_name}_expanding_mean'] = df_transformed[col_name].expanding().mean()

    # Reset configs to prevent re-application
    config_df["Criar Média Móvel (Intervalo)"] = 0
    config_df["Criar Lag (Passos Anteriores)"] = 0
    config_df["Criar Diferença (n-passos)"] = 0
    config_df["Criar Média Expansiva"] = False

    y_cols = []
    y_config = config_df_copy[config_df_copy["y"] == True]
    for _, row in y_config.iterrows():
        col_name = row["Coluna"]
        if col_name in new_cols_mapping:
            y_cols.extend(new_cols_mapping[col_name])
        elif col_name in df_transformed.columns:
            y_cols.append(col_name)
            
    X_cols = [col for col in df_transformed.columns if col not in y_cols]

    df_transformed.fillna(method='bfill', inplace=True)
    df_transformed.fillna(method='ffill', inplace=True)

    return df_transformed, list(dict.fromkeys(X_cols)), list(dict.fromkeys(y_cols))