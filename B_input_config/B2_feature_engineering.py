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
    st.markdown("2.1 Engenharia de Features")

    if "feature_config_df" not in st.session_state or st.session_state.get("last_df_columns") != df.columns.tolist():
        config_data = []
        for col in df.columns:
            is_target_default = "target" in col.lower()
            is_problematic_default = (df[col].isnull().sum() / len(df) > 0.9) or (df[col].nunique() <= 1)
            # Determine selected_type based on column dtype
            if pd.api.types.is_numeric_dtype(df[col]):
                selected_type = "Numeric"
            elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                selected_type = "Categorical"
            else:
                selected_type = "Other" # Default or handle other types as needed

            config_data.append({
                "Coluna": col,
                "column_name": col, # Add column_name
                "X": (not is_target_default and not is_problematic_default),
                "is_feature": (not is_target_default and not is_problematic_default), # Add is_feature
                "y": (is_target_default and not is_problematic_default),
                "selected_type": selected_type, # Add selected_type
                "Dummies / One-Hot": False,
                "Conversão Data": "Nenhum",
                "Criar Média Móvel (Intervalo)": 0,
                "Criar Lag (Passos Anteriores)": 0,
                "Criar Diferença (n-passos)": 0,
                "Criar Média Expansiva": False,
                "Eliminar Coluna": False,
                "Estatísticas": get_stats_string(df[col]),
                "Gráfico": get_chart_data(df[col]),
            })
        st.session_state.feature_config_df = pd.DataFrame(config_data)
        st.session_state.last_df_columns = df.columns.tolist()

    edited_df = st.data_editor(
        st.session_state.feature_config_df, # Edita diretamente o estado
        column_config={
            "Coluna": st.column_config.TextColumn("Coluna", disabled=True, width="medium"),
            "column_name": st.column_config.TextColumn("Nome da Coluna (Interno)", disabled=True, width="medium", help="Nome interno da coluna, usado para processamento."), # Add column_name
            "X": st.column_config.CheckboxColumn("Feature (X)", default=True),
            "is_feature": st.column_config.CheckboxColumn("É Feature (Interno)", disabled=True, help="Indica se a coluna é usada como feature (X)."), # Add is_feature
            "y": st.column_config.CheckboxColumn("Alvo (y)"),
            "selected_type": st.column_config.TextColumn("Tipo Selecionado (Interno)", disabled=True, width="small", help="Tipo de dado inferido para a coluna."), # Add selected_type
            "Dummies / One-Hot": st.column_config.CheckboxColumn("Dummies / One-Hot", help="Converte colunas de texto/categoria em múltiplas colunas numéricas (0 ou 1)."),
            "Conversão Data": st.column_config.SelectboxColumn("Data/Tempo", help="Extrai informações de colunas de data/tempo.", options=["Nenhum", "Timestamp (Separado)", "Timestamp (Contínuo)"]),
            "Criar Média Móvel (Intervalo)": st.column_config.NumberColumn("Média Móvel", help="Cria uma coluna com a média dos 'n' valores anteriores. Ex: 7", min_value=0, format="%d"),
            "Criar Lag (Passos Anteriores)": st.column_config.NumberColumn("Lag", help="Cria uma coluna com o valor de 'n' passos anteriores. Útil para prever o próximo valor baseado no anterior.", min_value=0, format="%d"),
            "Criar Diferença (n-passos)": st.column_config.NumberColumn("Diferença", help="Cria uma coluna com a diferença para o valor de 'n' passos atrás. Ajuda a estabilizar a série.", min_value=0, format="%d"),
            "Criar Média Expansiva": st.column_config.CheckboxColumn("Média Expansiva", help="Cria uma coluna com a média de todos os valores desde o início até o ponto atual."),
            "Eliminar Coluna": st.column_config.CheckboxColumn("Eliminar Coluna", help="Marque para remover esta coluna do dataset final."), # Nova coluna
            "Estatísticas": st.column_config.TextColumn("Estatísticas", disabled=True, width="large"),
            "Gráfico": st.column_config.LineChartColumn("Tendência"),
        },
        hide_index=True,
        width='stretch',
        key="feature_editor" # Usar uma chave diferente para o editor
    )

    # Botão para aplicar as alterações
    if st.button("Aplicar Alterações de Feature Engineering"):
        st.session_state.feature_config_df = edited_df # Atualiza o estado apenas no clique
        st.session_state.last_df_columns = df.columns.tolist() # Atualiza também
        st.rerun() # Força um rerun para aplicar as mudanças

    # O retorno da função deve ser o DataFrame de configuração atual do session_state
    return st.session_state.feature_config_df

    

def apply_transformations(df, config_df):
    df_transformed = df.copy()
    new_cols_mapping = {}

    # --- Step 1: Perform all transformations ---
    # Iterate through the original config_df to apply all transformations
    print(f"DEBUG: config_df received by apply_transformations:\n{config_df}")
    for _, row in config_df.iterrows():
        col_name = row["Coluna"]
        
        # Ensure the column exists in df_transformed before applying transformations
        if col_name not in df_transformed.columns:
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

    print(f"DEBUG: df_transformed columns AFTER all transformations (before elimination): {df_transformed.columns.tolist()}")

    # --- Step 2: Identify all columns to eliminate (original and derived) ---
    cols_to_eliminate_by_user = config_df[config_df['Eliminar Coluna'] == True]['Coluna'].tolist()
    print(f"DEBUG: cols_to_eliminate_by_user: {cols_to_eliminate_by_user}")
    
    all_cols_to_drop = set()
    for col in cols_to_eliminate_by_user:
        all_cols_to_drop.add(col) # Add original column
        if col in new_cols_mapping: # If it was a dummy/one-hot encoded column
            for new_col in new_cols_mapping[col]:
                all_cols_to_drop.add(new_col)
        
        # Add derived date columns if original date column was eliminated
        # This assumes date conversion creates columns with specific suffixes
        if col in config_df['Coluna'].values and config_df[config_df['Coluna'] == col]['Conversão Data'].iloc[0] != "Nenhum":
            if config_df[config_df['Coluna'] == col]['Conversão Data'].iloc[0] == "Timestamp (Separado)":
                all_cols_to_drop.add(f'{col}_year')
                all_cols_to_drop.add(f'{col}_month')
                all_cols_to_drop.add(f'{col}_day')
                all_cols_to_drop.add(f'{col}_weekofyear')
                all_cols_to_drop.add(f'{col}_dayofyear')
                all_cols_to_drop.add(f'{col}_quarter')
            elif config_df[config_df['Coluna'] == col]['Conversão Data'].iloc[0] == "Timestamp (Contínuo)":
                all_cols_to_drop.add(f'{col}_continuous')
        
        # Add derived moving average, lag, diff, expanding mean columns
        if col in config_df['Coluna'].values and config_df[config_df['Coluna'] == col]['Criar Média Móvel (Intervalo)'].iloc[0] > 0:
            all_cols_to_drop.add(f'{col}_ma{config_df[config_df["Coluna"] == col]["Criar Média Móvel (Intervalo)"].iloc[0]}')
        if col in config_df['Coluna'].values and config_df[config_df['Coluna'] == col]['Criar Lag (Passos Anteriores)'].iloc[0] > 0:
            all_cols_to_drop.add(f'{col}_lag{config_df[config_df["Coluna"] == col]["Criar Lag (Passos Anteriores)"].iloc[0]}')
        if col in config_df['Coluna'].values and config_df[config_df['Coluna'] == col]['Criar Diferença (n-passos)'].iloc[0] > 0:
            all_cols_to_drop.add(f'{col}_diff{config_df[config_df["Coluna"] == col]["Criar Diferença (n-passos)"].iloc[0]}')
        if col in config_df['Coluna'].values and config_df[config_df['Coluna'] == col]['Criar Média Expansiva'].iloc[0]:
            all_cols_to_drop.add(f'{col}_expanding_mean')

    print(f"DEBUG: all_cols_to_drop: {all_cols_to_drop}")

    # --- Step 3: Determine the final set of columns to KEEP and explicitly select them ---
    columns_to_keep = [col for col in df_transformed.columns if col not in all_cols_to_drop]
    print(f"DEBUG: columns_to_keep: {columns_to_keep}")
    df_transformed = df_transformed[columns_to_keep] # Explicitly select columns
    print(f"DEBUG: df_transformed columns AFTER explicit selection: {df_transformed.columns.tolist()}")

    # --- Step 4: Determine X_cols and y_cols from the cleaned df_transformed ---
    y_cols = []
    # Only consider columns that are still in df_transformed and marked as 'y'
    y_config = config_df[(config_df["y"] == True) & (config_df['Coluna'].isin(df_transformed.columns))]
    for _, row in y_config.iterrows():
        col_name = row["Coluna"]
        if col_name in new_cols_mapping:
            y_cols.extend(new_cols_mapping[col_name])
        elif col_name in df_transformed.columns:
            y_cols.append(col_name)
            
    X_cols = [col for col in df_transformed.columns if col not in y_cols]
    print(f"DEBUG: X_cols before return: {X_cols}")
    print(f"DEBUG: y_cols before return: {y_cols}")

    df_transformed.bfill(inplace=True)
    df_transformed.ffill(inplace=True)

    return df_transformed, list(dict.fromkeys(X_cols)), list(dict.fromkeys(y_cols))