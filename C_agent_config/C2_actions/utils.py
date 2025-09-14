import pandas as pd

def convert_column_to_list(string_series):
    # Converte as colunas de string para listas reais
    series_list = []
    for content in string_series:
        if isinstance(content, str):
            value = content.strip('[]')
            value = value.replace("'", "")
            value = value.strip()
            value = value.replace('.', '¥')
            value = value.replace(',', '¥')
            value = value.split('¥')
        else:
            value = [content]
        series_list.append(value)
    return series_list

def convert_inf_values(df):
    """
    Converte valores -9999 para -inf e 9999 para inf em colunas numéricas
    sem fazer os valores desaparecerem
    """
    # Cria uma cópia do dataframe para não modificar o original
    df_copy = df.copy()
    
    for col in df_copy.columns:
        if df_copy[col].dtype in ['int64', 'float64']:  # Aplica apenas em colunas numéricas
            # Usa replace com regex=False para evitar que valores desapareçam
            df_copy[col] = df_copy[col].replace(-9999, float('-inf'))
            df_copy[col] = df_copy[col].replace(9999, float('inf'))
    
    return df_copy
