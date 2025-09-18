import pandas as pd
import sys
import ast

def debug_parameter(param_name, estimator_name):
    try:
        df_params = pd.read_csv('D_training/parameters.tsv', sep='\t')
    except FileNotFoundError:
        print("Erro: Arquivo D_training/parameters.tsv não encontrado.")
        return

    # Convert 'estimators_list' from string to actual list for comparison
    df_params['estimators_list_parsed'] = df_params['estimators_list'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

    # Filter for the parameter name
    filtered_by_param = df_params[df_params['param_name'] == param_name]

    if filtered_by_param.empty:
        print(f"Parâmetro '{param_name}' não encontrado em parameters.tsv.")
        return

    # Further filter by estimator name
    matching_rows = filtered_by_param[
        filtered_by_param['estimators_list_parsed'].apply(lambda x: estimator_name in x)
    ]

    if matching_rows.empty:
        print(f"Parâmetro '{param_name}' encontrado, mas não associado ao estimador '{estimator_name}'.")
        print("Registros encontrados para o parâmetro:")
        print(filtered_by_param.drop(columns=['estimators_list_parsed']).to_string())
    else:
        print(f"Registro(s) encontrado(s) para o parâmetro '{param_name}' e estimador '{estimator_name}':")
        print(matching_rows.drop(columns=['estimators_list_parsed']).to_string())

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python debug_parameter.py <param_name> <estimator_name>")
    else:
        param_name = sys.argv[1]
        estimator_name = sys.argv[2]
        debug_parameter(param_name, estimator_name)
