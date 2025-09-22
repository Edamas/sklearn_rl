import pandas as pd
import numpy as np
import ast
import os
from collections import defaultdict

file_path = 'D_training/parameters.tsv'

try:
    # Specify dtype=str for param_min and param_max to force string interpretation
    df = pd.read_csv(file_path, sep='\t', dtype={'param_min': str, 'param_max': str})
except FileNotFoundError:
    print(f"Erro: Arquivo {file_path} não encontrado.")
    exit()

def parse_param_type(param_type_str):
    try:
        parsed = ast.literal_eval(param_type_str)
        if isinstance(parsed, list):
            return [item.strip() for item in parsed]
        else:
            return [param_type_str.strip()] # Not a list, treat as single type
    except (ValueError, SyntaxError):
        return [param_type_str.strip()] # Not a list, treat as single type

def parse_param_list(param_list_str):
    try:
        parsed = ast.literal_eval(param_list_str)
        if isinstance(parsed, list):
            return [item for item in parsed]
        else:
            return [] # Not a list, return empty
    except (ValueError, SyntaxError):
        return [] # Not a list, return empty

# Analyze param_dtype
param_dtype_counts = defaultdict(int)
for _, row in df.iterrows():
    param_dtypes = parse_param_type(str(row['param_dtype']))
    for dtype in param_dtypes:
        param_dtype_counts[dtype] += 1

# Analyze param_list
param_list_item_counts = defaultdict(int)
for _, row in df.iterrows():
    param_list_items = parse_param_list(str(row['param_list']))
    for item in param_list_items:
        param_list_item_counts[str(item)] += 1 # Convert to string for consistent keys

# Print analysis summary
output_lines = []
output_lines.append("--- Análise de Tipos de Parâmetros (param_dtype) ---")
for dtype, count in sorted(param_dtype_counts.items()):
    output_lines.append(f"Tipo: {dtype}, Contagem: {count}")

output_lines.append("\n--- Análise de Itens da Lista de Parâmetros (param_list) ---")
for item, count in sorted(param_list_item_counts.items()):
    output_lines.append(f"Item: {item}, Contagem: {count}")

print("\n".join(output_lines))

print("\nAnálise de parameters.tsv concluída.")