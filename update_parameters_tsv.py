import pandas as pd
import numpy as np
import ast

file_path = 'D_training/parameters.tsv'

try:
    df = pd.read_csv(file_path, sep='\t')
except FileNotFoundError:
    print(f"Erro: Arquivo {file_path} não encontrado.")
    exit()

# --- Part 1: Adjusting infinite values (-9999 to -99, 9999 to 99) ---
# Convert relevant columns to numeric, coercing errors to NaN
df['param_min'] = pd.to_numeric(df['param_min'], errors='coerce')
df['param_max'] = pd.to_numeric(df['param_max'], errors='coerce')

df['param_min'] = df['param_min'].replace(-9999, -99)
df['param_max'] = df['param_max'].replace(9999, 99)

# --- Part 2: Silencing verbose parameters ---
# Iterate through rows and modify 'verbose' parameters
for index, row in df.iterrows():
    if 'verbose' in str(row['param_name']).lower():
        param_dtype = str(row['param_dtype'])
        
        # Set param_standard to 0 for int or False for bool
        if 'int' in param_dtype:
            df.loc[index, 'param_standard'] = 0
            # Ensure param_list contains only 0 if it's an int type
            if '[int]' in param_dtype or 'int' == param_dtype:
                df.loc[index, 'param_list'] = str([0])
        elif 'bool' in param_dtype:
            df.loc[index, 'param_standard'] = False
            # Ensure param_list contains only False if it's a bool type
            if '[bool]' in param_dtype or 'bool' == param_dtype:
                df.loc[index, 'param_list'] = str([False])
        
        # If param_list contains True, remove it to force silent
        try:
            current_param_list = ast.literal_eval(str(row['param_list']))
            if True in current_param_list:
                current_param_list.remove(True)
                df.loc[index, 'param_list'] = str(current_param_list)
        except (ValueError, SyntaxError): # Handle cases where param_list is not a valid Python list string
            pass

# Save the modified DataFrame back to the original file
df.to_csv(file_path, sep='\t', index=False, encoding='utf-8')

print("parameters.tsv atualizado com sucesso: valores infinitos ajustados e parâmetros verbose silenciados.")
