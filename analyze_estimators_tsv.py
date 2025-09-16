import pandas as pd
import sys

def analyze_estimators_tsv(file_path):
    print(f"Analisando o arquivo: {file_path}\n")

    try:
        # Read the TSV file
        # Using low_memory=False to avoid mixed type warnings for large files
        # and ensure correct dtype inference, though it might use more memory initially.
        # For very large files, chunking would be better, but for analysis,
        # loading fully is often necessary to check all properties.
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        print(f"DataFrame carregado com {len(df)} linhas e {len(df.columns)} colunas.\n")

        # 1. Identify actively used columns (based on D1_agent_rl.py analysis)
        actively_used_columns = [
            'estimator_name', 'class_path', 'estimator_type', 'input_X_structure',
            'input_X_types', 'input_y_structure', 'input_y_types', 'output_X_structure',
            'output_X_types', 'output_y_structure', 'output_y_types', 'compatible_scores'
        ]

        # 2. Report unused columns
        unused_columns = [col for col in df.columns if col not in actively_used_columns]
        if unused_columns:
            print("--- Correção 1: Remover Colunas Não Utilizadas ---")
            print("As seguintes colunas não são utilizadas pelo algoritmo e podem ser removidas para economizar memória:")
            for col in unused_columns:
                print(f"- {col}")
            print("\n")
        else:
            print("---\n--- Correção 1: Nenhuma Coluna Não Utilizada Encontrada ---\n")

        # 3. Check for inconsistent categorical values
        print("--- Correção 2: Verificar Consistência dos Dados Categóricos ---")
        categorical_columns = [
            'estimator_type', 'input_X_structure', 'input_X_types', 'input_y_structure',
            'input_y_types', 'output_X_structure', 'output_X_types', 'output_y_structure',
            'output_y_types'
        ]
        
        found_inconsistencies = False
        for col in categorical_columns:
            if col in df.columns:
                unique_values = df[col].dropna().unique()
                if len(unique_values) > 50: # Arbitrary threshold for too many unique values
                    print(f"Atenção: Coluna '{col}' tem muitos valores únicos ({len(unique_values)}). Revise para consistência.")
                    print(f"Exemplo de valores: {unique_values[:5]}")
                    found_inconsistencies = True
                elif len(unique_values) > 1:
                    # Check for case inconsistencies if values are strings
                    if all(isinstance(val, str) for val in unique_values):
                        lower_case_values = [val.lower() for val in unique_values]
                        if len(set(lower_case_values)) < len(unique_values):
                            print(f"Atenção: Coluna '{col}' contém valores com inconsistências de caixa (case-insensitivity).")
                            print(f"Valores únicos: {unique_values}")
                            found_inconsistencies = True
                # else: print(f"Coluna '{col}' parece consistente (ou tem apenas um valor único).")
        if not found_inconsistencies:
            print("Nenhuma inconsistência aparente em colunas categóricas (baseado em amostragem e case-insensitivity).")
        print("\n")

        # 4. Check compatible_scores format
        print("--- Correção 3: Confirmar Formato de 'compatible_scores' ---")
        if 'compatible_scores' in df.columns:
            # Check for non-string types or malformed strings
            malformed_scores = df[df['compatible_scores'].apply(lambda x: not isinstance(x, str) or not (x.startswith('[') and x.endswith(']')))].index.tolist()
            if malformed_scores:
                print("Atenção: As seguintes linhas têm 'compatible_scores' com formato incorreto (não é uma string entre colchetes):")
                for idx in malformed_scores[:10]: # Show first 10 examples
                    print(f"- Linha {idx}: '{df.loc[idx, 'compatible_scores']}'")
                if len(malformed_scores) > 10:
                    print(f"... e mais {len(malformed_scores) - 10} linhas.")
                print("Formato esperado: '[score1,score2,score3]'")
            else:
                print("A coluna 'compatible_scores' parece estar no formato esperado (string entre colchetes).")
        else:
            print("Coluna 'compatible_scores' não encontrada.")
        print("\n")

        # 5. Check for duplicate entries
        print("--- Correção 4: Remover Entradas Duplicadas ---")
        # Define a subset of columns to consider for duplicates
        # 'estimator_name' and 'class_path' are good candidates for unique identification
        subset_for_duplicates = ['estimator_name', 'class_path']
        if all(col in df.columns for col in subset_for_duplicates):
            duplicates = df[df.duplicated(subset=subset_for_duplicates, keep=False)]
            if not duplicates.empty:
                print("Atenção: As seguintes linhas são duplicatas (baseado em 'estimator_name' e 'class_path'):")
                print(duplicates.sort_values(by=subset_for_duplicates).to_string())
                print("\nRecomenda-se remover as duplicatas, mantendo apenas uma ocorrência.")
            else:
                print("Nenhuma entrada duplicada encontrada (baseado em 'estimator_name' e 'class_path').")
        else:
            print(f"Não foi possível verificar duplicatas: uma ou mais colunas ({', '.join(subset_for_duplicates)}) estão faltando.")
        print("\n")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
    except Exception as e:
        print(f"Ocorreu um erro durante a análise: {e}")

if __name__ == "__main__":
    # The file path will be passed as a command-line argument
    if len(sys.argv) > 1:
        estimators_tsv_path = sys.argv[1]
        analyze_estimators_tsv(estimators_tsv_path)
    else:
        print("Uso: python analyze_estimators_tsv.py <caminho_para_estimators.tsv>")
        print("Exemplo: python analyze_estimators_tsv.py D:\PROGRAMACAO\sklearn_rl\D_training\estimators.tsv")
