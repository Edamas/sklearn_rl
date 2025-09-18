import pandas as pd

file_path = "D:\\PROGRAMACAO\\sklearn_rl\\docs\\rubricas.tsv"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Skip header
        if i == 0:
            continue
        
        # Try to split by tab
        parts = line.strip().split('\t')
        
        # Check if the number of parts is consistent with the header
        # The header has 16 columns, so we expect 16 parts
        expected_fields = 16
        if len(parts) != expected_fields:
            print(f"Problematic line: {i+1}")
            print(f"Content: {line.strip()}")
            print(f"Number of fields: {len(parts)}, Expected: {expected_fields}")
            print(f"Parts: {parts}")
            break # Stop at the first problematic line

except Exception as e:
    print(f"Error reading file: {e}")

