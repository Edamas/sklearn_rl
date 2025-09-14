import pandas as pd
import streamlit as st
import ast

def parse_metadata_string(s):
    if not isinstance(s, str):
        return s
    s = s.strip()

    # Handle special string values
    if s.lower() == 'none':
        return None
    if s.lower() == 'any':
        return 'any'
    if s.lower() == 'ignored':
        return 'ignored'
    
    # Try literal evaluation for lists
    if s.startswith('[') and s.endswith(']'):
        content = s[1:-1].strip()
        if not content:
            return []
        try:
            # This will work for "['float', 'int']"
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # Fallback for unquoted strings in list, e.g., "[float,int]"
            return [item.strip() for item in content.split(',')]

    # Handle tuple-like strings like "(n_samples, n_features)"
    if s.startswith('(') and s.endswith(')'):
        content = s[1:-1] # Remove parentheses
        if ',' in content:
            return tuple(item.strip() for item in content.split(','))
        else: # Handle single element tuples like "(n_samples,)"
            return (content.strip(),)

    # Handle comma-separated values as lists
    if ',' in s:
        return [item.strip() for item in s.split(',')]
    
    # Try converting to int or float
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            pass # Fall through

    return s

def load_estimators():
    df = pd.read_csv(st.session_state.files.get('estimators'), sep='\t')
    
    # Columns that need robust parsing
    parse_cols = [
        'params_list', 'submethods_list', 'compatible_scores', 
        'input_X_types', 'input_y_types', 'output_X_types', 'output_y_types',
        'input_X_structure', 'input_y_structure', 'output_X_structure', 'output_y_structure'
    ]
    
    for col in parse_cols:
        if col in df.columns: # Check if column exists before applying
            df[col] = df[col].apply(parse_metadata_string)
        
    df['apt_for_training'] = df['apt_for_training'].astype(bool)
    return df


def load_parameters():
    df = pd.read_csv(st.session_state.files.get('parameters'), sep='\t')
    
    df['estimators_list'] = df['estimators_list'].apply(parse_metadata_string)
    df['param_list'] = df['param_list'].apply(parse_metadata_string)
    
    return df
