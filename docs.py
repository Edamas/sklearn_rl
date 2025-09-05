import streamlit as st
import pandas as pd
import os

def show_docs(file_path):
    """
    Displays the content of a file in the Streamlit app, using the appropriate widget based on the file extension.
    """
    if file_path.endswith(".md"):
        # Special handling for CRONOGRAMA.md if it needs to be a dataframe
        if os.path.basename(file_path) == "CRONOGRAMA.md":
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            # Find the table lines
            table_lines = []
            in_table = False
            for line in lines:
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    in_table = True
                    table_lines.append(line.strip())
                elif in_table:
                    break # End of table
            
            if len(table_lines) >= 2:
                # Parse header
                header = [h.strip() for h in table_lines[0].strip('|').split('|')]
                
                # Parse data
                data = []
                for line in table_lines[2:]: # Skip header and separator
                    row = [d.strip() for d in line.strip('|').split('|')]
                    data.append(row)
                
                df = pd.DataFrame(data, columns=header)
                st.dataframe(df, width="stretch")   # <--- substitui use_container_width=True
            else:
                # Fallback to markdown if table not found or malformed
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                st.markdown(content, unsafe_allow_html=True)

        else: # Regular markdown file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            st.markdown(content, unsafe_allow_html=True)

    elif file_path.endswith(".tsv"):
        df = pd.read_csv(file_path, sep='\t')
        st.dataframe(df, width="stretch")   # <--- substitui use_container_width=True
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        st.text_area("Content", content, height=400)
    elif file_path.endswith(".jpg"):
        st.image(file_path)
    else:
        st.warning(f"File type not supported for: {file_path}")

# Functions for each specific document
def show_readme_md():
    show_docs("README.md")

def show_anotacoes_md():
    show_docs("docs/ANOTACOES.md")

def show_cronograma_md(): # This will now display as dataframe
    show_docs("docs/CRONOGRAMA.md")

def show_cronograma_tsv():
    show_docs("docs/cronograma.tsv")

def show_proposta_tsv():
    show_docs("docs/proposta.tsv")

def show_rubricas_md():
    show_docs("docs/rubricas.md")

def show_tcc_formatado_md():
    show_docs("docs/TCC_FORMATADO.md")

def show_agent_md():
    show_docs("docs/agent.md")
