import streamlit as st
import pandas as pd
from functions import df_select_rows

def render_pesquisa_academica():
    st.title("Pesquisa Acadêmica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("A gestão de pesquisa cria táticas para a equipe, com embasamento e desenvolvimento teórico. No contexto acadêmico do TCC de Ciência de Dados, explora a teoria de bibliotecas online e livros. O gestor de pesquisa busca referências (principais e secundárias) da disciplina de TCC, orientações, e se comunica com a orientadora, banca avaliadora e demais membros do grupo.")
        st.divider()
        st.header("Fundamentação Teórica")
        st.markdown("A fundamentação aborda conceitos de agentes inteligentes, sistemas autônomos, AutoML e a biblioteca Scikit-learn, com base em autores como Russell & Norvig (2020), Pedregosa et al. (2011) e Feurer et al. (2019).")
        st.divider()
        st.header("Rubricas de Avaliação (Nota 8)")

        @st.cache_data(show_spinner=False)
        def load_and_process_rubricas_data_for_pesquisa_academica():
            import pandas as pd
            import re

            try:
                with open("D:\\PROGRAMACAO\\sklearn_rl\\docs\\rubricas.md", 'r', encoding='utf-8') as f:
                    md_content = f.read()
            except FileNotFoundError:
                st.error("Arquivo rubricas.md não encontrado.")
                return pd.DataFrame()

            entrega = None
            competencia = None
            rubricas_map = {}

            for line in md_content.splitlines():
                line = line.strip()
                if line.startswith('# '):
                    match = re.search(r'`(Entrega[^`]+)`', line)
                    if match:
                        entrega = match.group(1)
                elif line.startswith('##### '):
                    match = re.search(r'`([^`]+)`', line)
                    if match:
                        competencia = match.group(1)
                elif re.match(r'^\d+\.\d+\.\d+', line):
                    rubrica_text_from_md = re.sub(r'^\d+\.\d+\.\d+\s+', '', line).strip()
                    rubrica_id_match = re.match(r'^(\d+\.\d+\.\d+)', line)
                    if rubrica_id_match:
                        rubrica_id = rubrica_id_match.group(1)
                        rubricas_map[rubrica_id] = {
                            "Entrega": entrega,
                            "Competência": competencia,
                        }
            
            df_map = pd.DataFrame.from_dict(rubricas_map, orient='index').reset_index().rename(columns={'index': 'id'})

            try:
                df_rubricas = pd.read_csv("D:\\PROGRAMACAO\\sklearn_rl\\docs\\rubricas.tsv", sep='\t')
            except FileNotFoundError:
                st.error("Arquivo rubricas.tsv não encontrado.")
                return pd.DataFrame()
            
            def extract_id(text):
                match = re.match(r'^(\d+\.\d+\.\d+)', str(text))
                if match:
                    return match.group(1)
                return None

            df_rubricas['id'] = df_rubricas['Rubrica de Avaliação'].apply(extract_id)
            df_full = pd.merge(df_rubricas, df_map, on='id', how='left')
            return df_full

        df_full = load_and_process_rubricas_data_for_pesquisa_academica()
        
        if not df_full.empty:
            funcao_nome = "Pesquisa Acadêmica"
            if funcao_nome in df_full.columns:
                df_filtered = df_full[df_full[funcao_nome] == 8].copy()

                if not df_filtered.empty:
                    df_display = df_filtered[['Rubrica de Avaliação']].copy()
                    df_display.rename(columns={'Rubrica de Avaliação': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
                    
                    selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_pesquisa_academica")

                    if selected_index is not None and selected_index in df_filtered.index:
                        selected_rubrica = df_filtered.loc[selected_index]
                        st.subheader("Ficha da Rubrica")
                        
                        st.markdown(f"**Entrega:** {selected_rubrica.get('Entrega', 'N/A')}")
                        st.markdown(f"**Competência:** {selected_rubrica.get('Competência', 'N/A')}")
                        st.markdown(f"**Rubrica de Avaliação:** {selected_rubrica.get('Rubrica de Avaliação', 'N/A')}")
                        st.markdown(f"**Aplicação no projeto:** {funcao_nome}")
                else:
                    st.info(f"Nenhuma rubrica com nota 8 para '{funcao_nome}'.")
            else:
                st.error(f"Coluna '{funcao_nome}' não encontrada em rubricas.tsv.")
        

        st.divider()
        with st.expander("Disciplinas Relacionadas ao Projeto", expanded=False):
            disciplinas_path = "D:\\PROGRAMACAO\\sklearn_rl\\docs\\disciplinas_relacionadas.tsv"
            try:
                df_disciplinas = pd.read_csv(disciplinas_path, sep='\t')

                # Add 'Relevância' column based on a simple mapping
                relevance_map = {
                    "Algoritmos e Programação de Computadores I": 5,
                    "Algoritmos e Programação de Computadores II": 5,
                    "Introdução a Ciência de Dados": 5,
                    "Aprendizado de Máquinas": 5,
                    "Mineração de Dados": 5,
                    "Computação Escalável": 5,
                    "Redes Neurais": 5,
                    "Aprendizado Profundo": 5,
                    "Visualização Computacional": 5,
                    "Processamento de Linguagem Natural": 5,
                    "Engenharia de Software": 4,
                    "Modelagem e Inferência Estatística": 4,
                    "Banco de Dados": 3,
                    "Infraestrutura de Sistemas de Software Redes, Nuvem": 3,
                    "Pensamento Computacional": 3,
                    "Fundamentos Matemáticos para Computação": 3,
                    "Cálculo I": 2,
                    "Cálculo II": 2,
                    "Projeto Integrador em Computação I": 4,
                    "Projeto Integrador em Computação II": 4,
                    "Projeto Integrador em Computação III": 4,
                    "Projeto Integrador em Computação IV": 4,
                    "Desenvolvimento web": 3,
                    "Introdução a Conceitos de Computação": 3,
                    "Sistemas Computacionais (Organização e Arquitetura de Computadores, SO)": 3,
                    "Ética, Cidadania e Sociedade": 1,
                    "Leitura e Produção de Textos": 1,
                    "Inglês": 1,
                    "Matemática Básica": 2,
                    "Gestão da Inovação e Desenvolvimento de Produtos": 2,
                    "Formação Profissional em Computação": 1,
                    "Impactos da Computação na Sociedade": 1,
                    "Planejamento Estratégico de Negócios": 2,
                    "Estágio Supervisionado para Bacharelado em Ciência de Dados": 1,
                    "Trabalho de Conclusão de Curso (TCC)": 5, # TCC is highly relevant
                    "Eletiva": 1 # Default for unknown
                }
                df_disciplinas['Relevância'] = df_disciplinas['Disciplina:'].map(relevance_map).fillna(1).astype(int)

                # Sort by relevance
                df_disciplinas_sorted = df_disciplinas.sort_values(by='Relevância', ascending=False)

                # Select relevant columns for display
                display_cols = ['Bimestre', 'Disciplina:', 'Relevância', 'Objetivo: ', 'Ementa:', 'Conteúdo programático', 'Bibliografia Básica e Complementar']
                st.dataframe(df_disciplinas_sorted[display_cols], hide_index=True, width='stretch')

            except FileNotFoundError:
                st.error(f"Arquivo de disciplinas não encontrado: {disciplinas_path}")
            except Exception as e:
                st.error(f"Erro ao carregar ou processar as disciplinas: {e}")
    st.divider()
    st.header("Referências e Fontes")
    st.markdown("- ABNT – NBR 14724.\n- FEURER, M. et al. Auto-sklearn 2.0.\n- PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python.\n- RUSSELL, S.; NORVIG, P. Artificial Intelligence: A Modern Approach.\n- ZOPH, B.; LE, Q. V. Neural Architecture Search with Reinforcement Learning.")
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Pesquisa Acadêmica": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Bibliografia do curso")
        st.markdown("##### Outputs\n- Embasamento teórico")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Pesquisar bases científicas.\n- Usar gerenciador de referências.")
