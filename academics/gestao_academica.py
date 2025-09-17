import streamlit as st
import pandas as pd
from functions import df_select_rows

def render_gestao_academica():
    st.title("Gestão Acadêmica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("O membro acadêmico no time Academics se comunica com a parte administrativa ou executiva do grupo (de 8 pessoas) e com o sistema AVA, cuidando das entregas do grupo para a disciplina de TCC. Enquanto a turma de pesquisa pode focar em conteúdos de outras disciplinas, o acadêmico se dedica à disciplina de TCC, comunicando ao grupo resumos e informações importantes. Também interage com o fórum e organiza os arquivos acadêmicos.")
        st.divider()
        st.header("Cronograma e Entregas")
        cronograma_path = "docs/cronograma.tsv"
        try:
            df_cronograma = pd.read_csv(cronograma_path, sep='\t')
            df_cronograma.rename(columns={'Vencimento das atividades': 'Vencimento'}, inplace=True)
            st.dataframe(df_cronograma, hide_index=True, width='stretch')
        except FileNotFoundError:
            st.error(f"Arquivo de cronograma não encontrado: {cronograma_path}")
        except Exception as e:
            st.error(f"Erro ao carregar ou processar o cronograma: {e}")
        
        
        st.divider()
        st.header("Rubricas de Avaliação (Nota 8)")

        @st.cache_data(show_spinner=False)
        def load_and_process_rubricas_data_for_gestao_academica():
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
                elif re.match(r'^\\d+\\.\\d+\\.\\d+', line):
                    rubrica_text_from_md = re.sub(r'^\\d+\\.\\d+\\.\\d+\\s+', '', line).strip()
                    rubrica_id_match = re.match(r'^(\\d+\\.\\d+\\.\\d+)', line)
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
                match = re.match(r'^(\\d+\\.\\d+\\.\\d+)', str(text))
                if match:
                    return match.group(1)
                return None

            df_rubricas['id'] = df_rubricas['Rubrica de Avaliação'].apply(extract_id)
            df_full = pd.merge(df_rubricas, df_map, on='id', how='left')
            return df_full

        df_full = load_and_process_rubricas_data_for_gestao_academica()
        
        if not df_full.empty:
            funcao_nome = "Gestão Acadêmica"
            if funcao_nome in df_full.columns:
                df_filtered = df_full[df_full[funcao_nome] == 8].copy()

                if not df_filtered.empty:
                    df_display = df_filtered[['Rubrica de Avaliação']].copy()
                    df_display.rename(columns={'Rubrica de Avaliação': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
                    
                    selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_gestao_academica")

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

        st.header("Referências e Fontes")
        st.markdown("- Documentação e calendário oficial da disciplina de TCC.")
    
    
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Gestão Acadêmica": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Orientações e Rubricas")
        st.markdown("##### Outputs\n- Entregas e Registros no AVA")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Gestão de prazos.")