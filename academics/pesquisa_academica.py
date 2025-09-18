import streamlit as st
import pandas as pd
import functions as f
from functions import df_select_rows, get_rubricas_by_function_score_8

def render_pesquisa_academica():
    st.title("Pesquisa Acadêmica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("A gestão de pesquisa cria táticas para a equipe, com embasamento e desenvolvimento teórico. No contexto acadêmico do TCC de Ciência de Dados, explora a teoria de bibliotecas online e livros. O gestor de pesquisa busca referências (principais e secundárias) da disciplina de TCC, orientações, e se comunica com a orientadora, banca avaliadora e demais membros do grupo.")
        st.divider()
        st.header("Fundamentação Teórica")
        st.markdown("A fundamentação aborda conceitos de agentes inteligentes, sistemas autônomos, AutoML e a biblioteca Scikit-learn, com base em autores como Russell & Norvig (2020), Pedregosa et al. (2011) e Feurer et al. (2019).")
        st.divider()
        st.subheader("Rubricas relacionadas")
        df_filtered_rubricas = get_rubricas_by_function_score_8("Pesquisa Acadêmica")

        if not df_filtered_rubricas.empty:
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_pesquisa_academica", prompt=None)

            if selected_index is not None and selected_index in df_filtered_rubricas.index:
                selected_rubrica = df_filtered_rubricas.loc[selected_index]
                st.subheader("Ficha da Rubrica")
                
                st.markdown(f"**<font color='#FFD700'>Entrega {selected_rubrica['item_entrega']}: {selected_rubrica['entrega']}</font>**", unsafe_allow_html=True)
                st.markdown(f"  **<font color='#ADD8E6'>Subitem {selected_rubrica['subitem']}: {selected_rubrica['competencia']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    **<font color='#90EE90'>Rubrica {selected_rubrica['item_rubrica']}: {selected_rubrica['rubrica']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    Aplicação no projeto:")
                st.markdown(f"      {selected_rubrica['aplicacao_no_projeto']}")
            else:
                pass # Removido: st.info(f"Nenhuma rubrica selecionada ou nenhuma rubrica relacionada à função atual.")
        else:
            st.info("Nenhuma rubrica relacionada à função 'Pesquisa Acadêmica' encontrada.")
        

        st.divider()
        st.header("Análise de Temas de TCCs de Ciência de Dados da Univesp")
        st.markdown("""
        A análise dos temas de TCCs de Ciência de Dados da Univesp revela que a maioria dos trabalhos se concentra em pesquisa científica, em vez de desenvolvimento de produtos ou soluções de negócio. Essa tendência influenciou a decisão de restringir o escopo inicial deste projeto para um TCC de pesquisa científica.
        No entanto, é importante ressaltar que o agente de RL desenvolvido neste projeto tem potencial para se tornar um produto. Ele pode ser expandido para uma ferramenta interativa que auxilia estudantes e pesquisadores na seleção de algoritmos e configuração de parâmetros do Scikit-learn, otimizando o fluxo de trabalho de modelagem preditiva.
        """)
        
        df_temas = pd.read_csv(st.session_state['files']['temas_de_TCC_Univesp'], sep='\t', engine='python')
        
        selected_index = f.df_select_rows(df_temas, selection_mode='single-row', key=f"temas_tcc_univesp", prompt=None)

        if selected_index is not None and selected_index in df_temas.index:
            selected_tema = df_temas.loc[selected_index]
            st.subheader("Ficha do Tema")
            st.markdown(f.get_card_style(), unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card'>
                <div class='card-body'>
                    <h5 class='card-title'><font color='#FFD700'>{selected_tema['títulos tcc univesp']}</font></h5>
                    <p class='card-text'><font color='#ADD8E6'>Nº Páginas:</font> {selected_tema['nº páginas']}</p>
                    <p class='card-text'><font color='#90EE90'>Complexidade do Tema:</font> {selected_tema['Complexidade do tema']}</p>
                    <p class='card-text'><font color='#FFA07A'>Nível de Incerteza Científica:</font> {selected_tema['Nível de Incerteza Científica']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pass # Removido: st.info("Nenhum tema selecionado.")
        st.divider()
        st.header("Registro de Atividades")
        f.show_registro_atividades_by_function("Pesquisa Acadêmica")
        st.divider()
        st.header("Cronograma e Entregas")
        f.show_cronograma_by_function("Pesquisa Acadêmica")
        st.divider()
        st.header("Disciplinas Relacionadas")
        f.show_disciplinas_relacionadas_vri("Pesquisa Acadêmica")

        f.show_referencias_by_function("Pesquisa Acadêmica")

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
        st.markdown("##### Inputs\n- Objetivos e justificativas do TCC\n- Exploração de bibliotecas online e livros\n- Referências principais e secundárias da disciplina de TCC\n- Orientações da Univesp/AVA\n- Comunicações da orientadora, banca avaliadora e grupo")
        st.markdown("##### Outputs\n- Referências bibliográficas (organizadas)\n- Embasamento teórico para o TCC\n- Análise de temas de TCCs da Univesp\n- Fundamentação teórica (agentes inteligentes, AutoML, Scikit-learn)")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("##### Requisitos do TCC\n- Levantamento bibliográfico abrangente e relevante.\n- Organização e citação correta das referências (ABNT).\n- Desenvolvimento de uma fundamentação teórica sólida.\n- Análise crítica de temas e abordagens existentes.")
        st.markdown("##### Requisitos da Proposta (sobre o agente)\n- Pesquisa sobre tecnologias e algoritmos de IA/ML relevantes para o agente.\n- Identificação de lacunas na literatura que o agente pode preencher.\n- Avaliação da viabilidade científica e técnica da proposta do agente.")