import streamlit as st
import pandas as pd
import functions as f
from academics.Relatorio_de_Projeto import show_report

def render_gestao_academica():
    st.title("Gestão Acadêmica")
    
    # Exibe o relatório do projeto em um expander
    # show_report()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("O membro acadêmico no time Academics se comunica com a parte administrativa ou executiva do grupo (de 8 pessoas) e com o sistema AVA, cuidando das entregas do grupo para a disciplina de TCC. Enquanto a turma de pesquisa pode focar em conteúdos de outras disciplinas, o acadêmico se dedica à disciplina de TCC, comunicando ao grupo resumos e informações importantes. Também interage com o fórum e organiza os arquivos acadêmicos.")
        st.divider()
        st.header("Cronograma e Entregas")
        f.show_cronograma_by_function("Gestão Acadêmica")
        
        st.divider()
        st.header("Registro de Atividades")
        f.show_registro_atividades_by_function("Gestão Acadêmica")
        st.divider()
        st.subheader("Rubricas relacionadas")
        df_filtered_rubricas = f.get_rubricas_by_function_score_8("Gestão Acadêmica")

        if not df_filtered_rubricas.empty:
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = f.df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_gestao_academica", prompt=None)

            if selected_index is not None and selected_index in df_filtered_rubricas.index:
                selected_rubrica = df_filtered_rubricas.loc[selected_index]
                st.subheader("Ficha da Rubrica")
                
                st.markdown(f"**<font color='#FFD700'>Entrega {selected_rubrica['item_entrega']}: {selected_rubrica['entrega']}</font>**", unsafe_allow_html=True)
                st.markdown(f"  **<font color='#ADD8E6'>Subitem {selected_rubrica['subitem']}: {selected_rubrica['competencia']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    **<font color='#90EE90'>Rubrica {selected_rubrica['item_rubrica']}: {selected_rubrica['rubrica']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    Aplicação no projeto:")
                st.markdown(f"      {selected_rubrica['aplicacao_no_projeto']}")
            else:
                pass # Removido: st.info("Nenhuma rubrica selecionada ou nenhuma rubrica relacionada à função atual.")
        else:
            st.info("Nenhuma rubrica relacionada à função 'Gestão Acadêmica' encontrada.")

        st.divider()
        st.header("Disciplinas Relacionadas")
        f.show_disciplinas_relacionadas_vri("Gestão Acadêmica")

        f.show_referencias_by_function("Gestão Acadêmica")

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
        st.markdown("##### Inputs\n- Orientações da Univesp/AVA\n- Rubricas de avaliação\n- Cronograma oficial do TCC\n- Comunicações da orientação\n- Registro de atividades do grupo")
        st.markdown("##### Outputs\n- Entregas do grupo no AVA\n- Comunicações e resumos para o grupo\n- Organização de arquivos acadêmicos\n- Feedback da orientação")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("##### Requisitos do TCC\n- Cumprimento de prazos e requisitos do AVA.\n- Comunicação eficaz com a coordenação do TCC.\n- Organização e arquivamento de documentos acadêmicos.")
        st.markdown("##### Requisitos da Proposta (sobre o agente)\n- Garantir que o desenvolvimento do agente esteja alinhado com os objetivos acadêmicos do TCC.\n- Acompanhar o progresso do projeto para assegurar a viabilidade das entregas.")