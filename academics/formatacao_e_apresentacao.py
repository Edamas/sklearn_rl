import streamlit as st
import pandas as pd
import functions as f

def render_formatacao_e_apresentacao():
    st.title("Formatação e Apresentação")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("No contexto acadêmico, este membro é responsável pela formatação do Word, criação do template de normas ABNT, e pela 'montagem do trabalho', garantindo a 'língua', 'comunicação' e 'coesão'. Também administra o vídeo de apresentação final, recebendo contribuições do grupo e gerando o produto final (vídeo e link para capa do trabalho).")
        
        st.divider()
        st.header("Dinâmica para Vídeo")
        with st.expander("Roteiro", expanded=False):
            with open(st.session_state['files']['roteiro_video'], 'r', encoding='utf-8') as f_file:
                roteiro_video = f_file.read()
            st.markdown(roteiro_video)
    
    
        st.divider()
        st.header("Registro de Atividades")
        f.show_registro_atividades_by_function("Formatação e Apresentação")
        st.divider()
        st.header("Cronograma e Entregas")
        f.show_cronograma_by_function("Formatação e Apresentação")
        st.divider()
        st.header("Rubricas relacionadas")
        # Obter o DataFrame filtrado da função
        df_filtered_rubricas = f.get_rubricas_by_function_score_8("Formatação e Apresentação")

        if not df_filtered_rubricas.empty:
            # Preparar o DataFrame para exibição interativa
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = f.df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_formatacao_e_apresentacao")

            if selected_index is not None and selected_index in df_filtered_rubricas.index:
                selected_rubrica = df_filtered_rubricas.loc[selected_index]
                st.subheader("Ficha da Rubrica")
                
                # Exibir a ficha da rubrica com a formatação desejada
                st.markdown(f"**<font color='#FFD700'>Entrega {selected_rubrica['item_entrega']}: {selected_rubrica['entrega']}</font>**", unsafe_allow_html=True)
                st.markdown(f"  **<font color='#ADD8E6'>Subitem {selected_rubrica['subitem']}: {selected_rubrica['competencia']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    **<font color='#90EE90'>Rubrica {selected_rubrica['item_rubrica']}: {selected_rubrica['rubrica']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    Aplicação no projeto:")
                st.markdown(f"      {selected_rubrica['aplicacao_no_projeto']}")
            else:
                st.info(f"Nenhuma rubrica selecionada ou nenhuma rubrica relacionada à função atual.")
        else:
            st.info("Nenhuma rubrica relacionada à função 'Formatação e Apresentação' encontrada.")


    st.divider()
    st.header("Disciplinas Relacionadas")
    f.show_disciplinas_relacionadas_vri("Formatação e Apresentação")

    f.show_referencias_by_function("Formatação e Apresentação")

    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Formatação e Apresentação": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Conteúdo textual e visual")
        st.markdown("##### Outputs\n- Documento formatado e vídeo")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Estrutura completa do TCC.\n- Apresentação oral clara.")