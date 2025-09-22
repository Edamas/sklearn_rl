import streamlit as st
import pandas as pd
import functions as f

def render_documentacao_software():
    st.title("Documentação de Software")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("No time Developers, o membro com foco acadêmico converte a tecnicidade dos desenvolvedores e as especificidades do projeto em termos escritos e encorpados. A revisão final é feita pelo acadêmico do time Academics, mas cada membro revisa a parte do seu 'par' de setor, aplicando suas competências em processos de aplicação diferentes, mas da mesma natureza.")
        

        st.subheader("Rubricas relacionadas")
        # Obter o DataFrame filtrado da função
        df_filtered_rubricas = f.get_rubricas_by_function_score_8("Documentação de Software")

        if not df_filtered_rubricas.empty:
            # Preparar o DataFrame para exibição interativa
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = f.df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_documentacao_de_software")

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
            st.info("Nenhuma rubrica relacionada à função 'Documentação de Software' encontrada.")


        st.divider()
        st.header("Disciplinas Relacionadas")
        f.show_disciplinas_relacionadas_vri("Documentação de Software")

        f.show_referencias_by_function("Documentação de Software")

    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Documentação de Software": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Documentação técnica dos desenvolvedores\n- Especificações do projeto\n- Documentação de bibliotecas e frameworks utilizados\n- Requisitos do TCC e da proposta")
        st.markdown("##### Outputs\n- Documentação técnica clara e concisa\n- Manuais de uso (se aplicável)\n- Relatórios de progresso da documentação\n- Glossário de termos técnicos")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("##### Requisitos do TCC\n- Elaboração de documentação técnica que suporte o TCC.\n- Clareza e precisão na descrição de funcionalidades e arquitetura.\n- Conformidade com padrões de documentação (se houver).")
        st.markdown("##### Requisitos da Proposta (sobre o agente)\n- Documentar a arquitetura e implementação do agente de IA.\n- Descrever os algoritmos e modelos utilizados pelo agente.\n- Registrar os resultados dos experimentos e análises do agente.")
