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
        st.header("Relatório e Estilos ABNT")

        st.subheader("Conteúdo do Relatório")
        df_relatorio = pd.read_csv(st.session_state['files']['relatorio_1_projeto'], sep='\t', engine='python')

        # Add hierarchical numbering and a boolean field for numbering display
        df_relatorio['num_hierarquica'] = df_relatorio['secao_id'] + '.' + df_relatorio['paragrafo_id'].astype(str) + '.' + df_relatorio['linha_id'].astype(str)
        df_relatorio['exibir_num'] = True # Default to True, can be changed by user if needed

        st.info("Selecione uma linha do relatório para ver os detalhes.")
        selected_index_relatorio = f.df_select_rows(df_relatorio, selection_mode='single-row', key="relatorio_selector")

        if selected_index_relatorio is not None and selected_index_relatorio in df_relatorio.index:
            selected_linha_relatorio = df_relatorio.loc[selected_index_relatorio]
            st.subheader("Ficha da Linha do Relatório")
            st.markdown(f.get_card_style(), unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card'>
                <div class='card-body'>
                    <h5 class='card-title'><font color='#FFD700'>Seção:</font> {selected_linha_relatorio['secao_id']}</font></h5>
                    <p class='card-text'><font color='#ADD8E6'>Parágrafo ID:</font> {selected_linha_relatorio['paragrafo_id']}</p>
                    <p class='card-text'><font color='#90EE90'>Linha ID:</font> {selected_linha_relatorio['linha_id']}</p>
                    <p class='card-text'><font color='#FFA07A'>Texto:</font> {selected_linha_relatorio['texto']}</p>
                    <p class='card-text'><font color='#4682B4'>Estilo ID:</font> {selected_linha_relatorio['estilo_id']}</p>
                    <p class='card-text'><font color='#FFD700'>Formato Específico:</font> {selected_linha_relatorio['formato_especifico']}</p>
                    <p class='card-text'><font color='#ADD8E6'>Observações:</font> {selected_linha_relatorio['observacoes']}</p>
                    <p class='card-text'><font color='#90EE90'>Numeração Hierárquica:</font> {selected_linha_relatorio['num_hierarquica']}</p>
                    <p class='card-text'><font color='#FFA07A'>Exibir Numeração:</font> {selected_linha_relatorio['exibir_num']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Nenhuma linha do relatório selecionada.")

        st.subheader("Estilos ABNT")
        df_formatacao = pd.read_csv(st.session_state['files']['formatacao'], sep='\t', engine='python')

        st.info("Selecione um estilo para ver os detalhes.")
        selected_index_formatacao = f.df_select_rows(df_formatacao, selection_mode='single-row', key="formatacao_selector")

        if selected_index_formatacao is not None and selected_index_formatacao in df_formatacao.index:
            selected_estilo = df_formatacao.loc[selected_index_formatacao]
            st.subheader("Ficha do Estilo")
            st.markdown(f.get_card_style(), unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card'>
                <div class='card-body'>
                    <h5 class='card-title'><font color='#FFD700'>Estilo ID:</font> {selected_estilo['estilo_id']}</font></h5>
                    <p class='card-text'><font color='#ADD8E6'>Nome do Estilo:</font> {selected_estilo['nome_estilo']}</p>
                    <p class='card-text'><font color='#90EE90'>Fonte:</font> {selected_estilo['fonte_nome']} ({selected_estilo['fonte_tamanho']}pt)</p>
                    <p class='card-text'><font color='#FFA07A'>Alinhamento:</font> {selected_estilo['alinhamento']}</p>
                    <p class='card-text'><font color='#4682B4'>Recuo Primeira Linha:</font> {selected_estilo['recuo_primeira_linha']}</p>
                    <p class='card-text'><font color='#FFD700'>Espaçamento Antes:</font> {selected_estilo['espacamento_antes']}</p>
                    <p class='card-text'><font color='#ADD8E6'>Espaçamento Depois:</font> {selected_estilo['espacamento_depois']}</p>
                    <p class='card-text'><font color='#90EE90'>Espaçamento Entre Linhas:</font> {selected_estilo['espacamento_entre_linhas']}</p>
                    <p class='card-text'><font color='#FFA07A'>Negrito:</font> {selected_estilo['negrito']}</p>
                    <p class='card-text'><font color='#4682B4'>Itálico:</font> {selected_estilo['italico']}</p>
                    <p class='card-text'><font color='#FFD700'>Sublinhado:</font> {selected_estilo['sublinhado']}</p>
                    <p class='card-text'><font color='#ADD8E6'>Cor da Fonte:</font> {selected_estilo['cor_fonte']}</p>
                    <p class='card-text'><font color='#90EE90'>Observações:</font> {selected_estilo['observacoes']}</p>
                    <p class='card-text'><font color='#FFA07A'>Código Visual:</font> {selected_estilo['codigo_visual']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Nenhum estilo selecionado.")

        st.divider()
        st.header("Visualização do Relatório Formatado")

        # Read the dataframes again to ensure they are up-to-date
        df_relatorio = pd.read_csv(st.session_state['files']['relatorio_1_projeto'], sep='\t', engine='python')
        df_formatacao = pd.read_csv(st.session_state['files']['formatacao'], sep='\t', engine='python')

        # Merge the dataframes to get the visual code for each line
        df_merged = pd.merge(df_relatorio, df_formatacao, on='estilo_id', how='left')

        # Generate the formatted report content
        formatted_report_content = ""
        for index, row in df_merged.iterrows():
            text_content = row['texto']
            visual_code_template = row['codigo_visual']
            num_hierarquica = row['num_hierarquica']
            exibir_num = row['exibir_num']

            # Apply specific formatting (bold, italic, underline) if present in 'formato_especifico'
            # This is a simplified example, a more robust solution would parse the 'formato_especifico' field
            if 'negrito' in str(row['formato_especifico']).lower():
                text_content = f"<b>{text_content}</b>"
            if 'italico' in str(row['formato_especifico']).lower():
                text_content = f"<i>{text_content}</i>"
            if 'sublinhado' in str(row['formato_especifico']).lower():
                text_content = f"<u>{text_content}</u>"

            # Add hierarchical numbering if 'exibir_num' is True
            if exibir_num:
                text_content = f"{num_hierarquica} {text_content}"

            # Replace the placeholder in codigo_visual with the actual text content
            # Assuming the visual_code_template has a placeholder like '</span>' at the end
            if visual_code_template and '</span>' in visual_code_template:
                formatted_line = visual_code_template.replace('</span>', f'{text_content}</span>')
            else:
                formatted_line = f"<div>{text_content}</div>" # Fallback if no visual code or placeholder

            formatted_report_content += formatted_line + "<br>" # Add a line break for readability

        st.markdown(formatted_report_content, unsafe_allow_html=True)

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