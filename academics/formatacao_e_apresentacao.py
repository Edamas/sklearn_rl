import streamlit as st
import pandas as pd
import functions as f

def render_formatacao_e_apresentacao():
    st.title("Formatação e Apresentação")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("No contexto acadêmico, este membro é responsável pela formatação do Word, criação do template de normas ABNT, e pela 'montagem do trabalho', garantindo a 'língua', 'comunicação' e 'coesão'. Também administra o vídeo de apresentação final, recebendo contribuições do grupo e gerando o produto final (vídeo e link para capa do trabalho).")
        
        st.divider()
        st.header("Gestão de Relatórios")
        st.markdown("Este módulo representa um esforço de inovação nos processos de relatórios acadêmicos, conforme as rubricas estabelecidas. Nosso objetivo é otimizar a geração de documentos, buscando performance e resultados que permitam ao próprio aplicativo gerar PDFs totalmente em conformidade com as normas ABNT, reduzindo a carga manual de formatação.")



        st.subheader("Estilos ABNT")
        df_formatacao_display = pd.read_csv(st.session_state['files']['formatacao'], sep='\t')

        selected_index_formatacao = f.df_select_rows(df_formatacao_display, selection_mode='single-row', key="formatacao_selector", prompt=None)

        if selected_index_formatacao is not None and selected_index_formatacao in df_formatacao_display.index:
            selected_estilo = df_formatacao_display.loc[selected_index_formatacao]
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
            pass # Removido: st.info("Nenhum estilo selecionado.")

        st.divider()

        # --- Relatório 1 --- 
        with st.expander("Relatório 1 - Entrega", expanded=False):
            st.markdown("Este relatório, correspondente à primeira entrega, detalha a estrutura inicial do projeto, a fundamentação teórica e a metodologia proposta. Ele serve como base para as próximas etapas de desenvolvimento e experimentação. Prazo de entrega: até 23:59 do dia 21/09/2024.")
            
            st.subheader("Conteúdo do Relatório 1")
            df_relatorio_display_1 = pd.read_csv(st.session_state['files']['relatorio_1_projeto'], sep='\t')
            selected_index_relatorio_1 = f.df_select_rows(df_relatorio_display_1, selection_mode='single-row', key="relatorio_selector_1", prompt=None)

            if selected_index_relatorio_1 is not None and selected_index_relatorio_1 in df_relatorio_display_1.index:
                selected_linha_relatorio_1 = df_relatorio_display_1.loc[selected_index_relatorio_1]
                st.subheader("Ficha da Linha do Relatório 1")
                st.markdown(f.get_card_style(), unsafe_allow_html=True)
                st.markdown(f"""
                <div class='card'>
                    <div class='card-body'>
                        <h5 class='card-title'><font color='#FFD700'>Seção:</font> {selected_linha_relatorio_1['secao_id']}</font></h5>
                        <p class='card-text'><font color='#ADD8E6'>Parágrafo ID:</font> {selected_linha_relatorio_1['paragrafo_id']}</p>
                        <p class='card-text'><font color='#90EE90'>Linha ID:</font> {selected_linha_relatorio_1['linha_id']}</p>
                        <p class='card-text'><font color='#FFA07A'>Texto:</font> {selected_linha_relatorio_1['texto']}</p>
                        <p class='card-text'><font color='#4682B4'>Estilo ID:</font> {selected_linha_relatorio_1['estilo_id']}</p>
                        <p class='card-text'><font color='#FFD700'>Formato Específico:</font> {selected_linha_relatorio_1['formato_especifico']}</p>
                        <p class='card-text'><font color='#ADD8E6'>Observações:</font> {selected_linha_relatorio_1['observacoes']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                pass # Removido: st.info("Nenhuma linha do Relatório 1 selecionada.")

            show_html_relatorio_1 = st.toggle("Visualizar Relatório 1 em HTML", key="toggle_relatorio_1", value=False)
            if show_html_relatorio_1:
                df_relatorio = pd.read_csv(st.session_state['files']['relatorio_1_projeto'], sep='\t')
                df_formatacao = pd.read_csv(st.session_state['files']['formatacao'], sep='\t')
                df_completo = pd.merge(df_relatorio, df_formatacao, on='estilo_id', how='left')
                
                current_secao = None
                current_paragrafo = None

                for index, row in df_completo.iterrows():
                    secao_id = row['secao_id']
                    paragrafo_id = row['paragrafo_id']
                    texto = row['texto']
                    codigo_visual = row['codigo_visual']

                    if secao_id != current_secao:
                        st.subheader(secao_id.replace('_', ' ').title()) 
                        current_secao = secao_id
                        current_paragrafo = None 

                    if paragrafo_id != current_paragrafo:
                        current_paragrafo = paragrafo_id

                    if pd.notna(codigo_visual):
                        codigo_visual_ajustado = codigo_visual.replace('color: black;', 'color: white;')
                        if row['estilo_id'] == 9:
                            codigo_visual_ajustado = codigo_visual_ajustado.replace('text-align: justify;', 'text-align: right;')
                        codigo_visual_ajustado = codigo_visual_ajustado.replace('<span', '<p').replace('</span>', '</p>')
                        formatted_line = codigo_visual_ajustado.replace('</p>', f"{texto}</p>")
                        st.markdown(formatted_line, unsafe_allow_html=True)
                    else:
                        st.write(texto) 

        st.divider()

        # --- Relatório 2 --- 
        with st.expander("Relatório 2 - Desenvolvimento", expanded=False):
            st.markdown("Este relatório aborda o desenvolvimento do protótipo, os experimentos iniciais e a análise dos primeiros resultados. Ele reflete o progresso técnico e as decisões de implementação. Prazo de entrega: até 23:59 do dia 02/11/2025.")
            
            st.subheader("Conteúdo do Relatório 2")
            st.info("Conteúdo do Relatório 2 será exibido aqui.")

            show_html_relatorio_2 = st.toggle("Visualizar Relatório 2 em HTML", key="toggle_relatorio_2", value=False)
            if show_html_relatorio_2:
                st.markdown("<h3>Relatório 2 em HTML (conteúdo placeholder)</h3>", unsafe_allow_html=True)

        st.divider()

        # --- Entrega 3 --- 
        with st.expander("Entrega 3 - Banca Examinadora", expanded=False):
            st.markdown("Esta entrega final consolida todo o trabalho realizado, incluindo a análise de resultados, discussões e considerações finais. É o documento que será apresentado à banca examinadora. Prazo de entrega: até 23:59 do dia 16/11/2025.")
            
            st.subheader("Conteúdo da Entrega 3")
            st.info("Conteúdo da Entrega 3 será exibido aqui.")

            show_html_entrega_3 = st.toggle("Visualizar Entrega 3 em HTML", key="toggle_entrega_3", value=False)
            if show_html_entrega_3:
                st.markdown("<h3>Entrega 3 em HTML (conteúdo placeholder)</h3>", unsafe_allow_html=True)

        st.divider()

        # --- Apresentação em Vídeo (Movido para o final) ---
        with st.expander("Apresentação em Vídeo", expanded=False):
            st.markdown("Este vídeo é a apresentação final do projeto, consolidando os principais resultados e contribuições. Ele serve como um resumo visual e dinâmico do trabalho realizado.")
            if 'roteiro_video' in st.session_state['files']:
                with open(st.session_state['files']['roteiro_video'], 'r', encoding='utf-8') as f_file:
                    roteiro_video = f_file.read()
                st.subheader("Roteiro do Vídeo")
                st.markdown(roteiro_video)
            else:
                st.error("Caminho para 'roteiro_video' não encontrado em st.session_state['files'].")

        st.divider()
        st.header("Registro de Atividades")
        f.show_registro_atividades_by_function("Formatação e Apresentação")
        st.divider()
        st.header("Cronograma e Entregas")
        f.show_cronograma_by_function("Formatação e Apresentação")
        st.divider()
        st.header("Rubricas relacionadas")
        df_filtered_rubricas = f.get_rubricas_by_function_score_8("Formatação e Apresentação")

        if not df_filtered_rubricas.empty:
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = f.df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_formatacao_e_apresentacao")

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
        st.markdown("##### Inputs\n- `relatorio_1_projeto.tsv` (Conteúdo do Relatório)\n- `formatacao.tsv` (Estilos ABNT)\n- `roteiro_video.md` (Roteiro do Vídeo)\n- `cronograma.tsv` (Prazos)\n- Rubricas e Disciplinas relacionadas (Dados de referência)")
        st.markdown("##### Outputs\n- Relatórios formatados (HTML)\n- Fichas de detalhes (Relatório, Estilos, Rubricas)\n- Informações de Cronograma e Atividades")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Geração de relatórios ABNT-compatíveis.\n- Visualização interativa de conteúdo e estilos.\n- Apresentação clara de prazos e rubricas.\n- Capacidade de gerar PDF (futuro).")