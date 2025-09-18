import streamlit as st
import pandas as pd
import functions as f

def render_desenvolvimento_software():
    st.title("Desenvolvimento de Software")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("No time Developers, o desenvolvedor é o programador ou gestor de programação. Ele define linguagens, servidor, versionamento, colaboração, framework e estratégia de programação. Após receber requisitos e escopo, foca na materialização do produto, desde a infraestrutura (em diálogo com a pesquisa) até a geração de artefatos palpáveis (prints, gráficos, tabelas, arquivos, vídeos).")
        st.divider()

        st.header("Visão Geral do Software")
        st.markdown("O software é a materialização do agente de IA autônomo, permitindo a interação com o ambiente Scikit-learn e a visualização dos resultados. Ele é construído para ser modular, escalável e de fácil manutenção.")
        st.divider()

        st.header("Instalação e Configuração")
        st.markdown("Para configurar o ambiente de desenvolvimento e executar o projeto, siga os passos abaixo:")
        install_cols = st.columns(2)
        with install_cols[0]:
            st.markdown("<h5 style='color: #1E90FF;'><b>1. Clonar o Repositivo</b></h5>", unsafe_allow_html=True)
            st.code("git clone https://github.com/seu-usuario/sklearn_rl.git")
            st.code("cd sklearn_rl")
        with install_cols[1]:
            st.markdown("<h5 style='color: #1E90FF;'><b>2. Criar Ambiente Virtual</b></h5>", unsafe_allow_html=True)
            st.code("python -m venv venv")
            st.code("source venv/bin/activate  # Linux/macOS")
            st.code("venv\\Scripts\\activate  # Windows")
        
        install_cols_2 = st.columns(2)
        with install_cols_2[0]:
            st.markdown("<h5 style='color: #1E90FF;'><b>3. Instalar Dependências</b></h5>", unsafe_allow_html=True)
            st.code("pip install -r requirements.txt")
        with install_cols[1]:
            st.markdown("<h5 style='color: #1E90FF;'><b>4. Executar a Aplicação</b></h5>", unsafe_allow_html=True)
            st.code("streamlit run app.py")
        st.divider()

        st.header("Estrutura do Projeto")
        st.markdown("O projeto segue uma estrutura modular para garantir organização e manutenibilidade:")
        structure_cols = st.columns(2)
        with structure_cols[0]:
            st.markdown("<h5 style='color: #FF8C00;'><b>Arquivos Principais</b></h5>", unsafe_allow_html=True)
            st.markdown("- `app.py`: Ponto de entrada da aplicação Streamlit.\n- `menu.py`: Define a navegação e estrutura do menu.\n- `requirements.txt`: Lista de dependências do projeto.")
        with structure_cols[1]:
            st.markdown("<h5 style='color: #FF8C00;'><b>Pastas de Módulos</b></h5>", unsafe_allow_html=True)
            st.markdown("- `A_inputs/`: Módulos para carregamento e preparação de dados.\n- `B_input_config/`: Configurações de features e engenharia de atributos.\n- `C_agent_config/`: Configurações e ações do agente de RL.\n- `D_training/`: Módulos relacionados ao treinamento do agente.\n- `E_results/`: Módulos para visualização e análise de resultados.")
        st.divider()

        st.header("Tecnologias Utilizadas")
        st.markdown("As principais tecnologias e bibliotecas empregadas no desenvolvimento do projeto incluem:")
        tech_cols = st.columns(3)
        with tech_cols[0]:
            st.markdown("<h5 style='color: #32CD32;'><b>Linguagem</b></h5>", unsafe_allow_html=True)
            st.markdown("- Python")
        with tech_cols[1]:
            st.markdown("<h5 style='color: #32CD32;'><b>Framework Web</b></h5>", unsafe_allow_html=True)
            st.markdown("- Streamlit")
        with tech_cols[2]:
            st.markdown("<h5 style='color: #32CD32;'><b>Bibliotecas ML/Dados</b></h5>", unsafe_allow_html=True)
            st.markdown("- Scikit-learn\n- Pandas\n- NumPy")
        st.divider()

        st.header("Diretrizes e Boas Práticas")
        st.markdown("Para garantir a qualidade e manutenibilidade do código, seguimos as seguintes diretrizes:")
        guideline_cols = st.columns(2)
        with guideline_cols[0]:
            st.markdown("<h5 style='color: #FF4500;'><b>Modularidade</b></h5>", unsafe_allow_html=True)
            st.markdown("- Separar lógica de RL, visualização e manipulação de arquivos.\n- Evitar funções desatualizadas do Streamlit (e.g., `st.beta_*`).")
        with guideline_cols[1]:
            st.markdown("<h5 style='color: #FF4500;'><b>Consistência</b></h5>", unsafe_allow_html=True)
            st.markdown("- Manter coerência entre a visão do agente e o radar/matriz de descoberta.\n- Garantir que gráficos sejam atualizados corretamente em containers/colunas.")
        st.markdown("<h5 style='color: #FF4500; margin-top: 1rem;'><b>Robustez</b></h5>", unsafe_allow_html=True)
        st.markdown("- Nunca depender de caminhos absolutos; usar `os.path.join()`.\n- Documentar todas as funções com docstrings, incluindo tipos de entrada e saída.")
        st.divider()

        st.header("Links Úteis")
        st.markdown("- [Repositório GitHub](https://github.com/Edamas/sklearn_rl)\n- [Aplicação Streamlit Live](https://sklearn-rl.streamlit.app/)")
        st.divider()

        

        st.divider()

        st.header("Histórico de Commits Git")
        st.markdown("Visualize os commits mais recentes do repositório, com detalhes sobre cada alteração.")

        if st.button("Atualizar Commits"):
            st.info("Para atualizar a lista de commits, por favor, solicite ao agente que execute o script `create_git_commits_tsv.py`. Após a execução, recarregue a página para ver as atualizações.")

        # Read commits from TSV file
        try:
            df_commits = pd.read_csv(st.session_state['files']['git_commits'], sep='\t')
        except KeyError:
            st.error("Arquivo 'git_commits.tsv' não encontrado em st.session_state['files']. Certifique-se de que ele está listado em files.tsv.")
            df_commits = pd.DataFrame(columns=['Hash', 'Autor', 'Data', 'Mensagem'])
        except FileNotFoundError:
            st.error("Arquivo 'docs/git_commits.tsv' não encontrado. Por favor, gere o arquivo de commits.")
            df_commits = pd.DataFrame(columns=['Hash', 'Autor', 'Data', 'Mensagem'])

        selected_index_commit = f.df_select_rows(df_commits, selection_mode='single-row', key="commits_selector", prompt=None)

        if selected_index_commit is not None and selected_index_commit in df_commits.index:
            selected_commit = df_commits.loc[selected_index_commit]
            st.subheader("Detalhes do Commit")
            st.markdown(f.get_card_style(), unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card'>
                <div class='card-body'>
                    <h5 class='card-title'><font color='#FFD700'>Hash:</font> {selected_commit['Hash']}</font></h5>
                    <p class='card-text'><font color='#ADD8E6'>Autor:</font> {selected_commit['Autor']}</p>
                    <p class='card-text'><font color='#90EE90'>Data:</font> {selected_commit['Data']}</p>
                    <p class='card-text'><font color='#FFA07A'>Mensagem:</font> {selected_commit['Mensagem']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pass # Removido: st.info("Nenhum commit selecionado.")

        st.divider()
        st.header("Registro de Atividades")
        f.show_registro_atividades_by_function("Desenvolvimento de Software")
        st.divider()
        st.header("Cronograma e Entregas")
        f.show_cronograma_by_function("Desenvolvimento de Software")
        st.divider()
        st.subheader("Rubricas relacionadas")
        # Obter o DataFrame filtrado da função
        df_filtered_rubricas = f.get_rubricas_by_function_score_8("Desenvolvimento de Software")

        if not df_filtered_rubricas.empty:
            # Preparar o DataFrame para exibição interativa
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = f.df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_desenvolvimento_de_software")

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
            st.info("Nenhuma rubrica relacionada à função 'Desenvolvimento de Software' encontrada.")

        st.divider()
        st.header("Disciplinas Relacionadas")
        f.show_disciplinas_relacionadas_vri("Desenvolvimento de Software")

        f.show_referencias_by_function("Desenvolvimento de Software")

    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Desenvolvimento de Software": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Requisitos da Proposta (funcionais e não funcionais)\n- Resultados experimentais da Pesquisa Científica\n- Diretrizes de arquitetura e design\n- Tecnologias e bibliotecas selecionadas (Python, Streamlit, Scikit-learn, Pandas, NumPy)\n- Diretrizes de modularidade, consistência e robustez")
        st.markdown("##### Outputs\n- Código-fonte do agente de IA autônomo\n- Ambiente de desenvolvimento configurado\n- Aplicação Streamlit interativa\n- Artefatos visuais (gráficos, tabelas, vídeos)\n- Repositório GitHub atualizado\n- Versões do protótipo (v1.0, v1.1, etc.)")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("##### Requisitos do TCC\n- Implementação de um protótipo funcional do agente de IA.\n- Geração de artefatos visuais para o TCC (gráficos, tabelas).\n- Conformidade com as diretrizes de apresentação do TCC.")
        st.markdown("##### Requisitos da Proposta (sobre o agente)\n- Desenvolvimento de um agente de IA autônomo capaz de interagir com o Scikit-learn.\n- Modularidade e escalabilidade do código.\n- Manutenibilidade e boas práticas de programação.\n- Capacidade de gerar resultados reproduzíveis.\n- Integração com a plataforma Streamlit para visualização.")