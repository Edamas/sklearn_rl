import streamlit as st
import pandas as pd

def render_desenvolvimento_software():
    st.title("Desenvolvimento de Software")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão define as linguagens, servidor, versionamento e frameworks, focando na materialização do produto e na geração de artefatos palpáveis (prints, vídeos, etc.).")
        st.divider()

        st.header("Visão Geral do Software")
        st.markdown("O software é a materialização do agente de IA autônomo, permitindo a interação com o ambiente Scikit-learn e a visualização dos resultados. Ele é construído para ser modular, escalável e de fácil manutenção.")
        st.divider()

        st.header("Instalação e Configuração")
        st.markdown("Para configurar o ambiente de desenvolvimento e executar o projeto, siga os passos abaixo:")
        install_cols = st.columns(2)
        with install_cols[0]:
            st.markdown("<h5 style='color: #1E90FF;'><b>1. Clonar o Repositório</b></h5>", unsafe_allow_html=True)
            st.code("git clone https://github.com/seu-usuario/sklearn_rl.git")
            st.code("cd sklearn_rl")
        with install_cols[1]:
            st.markdown("<h5 style='color: #1E90FF;'><b>2. Criar Ambiente Virtual</b></h5>", unsafe_allow_html=True)
            st.code("python -m venv venv")
            st.code("source venv/bin/activate  # Linux/macOS")
            st.code("venv\Scripts\activate  # Windows")
        
        install_cols_2 = st.columns(2)
        with install_cols_2[0]:
            st.markdown("<h5 style='color: #1E90FF;'><b>3. Instalar Dependências</b></h5>", unsafe_allow_html=True)
            st.code("pip install -r requirements.txt")
        with install_cols_2[1]:
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

        st.header("Referências e Fontes")
        st.markdown("- N/A")

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
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Requisitos da Proposta")
        st.markdown("##### Outputs\n- Código e Artefatos visuais")