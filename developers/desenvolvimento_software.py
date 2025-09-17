import streamlit as st
import pandas as pd

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

        st.divider()
        st.header("Rubricas de avaliação relacionadas")
        st.markdown("""# 2. `Entrega 2 - Desenvolvimento`
##### 2.1 	`Colaborativa`
	2.1.1 	Individual: Contribuir para a construção conjunta buscando objetivos comuns
	2.1.2 	Individual: se integrar com o grupo
	2.1.3 	Individual: participar de todas as reuniões
	2.1.4 	Individual: contribuir com trabalhos entregues
	2.1.5 	Individual: Contribuiu entregando o combinado de forma completa
	2.1.6 	Individual: Contribuir com qualidade
	2.1.7 	Individual: Ser pontual, respeitando o prazo/tempo combinado
##### 2.2 	`Comunicação / Linguagem`
	2.2.1 	Empregar habilidades para comunicar-se utilizando as variadas linguagens
	2.2.2 	Garantir que a estruturação da escrita geral facilita a compreensão.
	2.2.3 	Garantir que a escrita não contém erros de ortografia
	2.2.4 	Garantir que a escrita não contém erros de gramática
	2.2.5 	Garantir que a escrita não contém erros de pontuação
	2.2.6 	Criar estratégia para que o texto seja organizado
	2.2.7 	Criar estratégia para que o texto seja coerente, sem trechos incoerentes
	2.2.8 	Criar estratégia para que o texto apresente sentido
	2.2.9 	Criar estratégia para que o texto seja bem estruturado
	2.2.10 	Criar estratégia para que o texto tenha sempre articulação entre as partes
	2.2.11 	Apresentar propositalmente sentido e articulação na integralidade do texto
	2.2.12 	Escrever todas as citações e referências bibliográficas na norma ABNT
##### 2.3 	`Considerações finais`
	2.3.1 	Apresentar claramente as contribuições do trabalho realizado
	2.3.2 	Apresentar claramente as limitações do trabalho realizado
	2.3.3 	Apresentar de forma completa as contribuições ou limitações do trabalho
##### 2.4 	`Inovação`
	2.4.1 	Empregar habilidades e estratégias para criar soluções profissionais inovadoras
	2.4.2 	Contribuir com responsabilidade para inovar (no processo de TCC)
	2.4.3 	Contribuir com responsabilidade na análise de dados da solução/pesquisa (no processo de TCC)
	2.4.4 	Buscar inovações de formatação (no processo de TCC)
	2.4.5 	Buscar inovações de escrita  no processo de TCC
	2.4.6 	Buscar inovações na relação entre as pessoas
	2.4.7 	Buscar inovar na análise de dados do trabalho que foi proposto, a partir conhecimento adquirido durante o curso
	2.4.8 	Assumir riscos
	2.4.9 	Compreender o risco como tentativa de inovar, independente de sucesso
	2.4.10 	Se comprometer mais com os riscos para testar uma inovação para o trabalho
##### 2.5 	`Investigativa`
	2.5.1 	Empregar habilidades para conciliar a teoria acadêmica com problema real
	2.5.2 	Restringir, Não deixar abrangente
	2.5.3 	O grupo escolheu o método adequado e/ou combinou alguns métodos disponíveis
##### 2.6 	`Profissional / Referencial teórico`
	2.6.1 	Relacionar conhecimentos desenvolvidos com o curso
	2.6.2 	Empregar habilidades de relacionar os conhecimentos desenvolvidos no curso com o campo profissional
	2.6.3 	Não fugir do tema
	2.6.4 	Não ser redundante com a parte introdutória do texto
	2.6.5 	Escolher e/ou combinar métodos adequados disponíveis
	2.6.6 	Pesquisar além da literatura apresentada em fontes confiáveis (Pesquisar mais sobre o assunto)
	2.6.7 	Dar contribuições ao texto original (ampliar compreensão sobre o tema)
	2.6.8 	Adaptar os métodos a partir da sua capacidade de análise, criação e ajustes a uma realidade apresentada
	2.6.9 	Evidenciar compreensão do método como um caminho para um fim determinado.
##### 2.7 	`Resolução de problemas / Objetivo geral / Objetivos específicos / Desenvolvimento`
	2.7.1 	Empregar habilidades para entender e resolver problemas de um cenário profissional
##### 2.8 	`Resolução de problemas / Objetivo geral / Objetivos específicos /Desenvolvimento`
	2.8.1 	Desenvolver o problema solucionado (justificativa)
	2.8.2 	Descrever os desafios para a solução
	2.8.3 	Descrever claramente os objetivos
	2.8.4 	Distinguir os objetivos gerais dos específicos
	2.8.5 	Deve haver mais de um objetivo complementar
	2.8.6 	Apresentar um objetivo geral
	2.8.7 	Apresentar múltiplos objetivos complementares
	2.8.8 	Apresentar os problemas de forma abrangente (não parte dele)
	2.8.9 	Apresentar claramente e completa as limitações do trabalho realizado
	2.8.10 	Apresentar de forma clara e completa as contribuições do trabalho realizado
	2.8.11 	Apresentar clareza na formulação do problema/solução científica
	2.8.12 	Apresentar clareza no desenvolvimento do problema/solução científica
##### 2.9 	`Resultados / Discussão dos dados`
	2.9.1 	Analisar os resultadosà luz do referencial teórico
	2.9.2 	Apresentar os resultados
##### 2.10 	`Tecnológica`
	2.10.1 	Usar tecnologia para solucionar problemas""")

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
        st.markdown("##### Inputs\n- Requisitos da Proposta")
        st.markdown("##### Outputs\n- Código e Artefatos visuais")