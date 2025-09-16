import streamlit as st
import pandas as pd

def render_gestao_proposta():
    st.title("Gestão da Proposta")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("Esta gestão traça as estratégias do projeto, desde sua concepção (coleta de ~300 propostas, seleção de 8 finalistas e votação) até o planejamento tático (backlog, sprints) e garantia do escopo, em comunicação direta com a Gestão do TCC.")
        st.divider()

        st.header("Processo de Seleção da Proposta")
        st.markdown("""A metodologia de Design Thinking foi central no processo de seleção da proposta. A fase de **Divergência** (Imersão e Ideação) envolveu um brainstorming que gerou cerca de 300 ideias iniciais de projetos. Em seguida, na fase de **Convergência** (Prototipação e Teste), as propostas foram avaliadas e pontuadas em múltiplos critérios (e.g., Apelo, Inovação, Viabilidade), resultando na seleção das 8 finalistas e na posterior eleição do projeto atual por votação.\n\nAbaixo, os detalhes da proposta finalista, incluindo os critérios de avaliação que não fazem parte do TCC Model Canvas ou Project Charter:
""")

        proposal_details_data = {
            "Atributo": [
                "Viabilidade 📅", "Apelo 🚀", "Tendência 🔥", "Inovação 💡", 
                "Contrib. Social 🤝", "Relação com BCD 🤖", "Adequação ao TCC 🎓", 
                "Potencial 💰", "Facilidade de Aquisição 📥", "Qualidade dos Dados ✨", 
                "Nível de Atualização 🕒", "Peso+", "Peso-", "Outras Vantagens", 
                "Outras Desvantagens", "Observações, Links, Vídeos, etc.", "Datasets", 
                "Algoritmos", "Objetivo Geral", "Objetivos Específicos", "Premissas", 
                "Restrições", "Fornecedores", "Atividades Chave", "Recursos Chave", 
                "Relacionamento", "Canais", "Segmentos de Clientes", "Estrutura (TCC)"
            ],
            "Valor": [
                "2", "5", "5", "5", "4", "5", "5", "5", "5", "4", "5", "5", "-5",
                "Define o estado da arte na automação do aprendizado de máquina. Altíssimo potencial de publicação científica e de criação de propriedade intelectual.",
                "A complexidade teórica e de implementação de um ambiente de RL é extremamente alta. O treinamento do agente pode ser computacionalmente caro.",
                "",
                "Datasets públicos do OpenML ou Kaggle, servindo como os \"ambientes\" onde o agente de RL será treinado e avaliado.",
                "Aprendizado por Reforço (Q-Learning, PPO), Meta-Aprendizagem, Scikit-learn.",
                "Investigar a viabilidade de utilizar Aprendizado por Reforço para a construção autônoma de fluxos de trabalho de aprendizado de máquina.",
                "1. Modelar o processo como um ambiente de RL. 2. Implementar um agente de RL. 3. Treinar o agente para otimizar métricas. 4. Comparar a performance com baselines padrão.",
                "Um agente de RL pode aprender estratégias de modelagem que generalizam para diferentes datasets. A análise da política aprendida pode gerar insights sobre a própria ciência de dados.",
                "Este é um projeto de P&D de alta complexidade e risco. O espaço de estados e ações é vasto, o que pode tornar o treinamento do agente muito lento.",
                "Bibliotecas Python (Gymnasium, Stable-Baselines3, Scikit-learn, Streamlit).",
                "Pesquisa e Desenvolvimento em RL; Design do ambiente de RL (estados, ações, recompensas); Implementação do agente de RL; Análise comparativa dos resultados.",
                "O ambiente de RL (IP principal); O agente de RL treinado (a \"política\"); A plataforma de visualização.",
                "Publicação dos resultados em conferências de IA (NeurIPS, ICML); Colaboração com a comunidade de código aberto.",
                "Artigos científicos; Repositório de código aberto (GitHub).",
                "Pesquisadores de automação de aprendizado de máquina e RL; Cientistas de dados sênior.",
                "Introdução (Apresentação da Teoria DPM), Metodologia (Modelagem do Ambiente de RL, Arquitetura do Agente), Resultados (Análise da Política Aprendida, Comparação com Baselines), Conclusão."
            ]
        }
        st.dataframe(pd.DataFrame(proposal_details_data), use_container_width=True)
        st.divider()

        # --- Project Charter ---
        st.header("Project Charter")
        st.markdown("O Termo de Abertura do Projeto formaliza seu início e confere ao gerente a autoridade para aplicar recursos nas atividades.")
        
        row1 = st.columns(3)
        row2 = st.columns(3)

        with row1[0]:
            st.markdown("<h5 style='color: #4682B4;'><b>Justificativa</b></h5>", unsafe_allow_html=True)
            st.markdown("- Preencher a lacuna na literatura de AutoML com uma abordagem baseada em Aprendizado por Reforço.")
        with row1[1]:
            st.markdown("<h5 style='color: #4682B4;'><b>Objetivos Mensuráveis</b></h5>", unsafe_allow_html=True)
            st.markdown("- Implementar um agente de RL funcional.\n- Superar o baseline aleatório.\n- Publicar os resultados.")
        with row1[2]:
            st.markdown("<h5 style='color: #4682B4;'><b>Requisitos de Alto Nível</b></h5>", unsafe_allow_html=True)
            st.markdown("- Plataforma de visualização (Streamlit).\n- Ambiente de RL customizado.\n- Mínimo de 3 estratégias de agentes.")
        
        with row2[0]:
            st.markdown("<h5 style='color: #B22222;'><b>Riscos Gerais</b></h5>", unsafe_allow_html=True)
            st.markdown("- Alta complexidade teórica.\n- Custo computacional elevado.\n- Vasto espaço de ações/estados.")
        with row2[1]:
            st.markdown("<h5 style='color: #4682B4;'><b>Cronograma de Marcos</b></h5>", unsafe_allow_html=True)
            st.markdown("- Entrega 1: Projeto\n- Entrega 2: Desenvolvimento\n- Entrega 3: Artigo Final\n- Apresentação para a Banca")
        with row2[2]:
            st.markdown("<h5 style='color: #4682B4;'><b>Stakeholders Chave</b></h5>", unsafe_allow_html=True)
            st.markdown("- Orientador\n- Membros do Grupo\n- Banca Avaliadora\n- Comunidade Acadêmica")
        st.divider()

        # --- TCC Model Canvas ---
        st.header("TCC Model Canvas")
        st.markdown("Para selecionar a proposta finalista, o Business Model Canvas foi adaptado para o contexto acadêmico. Essa ferramenta permitiu uma análise estruturada e comparativa de cada proposta. Abaixo estão os blocos do canvas para a proposta vencedora.")
        
        top_row = st.columns([2, 3, 2, 3, 2])
        bottom_row = st.columns([5, 5])

        with top_row[0]:
            st.markdown("<h5 style='color: #FF4B4B;'><b>Parcerias-Chave</b></h5>", unsafe_allow_html=True)
            st.markdown("- Univesp (orientação)\n- Comunidade Python/Scikit-learn")
        
        with top_row[1]:
            st.markdown("<h5 style='color: #1E90FF;'><b>Atividades-Chave</b></h5>", unsafe_allow_html=True)
            st.markdown("- P&D em RL\n- Design do ambiente de RL\n- Implementação do agente")
            st.markdown("<h5 style='color: #1E90FF; margin-top: 1rem;'><b>Recursos-Chave</b></h5>", unsafe_allow_html=True)
            st.markdown("- O ambiente de RL (IP principal)\n- O agente de RL treinado")

        with top_row[2]:
            st.markdown("<h5 style='color: #32CD32;'><b>Proposta de Valor</b></h5>", unsafe_allow_html=True)
            st.markdown("Contribuir para o estado da arte em AutoML, provando que um agente autônomo pode aprender a utilizar as ferramentas do Scikit-learn de forma eficiente.")

        with top_row[3]:
            st.markdown("<h5 style='color: #FFD700;'><b>Relacionamento</b></h5>", unsafe_allow_html=True)
            st.markdown("- Publicação dos resultados\n- Colaboração com a comunidade")
            st.markdown("<h5 style='color: #FFD700; margin-top: 1rem;'><b>Canais</b></h5>", unsafe_allow_html=True)
            st.markdown("- Artigos científicos\n- Repositório GitHub")

        with top_row[4]:
            st.markdown("<h5 style='color: #BA55D3;'><b>Segmentos de Clientes</b></h5>", unsafe_allow_html=True)
            st.markdown("- Pesquisadores de AutoML e RL\n- Cientistas de dados sênior")

        with bottom_row[0]:
            st.markdown("<h5 style='color: #F08080;'><b>Estrutura de Custos</b></h5>", unsafe_allow_html=True)
            st.markdown("- Tempo de desenvolvimento dos integrantes\n- Custo de recursos computacionais (se aplicável)")

        with bottom_row[1]:
            st.markdown("<h5 style='color: #20B2AA;'><b>Fontes de Receita</b></h5>", unsafe_allow_html=True)
            st.markdown("- Nota de aprovação no TCC\n- Potencial de publicação\n- Criação de portfólio e propriedade intelectual")
        
        st.divider()

        st.header("Referências e Fontes")
        st.markdown("""-   **PROJECT MANAGEMENT INSTITUTE. *Um guia do conhecimento em gerenciamento de projetos (Guia PMBOK®)*. 6. ed. Newtown Square, PA: Project Management Institute, 2017.**
    -   **Aplicação:** Utilizado para estruturar o Project Charter, definindo escopo, planejamento e cronograma.

-   **OSTERWALDER, Alexander; PIGNEUR, Yves. *Business Model Generation: inovação em modelos de negócios*. Rio de Janeiro: Alta Books, 2011.**
    -   **Aplicação:** Adaptado para o \"TCC Model Canvas\" para avaliar e comparar propostas de forma estruturada.

-   **BROWN, Tim. *Design thinking*. Rio de Janeiro: Elsevier, 2010.**
    -   **Aplicação:** Utilizado no processo de seleção da proposta, com Divergência (brainstorming) e Convergência (avaliação e pontuação).
""")

    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Gestão da Proposta": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Objetivos do TCC\n- Histórico da seleção")
        st.markdown("##### Outputs\n- Backlog e Sprints\n- Planejamento do escopo")