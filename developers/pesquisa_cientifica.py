import streamlit as st
import pandas as pd

def render_pesquisa_cientifica():
    st.title("Pesquisa Científica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão cria as táticas da equipe com embasamento teórico e prático, explorando bibliotecas, analisando dados e conduzindo experimentos em conexão direta com o desenvolvedor.")
        st.divider()

        st.header("Metodologia e Modelos de Agentes")
        st.markdown("A metodologia proposta neste trabalho compreende as seguintes etapas:")
        met_cols = st.columns(3)
        with met_cols[0]:
            st.markdown("<h5 style='color: #4682B4;'><b>1. Seleção de Dados</b></h5>\nEscolha de conjuntos de dados padrão de uso consolidado na literatura de aprendizado de máquina.", unsafe_allow_html=True)
        with met_cols[1]:
            st.markdown("<h5 style='color: #4682B4;'><b>2. Implementação de Agente Autônomo</b></h5>\nConfiguração de um pipeline AutoML que utilize a suíte Scikit-learn.", unsafe_allow_html=True)
        with met_cols[2]:
            st.markdown("<h5 style='color: #4682B4;'><b>3. Configuração Manual</b></h5>\nTreinamento de modelos equivalentes de forma manual, incluindo ajuste de hiperparâmetros.", unsafe_allow_html=True)
        
        met_cols_2 = st.columns(2)
        with met_cols_2[0]:
            st.markdown("<h5 style='color: #4682B4;'><b>4. Comparação de Desempenho</b></h5>\nAplicação de métricas padronizadas, como acurácia, precisão, recall, F1-score, tempo de execução e custo computacional.", unsafe_allow_html=True)
        with met_cols_2[1]:
            st.markdown("<h5 style='color: #4682B4;'><b>5. Discussão dos Resultados</b></h5>\nIdentificação de ganhos, limitações e possíveis melhorias.", unsafe_allow_html=True)
        st.divider()

        st.header("Tipos de Agentes de IA Autônomos")
        st.markdown("Este projeto visa comparar o desempenho de diferentes abordagens para a construção de agentes de IA autônomos na seleção e sequenciamento de ferramentas do Scikit-learn. Serão desenvolvidos e avaliados cinco tipos distintos de agentes:")
        
        agent_cols_1 = st.columns(2)
        with agent_cols_1[0]:
            st.markdown("<h5 style='color: #32CD32;'><b>Agente 1: Aleatório (Baseline)</b></h5>\n<b>Descrição:</b> Abordagem mais simples, onde a seleção do estimador e a definição de seus parâmetros são realizadas de forma completamente aleatória. Serve como linha de base para comparação.<br><b>Mecanismo:</b> Escolha uniforme aleatória de estimadores compatíveis e geração de parâmetros dentro de seus ranges definidos, sem qualquer aprendizado ou otimização baseada em experiências passadas.", unsafe_allow_html=True)
        with agent_cols_1[1]:
            st.markdown("<h5 style='color: #32CD32;'><b>Agente 2: Aleatório Ponderado por Experiência</b></h5>\n<b>Descrição:</b> Evolução do agente aleatório, incorpora um aprendizado básico a partir de experiências passadas. A aleatoriedade é mantida, mas ponderada pelos resultados obtidos anteriormente.<br><b>Mecanismo:</b> A escolha aleatória de estimadores é ponderada pelos scores alcançados. Parâmetros numéricos são amostrados de uma distribuição centrada nos melhores valores observados.", unsafe_allow_html=True)
        
        agent_cols_2 = st.columns(3)
        with agent_cols_2[0]:
            st.markdown("<h5 style='color: #32CD32;'><b>Agente 3: Meta-Aprendizado com Scikit-learn</b></h5>\n<b>Descrição:</b> Utiliza modelos do próprio Scikit-learn para prever o score potencial de diferentes configurações de pipeline antes de executá-las, evitando o treinamento de pipelines com baixo potencial.<br><b>Mecanismo:</b> Um modelo de meta-aprendizado (treinado em dados de experiências passadas) é usado para estimar o score de novas combinações de estimadores e parâmetros.", unsafe_allow_html=True)
        with agent_cols_2[1]:
            st.markdown("<h5 style='color: #32CD32;'><b>Agente 4: Algoritmo Genético / Evolucionário</b></h5>\n<b>Descrição:</b> Inspirado na evolução biológica, trata as configurações de pipeline como \"indivíduos\" em uma população, buscando otimizar o desempenho através de seleção, recombinação (crossover) e mutação.<br><b>Mecanismo:</b> Pipelines mais \"aptos\" são selecionados para \"reprodução\", gerando novas gerações através da combinação de características dos pais e introdução de variações aleatórias.", unsafe_allow_html=True)
        with agent_cols_2[2]:
            st.markdown("<h5 style='color: #32CD32;'><b>Agente 5: Otimização Bayesiana</b></h5>\n<b>Descrição:</b> Constrói um modelo probabilístico (modelo substituto) da função objetivo (o score do pipeline) com base em avaliações passadas, usando-o para guiar a busca por configurações de pipeline de forma mais eficiente.<br><b>Mecanismo:</b> Um modelo (e.g., Processo Gaussiano) é ajustado aos dados de configurações de pipeline e seus scores. Uma função de aquisição identifica a próxima configuração a ser avaliada.", unsafe_allow_html=True)
        st.divider()

        st.header("Scikit-learn: Fluxo de Trabalho do Agente")
        st.markdown("A biblioteca Scikit-learn é o conjunto de ferramentas que o agente de IA autônomo aprende a manipular para construir pipelines de Machine Learning. Suas funcionalidades são categorizadas e utilizadas pelo agente da seguinte forma:")
        
        sklearn_flow_cols = st.columns(3)
        with sklearn_flow_cols[0]:
            st.markdown("<h5 style='color: #FF8C00;'><b>Input</b></h5>\n- Escolha do dataset<br>- Feature Engineering (transformações de atributos)<br>- Configuração do agente (número de episódios, tipo de tarefa - supervisionado/não supervisionado)<br>- Seleção de estimadores/modelos que serão usados pelo agente", unsafe_allow_html=True)
        with sklearn_flow_cols[1]:
            st.markdown("<h5 style='color: #FF8C00;'><b>Processamento</b></h5>\n- Treinamento e previsão, conforme a estratégia de agente escolhida (aleatório, ponderado, meta-aprendizado, genético, bayesiano)", unsafe_allow_html=True)
        with sklearn_flow_cols[2]:
            st.markdown("<h5 style='color: #FF8C00;'><b>Saídas</b></h5>\n- Comparação dos resultados dos episódios/pipelines com visão geral e gráfico<br>- Visualização dos resultados de target x previsão x erro em detalhes e com gráfico por pipeline", unsafe_allow_html=True)
        st.divider()

        st.header("Referências e Fontes")
        st.markdown("-   <b>SUTTON, Richard S.; BARTO, Andrew G. <i>Reinforcement Learning: An Introduction</i>. 2. ed. Cambridge, MA: MIT Press, 2018.</b>\n    -   <b>Aplicação:</b> Livro fundamental para a compreensão dos conceitos de Aprendizado por Reforço que embasam a concepção e o treinamento dos agentes autônomos do projeto.\n\n-   <b>SCIKIT-LEARN. <i>User Guide</i>. Disponível em: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html). Acesso em: [Data do Acesso].</b>\n    -   <b>Aplicação:</b> Documentação oficial da biblioteca Scikit-learn, utilizada como referência para a identificação e implementação das ferramentas (estimadores, pré-processadores, pipelines) que compõem o espaço de ações do agente.\n\n-   <b>ARTIGO SUGERIDO: LI, Y. et al. AutoML: A Survey of the State-of-the-Art. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>, 2020.</b>\n    -   <b>Aplicação:</b> Artigo de pesquisa que oferece uma visão abrangente sobre o estado da arte em AutoML, contextualizando a relevância do projeto e as abordagens existentes.\n\n-   <b>GÉRON, Aurélien. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*. 2. ed. O'Reilly Media, 2019.</b>\n    -   <b>Aplicação:</b> Livro prático que serve como guia para a implementação de modelos de Machine Learning utilizando a biblioteca Scikit-learn, complementando a documentação oficial com exemplos e casos de uso.\n\n-   <b>RUSSELL, Stuart; NORVIG, Peter. *Artificial Intelligence: A Modern Approach*. 4. ed. Pearson, 2020.</b>\n    -   <b>Aplicação:</b> Obra clássica que fornece uma base sólida em Inteligência Artificial, incluindo conceitos sobre agentes inteligentes e sistemas autônomos, essenciais para a compreensão do funcionamento do agente de RL do projeto.", unsafe_allow_html=True)

    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Pesquisa Científica": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Pesquisa de mercado")
        st.markdown("##### Outputs\n- Resultados experimentais")
