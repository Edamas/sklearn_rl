import streamlit as st
import pandas as pd
import functions as f
from functions import df_select_rows, get_rubricas_by_function_score_8

def render_pesquisa_cientifica():
    st.title("Pesquisa Científica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("A gestão de pesquisa cria táticas para a equipe, com embasamento e desenvolvimento teórico. No time Developers, o gestor de pesquisa se conecta com os membros de desenvolvimento (programador/desenvolvedor) para explorar a parte teórica de bibliotecas online, livros ou desenvolvimento/exploração.")
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

        

    st.divider()
    st.header("Registro de Atividades")
    f.show_registro_atividades_by_function("Pesquisa Científica")
    st.divider()
    st.header("Cronograma e Entregas")
    f.show_cronograma_by_function("Pesquisa Científica")
    st.divider()
    st.subheader("Rubricas relacionadas")
    # Obter o DataFrame filtrado da função
    df_filtered_rubricas = get_rubricas_by_function_score_8("Pesquisa Científica")

    if not df_filtered_rubricas.empty:
        # Preparar o DataFrame para exibição interativa
        df_display = df_filtered_rubricas[['rubrica']].copy()
        df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
        
        selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_pesquisa_cientifica")

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
        st.info("Nenhuma rubrica relacionada à função 'Pesquisa Científica' encontrada.")


    st.divider()
    st.header("Disciplinas Relacionadas")
    f.show_disciplinas_relacionadas_vri("Pesquisa Científica")

    f.show_referencias_by_function("Pesquisa Científica")

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
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Pesquisa de mercado")
        st.markdown("##### Outputs\n- Resultados experimentais")