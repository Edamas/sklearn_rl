import streamlit as st
import pandas as pd
import functions as f
from functions import df_select_rows, get_rubricas_by_function_score_8

def render_gestao_proposta():
    st.title("Gestão da Proposta")
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("Esta gestão define as estratégias e definições maiores do projeto, como justificativa, objetivos, inovações, estratégias técnicas (infraestrutura, comunicação, plano, cronograma) e gestão de RH (com liderança servidora, democrática e liberal). No time Developers, a comunicação se dá com o setor de pesquisa (braço teórico) e com os membros do time Developers com foco acadêmico e em pesquisa.")
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
        st.dataframe(pd.DataFrame(proposal_details_data), width='stretch')
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
        st.header("Organograma Funcional do TCC")
        st.markdown('''
<style>
    .organograma table {
        width: 100%;
        border-collapse: collapse;
    }
    .organograma th, .organograma td {
        border: 1px solid #ccc;
        padding: 8px;
        text-align: center;
        vertical-align: top;
    }
    .organograma td {
        background-color: #ffffff;
        color: #333333;
    }
    .organograma .team-academics {
        background-color: #e6f7ff;
    }
    .organograma .team-developers {
        background-color: #f0f0f0;
    }
    .organograma .io {
        font-size: 12px;
        color: #555;
        text-align: left;
    }
    .organograma .arrows {
        font-size: 20px;
        color: #555;
    }
</style>
<div class="organograma">
    <table>
        <tr>
            <th colspan="4" class="team-academics">Time Academics</th>
        </tr>
        <tr>
            <!-- Gestão Acadêmica -->
            <td>
                <b>Gestão Acadêmica</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Resumos de disciplinas<br>
                &lt; Pesquisa em bibliografias<br>
                &lt; Entregas AVA</div>
                <div class="io"><b>Output:</b><br>
                &gt; Lista de Disciplinas<br>
                &gt; Orientações<br>
                &gt; Comunicação em Fóruns</div>
            </td>
            <!-- Gestão de Projetos (TCC) -->
            <td>
                <b>Gestão de Projetos (TCC)</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Lista de Disciplinas<br>
                &lt; Planos de sprints</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Objetivos, justificativas<br>
                &gt; Cronograma</div>
            </td>
            <!-- Pesquisa Acadêmica -->
            <td>
                <b>Pesquisa Acadêmica</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Objetivos, justificativas<br>
                &lt; Exploração de bibliotecas</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Referências bibliográficas<br>
                &gt; Embasamento teórico</div>
            </td>
            <!-- Desenvolvimento Acadêmico -->
            <td>
                <b>Desenvolvimento Acadêmico</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Referências bibliográficas<br>
                &lt; Artefatos (prints, vídeos)</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Templates ABNT<br>
                &gt; Vídeo final (YouTube)</div>
            </td>
        </tr>
        <tr>
            <td colspan="4"><span class="arrows">↕️</span></td>
        </tr>
        <tr>
            <th colspan="4" class="team-developers">Time Developers</th>
        </tr>
        <tr>
            <!-- Gestão de Documentação de Software -->
            <td>
                <b>Gestão de Documentação de Software</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Docs técnicos e de libs</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Documentação "traduzida"</div>
            </td>
            <!-- Gestão de Projeto de Software -->
            <td>
                <b>Gestão de Projeto de Software</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Documentação técnica<br>
                &lt; Objetivos do TCC</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Planos de sprints<br>
                &gt; Backlog</div>
            </td>
            <!-- Pesquisa Científica -->
            <td>
                <b>Pesquisa Científica</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Planos de sprints<br>
                &lt; Pesquisa de mercado</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Resultados experimentais</div>
            </td>
            <!-- Desenvolvimento de Software -->
            <td>
                <b>Desenvolvimento de Software</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Resultados experimentais<br>
                &lt; Requisitos da Proposta</div><br>
                <div class="io"><b>Output:</b><br>
                &gt; Código e Artefatos visuais (GitHub)</div>
            </td>
        </tr>
    </table>
</div>
''', unsafe_allow_html=True)

        st.divider()
        st.header("Registro de Atividades")
        f.show_registro_atividades_by_function("Gestão da Proposta")
        st.divider()
        st.header("Cronograma e Entregas")
        f.show_cronograma_by_function("Gestão da Proposta")
        st.divider()
        st.subheader("Rubricas relacionadas")
        # Obter o DataFrame filtrado da função
        df_filtered_rubricas = get_rubricas_by_function_score_8("Gestão da Proposta")

        if not df_filtered_rubricas.empty:
            # Preparar o DataFrame para exibição interativa
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_gestao_da_proposta")

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
            st.info("Nenhuma rubrica relacionada à função 'Gestão da Proposta' encontrada.")
        st.divider()
        st.header("Disciplinas Relacionadas")
        f.show_disciplinas_relacionadas_vri("Gestão da Proposta")

        f.show_referencias_by_function("Gestão da Proposta")

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
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Objetivos do TCC\n- Histórico da seleção")
        st.markdown("##### Outputs\n- Backlog e Sprints\n- Planejamento do escopo")
