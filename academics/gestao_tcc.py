import streamlit as st
import pandas as pd
import functions as f

def render_gestao_tcc():
    st.title("Gestão do TCC")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão é responsável pelas estratégias e definições maiores do projeto, como justificativa, objetivos, inovações, estratégias técnicas (infraestrutura, comunicação, plano, cronograma) e gestão de RH (com liderança servidora, democrática e liberal). A comunicação se dá com o setor de pesquisa (teórico, para frente) e com o acadêmico (para a Univesp, materiais de base e biblioteca acadêmica).")
        st.divider()
        st.header("Proposta do Projeto")
        st.markdown("""**Resumo:** O objetivo é analisar o desempenho de agentes de IA autônomos na utilização da suíte Scikit-learn em projetos de AutoML, visando automatizar etapas como preparação de dados e seleção de algoritmos para otimizar tempo e recursos.\n\n**Justificativa:** O campo do Aprendizado de Máquina (ML) é complexo. A automação (AutoML) surge para democratizar seu acesso. Este trabalho propõe o uso de Aprendizado por Reforço (RL) para preencher a lacuna de abordagens mais inteligentes em AutoML.""")
        st.divider()

        st.subheader("Rubricas relacionadas")
        df_filtered_rubricas = f.get_rubricas_by_function_score_8("Gestão do TCC")

        if not df_filtered_rubricas.empty:
            df_display = df_filtered_rubricas[['rubrica']].copy()
            df_display.rename(columns={'rubrica': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
            
            selected_index = f.df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_gestao_tcc", prompt=None)

            if selected_index is not None and selected_index in df_filtered_rubricas.index:
                selected_rubrica = df_filtered_rubricas.loc[selected_index]
                st.subheader("Ficha da Rubrica")
                
                st.markdown(f"**<font color='#FFD700'>Entrega {selected_rubrica['item_entrega']}: {selected_rubrica['entrega']}</font>**", unsafe_allow_html=True)
                st.markdown(f"  **<font color='#ADD8E6'>Subitem {selected_rubrica['subitem']}: {selected_rubrica['competencia']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    **<font color='#90EE90'>Rubrica {selected_rubrica['item_rubrica']}: {selected_rubrica['rubrica']}</font>**", unsafe_allow_html=True)
                st.markdown(f"    Aplicação no projeto:")
                st.markdown(f"      {selected_rubrica['aplicacao_no_projeto']}")
            else:
                pass # Removido: st.info("Nenhuma rubrica selecionada ou nenhuma rubrica relacionada à função atual.")
        else:
            st.info("Nenhuma rubrica relacionada à função 'Gestão do TCC' encontrada.")

        st.divider()
        st.subheader('Organograma funcional do TCC')
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
        
    
    st.divider()
    st.header("Disciplinas Relacionadas")
    f.show_disciplinas_relacionadas_vri("Gestão do TCC")
    
    f.show_referencias_by_function("Gestão do TCC")
    
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Gestão do TCC": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Orientações da Univesp/AVA\n- Rubricas de avaliação\n- Cronograma oficial do TCC\n- Comunicações da orientação\n- Registro de atividades do grupo\n- Pesquisas teóricas (do setor de pesquisa)\n- Materiais de base e biblioteca acadêmica (do setor acadêmico)")
        st.markdown("##### Outputs\n- Definição de justificativa, objetivos, inovações\n- Estratégias técnicas (infraestrutura, comunicação, plano, cronograma)\n- Plano de gestão de RH (liderança servidora, democrática e liberal)\n- Comunicação com setor de pesquisa e acadêmico")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("##### Requisitos do TCC\n- Definição clara da justificativa e objetivos do TCC.\n- Elaboração de um plano de projeto abrangente (cronograma, recursos).\n- Gestão eficaz da equipe e comunicação entre os membros.\n- Alinhamento com as diretrizes da Univesp para o TCC.")
        st.markdown("##### Requisitos da Proposta (sobre o agente)\n- Delinear as inovações propostas pelo agente de IA.\n- Definir as estratégias técnicas para o desenvolvimento do agente.\n- Garantir a viabilidade e relevância da proposta do agente.")