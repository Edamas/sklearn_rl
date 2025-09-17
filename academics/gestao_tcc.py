import streamlit as st
import pandas as pd
from functions import df_select_rows

def render_gestao_tcc():
    st.title("Gestão do TCC")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão é responsável pelas estratégias e definições maiores do projeto, como justificativa, objetivos, inovações, estratégias técnicas (infraestrutura, comunicação, plano, cronograma) e gestão de RH (com liderança servidora, democrática e liberal). A comunicação se dá com o setor de pesquisa (teórico, para frente) e com o acadêmico (para a Univesp, materiais de base e biblioteca acadêmica).")
        st.divider()
        st.header("Proposta do Projeto")
        st.markdown("""**Resumo:** O objetivo é analisar o desempenho de agentes de IA autônomos na utilização da suíte Scikit-learn em projetos de AutoML, visando automatizar etapas como preparação de dados e seleção de algoritmos para otimizar tempo e recursos.

**Justificativa:** O campo do Aprendizado de Máquina (ML) é complexo. A automação (AutoML) surge para democratizar seu acesso. Este trabalho propõe o uso de Aprendizado por Reforço (RL) para preencher a lacuna de abordagens mais inteligentes em AutoML.""")
        st.divider()
        st.header("Rubricas de Avaliação (Nota 8)")

        @st.cache_data(show_spinner=False)
        def load_and_process_rubricas_data_for_gestao_do_tcc():
            import pandas as pd
            import re

            try:
                with open("D:\\PROGRAMACAO\\sklearn_rl\\docs\\rubricas.md", 'r', encoding='utf-8') as f:
                    md_content = f.read()
            except FileNotFoundError:
                st.error("Arquivo rubricas.md não encontrado.")
                return pd.DataFrame()

            entrega = None
            competencia = None
            rubricas_map = {}

            for line in md_content.splitlines():
                line = line.strip()
                if line.startswith('# '):
                    match = re.search(r'`(Entrega[^`]+)`', line)
                    if match:
                        entrega = match.group(1)
                elif line.startswith('##### '):
                    match = re.search(r'`([^`]+)`', line)
                    if match:
                        competencia = match.group(1)
                elif re.match(r'^\d+\.\d+\.\d+', line):
                    rubrica_text_from_md = re.sub(r'^\d+\.\d+\.\d+\s+', '', line).strip()
                    rubrica_id_match = re.match(r'^(\d+\.\d+\.\d+)', line)
                    if rubrica_id_match:
                        rubrica_id = rubrica_id_match.group(1)
                        rubricas_map[rubrica_id] = {
                            "Entrega": entrega,
                            "Competência": competencia,
                        }
            
            df_map = pd.DataFrame.from_dict(rubricas_map, orient='index').reset_index().rename(columns={'index': 'id'})

            try:
                df_rubricas = pd.read_csv("D:\\PROGRAMACAO\\sklearn_rl\\docs\\rubricas.tsv", sep='\t')
            except FileNotFoundError:
                st.error("Arquivo rubricas.tsv não encontrado.")
                return pd.DataFrame()
            
            def extract_id(text):
                match = re.match(r'^(\d+\.\d+\.\d+)', str(text))
                if match:
                    return match.group(1)
                return None

            df_rubricas['id'] = df_rubricas['Rubrica de Avaliação'].apply(extract_id)
            df_full = pd.merge(df_rubricas, df_map, on='id', how='left')
            return df_full

        df_full = load_and_process_rubricas_data_for_gestao_do_tcc()
        
        if not df_full.empty:
            funcao_nome = "Gestão do TCC"
            if funcao_nome in df_full.columns:
                df_filtered = df_full[df_full[funcao_nome] == 8].copy()

                if not df_filtered.empty:
                    df_display = df_filtered[['Rubrica de Avaliação']].copy()
                    df_display.rename(columns={'Rubrica de Avaliação': 'Selecione uma rubrica para ver os detalhes'}, inplace=True)
                    
                    selected_index = df_select_rows(df_display, selection_mode='single-row', key=f"rubricas_gestao_do_tcc")

                    if selected_index is not None and selected_index in df_filtered.index:
                        selected_rubrica = df_filtered.loc[selected_index]
                        st.subheader("Ficha da Rubrica")
                        
                        st.markdown(f"**Entrega:** {selected_rubrica.get('Entrega', 'N/A')}")
                        st.markdown(f"**Competência:** {selected_rubrica.get('Competência', 'N/A')}")
                        st.markdown(f"**Rubrica de Avaliação:** {selected_rubrica.get('Rubrica de Avaliação', 'N/A')}")
                        st.markdown(f"**Aplicação no projeto:** {funcao_nome}")
                else:
                    st.info(f"Nenhuma rubrica com nota 8 para '{funcao_nome}'.")
            else:
                st.error(f"Coluna '{funcao_nome}' não encontrada em rubricas.tsv.")

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
                <span class="arrows">↔️</span><br>
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
                <span class="arrows">↔️</span><br>
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
                <span class="arrows">↔️</span><br>
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
                <span class="arrows">↔️</span><br>
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
                <span class="arrows">↔️</span><br>
                <div class="io"><b>Output:</b><br>
                &gt; Documentação "traduzida"</div>
            </td>
            <!-- Gestão de Projeto de Software -->
            <td>
                <b>Gestão de Projeto de Software</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Documentação técnica<br>
                &lt; Objetivos do TCC</div><br>
                <span class="arrows">↔️</span><br>
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
                <span class="arrows">↔️</span><br>
                <div class="io"><b>Output:</b><br>
                &gt; Resultados experimentais</div>
            </td>
            <!-- Desenvolvimento de Software -->
            <td>
                <b>Desenvolvimento de Software</b><br>
                <div class="io"><b>Input:</b><br>
                &lt; Resultados experimentais<br>
                &lt; Requisitos da Proposta</div><br>
                <span class="arrows">↔️</span><br>
                <div class="io"><b>Output:</b><br>
                &gt; Código e Artefatos visuais (GitHub)</div>
            </td>
        </tr>
    </table>
</div>
''', unsafe_allow_html=True)
            
        st.divider()
        
    
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
            if row.Função == "Gestão do TCC": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Estratégias de Relatório")
        st.markdown("##### Outputs\n- Objetivos e Justificativas")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Delinear área de estudo.\n- Apresentar autores.")

        
