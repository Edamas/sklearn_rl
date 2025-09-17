import streamlit as st
import pandas as pd

def render_gestao_academica():
    st.title("Gestão Acadêmica")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("O membro acadêmico no time Academics se comunica com a parte administrativa ou executiva do grupo (de 8 pessoas) e com o sistema AVA, cuidando das entregas do grupo para a disciplina de TCC. Enquanto a turma de pesquisa pode focar em conteúdos de outras disciplinas, o acadêmico se dedica à disciplina de TCC, comunicando ao grupo resumos e informações importantes. Também interage com o fórum e organiza os arquivos acadêmicos.")
        st.divider()
        st.header("Cronograma e Entregas")
        cronograma_path = "D:\\PROGRAMACAO\\sklearn_rl\\docs\\cronograma.tsv"
        try:
            df_cronograma = pd.read_csv(cronograma_path, sep='\t')
            df_cronograma.rename(columns={'Vencimento das atividades': 'Vencimento'}, inplace=True)
            st.dataframe(df_cronograma, hide_index=True, width='stretch')
        except FileNotFoundError:
            st.error(f"Arquivo de cronograma não encontrado: {cronograma_path}")
        except Exception as e:
            st.error(f"Erro ao carregar ou processar o cronograma: {e}")
        st.divider()
        st.header("Referências e Fontes")
        st.markdown("- Documentação e calendário oficial da disciplina de TCC.")
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Gestão Acadêmica": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Orientações e Rubricas")
        st.markdown("##### Outputs\n- Entregas e Registros no AVA")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Gestão de prazos.")