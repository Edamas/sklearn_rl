import streamlit as st
import pandas as pd

def render_formatacao_e_apresentacao():
    st.title("Formatação e Apresentação")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("Esta gestão é responsável pela formatação do documento em normas ABNT, pela criação e manutenção do template do TCC, e pela administração e edição do vídeo de apresentação final, focando na forma e não no conteúdo.")
        st.divider()
        st.header("Rúbricas da Banca de Avaliação")
        st.markdown("""**Estrutura do TCC**
- **Descrição:** Descreve claramente e de maneira completa todos os tópicos solicitados.

**Apresentação oral**
- **Descrição:** Apresentar oralmente o trabalho, respeitando o intervalo entre 15 a 20 minutos e de uma forma clara, com domínio.""")
        st.divider()
        st.header("Referências e Fontes")
        st.markdown("- Manual de Normas ABNT da instituição.")
    with col2:
        st.subheader("Organograma Funcional")
        data = {
            "Time": ["Academics"]*4 + ["Developers"]*4,
            "Função": ["Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação", "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"]
        }
        df = pd.DataFrame(data)
        def highlight_row(row):
            if row.Função == "Formatação e Apresentação": return ['color: white; background-color: #31333F'] * len(row)
            return ['color: black; background-color: white'] * len(row)
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, use_container_width=True)
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Conteúdo textual e visual")
        st.markdown("##### Outputs\n- Documento formatado e vídeo")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Estrutura completa do TCC.\n- Apresentação oral clara.")
