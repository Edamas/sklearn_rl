import streamlit as st
import pandas as pd

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
        st.header("Rúbricas da Banca de Avaliação")
        st.markdown("""**Estrutura do TCC**
- **Descrição:** Descreve claramente e de maneira completa todos os tópicos solicitados.

**Apresentação oral**
- **Descrição:** Apresentar oralmente o trabalho, respeitando o intervalo entre 15 a 20 minutos e de uma forma clara, com domínio.""")

        st.divider()
        with st.expander("Rubricas de avaliação relacionadas", expanded=False):
            st.markdown("""# 3. `Banca de Avaliação (2 avaliadores)`
##### 3.1 	`Apresentação oral`
	3.1.1 	(Cada aluno deve) Apresentar oralmente o trabalho
	3.1.2 	Apresentar respeitando o intervalo entre 15 a 20 minutos Apresentar o trabalho de uma forma clara.
	3.1.3 	Garantir que o trabalho seja expresso (apresentado) com domínio (sem dificuldades)
##### 3.2 	`Comunicação / Linguagem`
	3.2.1 	Empregar habilidades para comunicar-se utilizando as variadas linguagens
	3.2.2 	Garantir que a estruturação da escrita geral facilita a compreensão.
	3.2.3 	Garantir que a escrita não contém erros de ortografia
	3.2.4 	Garantir que a escrita não contém erros de gramática
	3.2.5 	Garantir que a escrita não contém erros de pontuação
	3.2.6 	Criar estratégia para que o texto seja organizado
	3.2.7 	Criar estratégia para que o texto seja coerente, sem trechos incoerentes
	3.2.8 	Criar estratégia para que o texto apresente sentido
	3.2.9 	Criar estratégia para que o texto seja bem estruturado
	3.2.10 	Criar estratégia para que o texto tenha sempre articulação entre as partes
	3.2.11 	Apresentar propositalmente sentido e articulação na integralidade do texto
	3.2.12 	Escrever todas as citações e referências bibliográficas na norma ABNT
##### 3.3 	`Considerações finais`
	3.3.1 	Apresentar claramente as contribuições do trabalho realizado
	3.3.2 	Apresentar claramente as limitações do trabalho realizado
	3.3.3 	Apresentar de forma completa as contribuições ou limitações do trabalho
##### 3.4 	`Estrutura do TCC`
	3.4.1 	Descreve claramente e de maneira completa todos os tópicos solicitados
	3.4.2 	Garantir que todos os tópicos solicitados sejam claramente atendidos
	3.4.3 	Demonstrar empenho
	3.4.4 	Demonstrar esforço em buscar as informações solicitadas
##### 3.5 	`Método/Métodos`
	3.5.1 	Apresentar a competência de escolha e desenvolvimento do método
	3.5.2 	Apresentar o processo de escolha ou combinação dos métodos
	3.5.3 	Escolher um ou mais métodos adequados
	3.5.4 	Pesquisar além da literatura (convencional)
	3.5.5 	Utilizar (deixar claro) referencial de fontes confiáveis
	3.5.6 	Adaptar os métodos a partir da sua capacidade de análise, criação e ajustes a uma realidade apresentada
	3.5.7 	Pesquisar "mais" sobre o assunto
	3.5.8 	Dar contribuições ao texto original
	3.5.9 	Demonstrar compreensão do método como um caminho para um fim determinado
	3.5.10 	Ater-se ao tema
	3.5.11 	Eliminar redundâncias de conteúdo (com a parte introdutória do texto)
##### 3.6 	`Resolução de problemas / Objetivo geral / Objetivos específicos / Desenvolvimento`
	3.6.1 	Empregar habilidades para entender e resolver problemas de um cenário profissional
	3.6.2 	Empregar habilidades para entender e resolver problemas de um cenário profissional
##### 3.7 	`Resolução de problemas / Objetivo geral / Objetivos específicos /Desenvolvimento`
	3.7.1 	Desenvolver o problema solucionado (justificativa)
	3.7.2 	Descrever os desafios para a solução
	3.7.3 	Descrever claramente os objetivos
	3.7.4 	Distinguir os objetivos gerais dos específicos
	3.7.5 	Deve haver mais de um objetivo complementar
	3.7.6 	Apresentar um objetivo geral
	3.7.7 	Apresentar múltiplos objetivos complementares
	3.7.8 	Apresentar os problemas de forma abrangente (não parte dele)
	3.7.9 	Apresentar claramente e completa as limitações do trabalho realizado
	3.7.10 	Apresentar de forma clara e completa as contribuições do trabalho realizado
	3.7.11 	Apresentar clareza na formulação do problema/solução científica
	3.7.12 	Apresentar clareza no desenvolvimento do problema/solução científica
##### 3.8 	`Resultados / Discussão dos dados`
	3.8.1 	Analisar os resultadosà luz do referencial teórico
	3.8.2 	Apresentar os resultados""")
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

        st.divider()
        st.markdown("### Organograma Funcional da Equipe\nEste organograma detalha as funções e interações entre os membros das equipes Academics e Developers, incluindo suas comunicações internas e externas.\n\n#### Equipe Academics\n| Gestão Acadêmica | Gestão de Projetos (TCC) | Pesquisa Acadêmica | Desenvolvimento Acadêmico |\n| :---: | :---: | :---: | :---: |\n| < Orientação da Orientadora, Estratégias de Relatório <br> Atualização da Orientadora, Resumos de disciplinas Relacionadas, Pesquisa em bibliografias do curso <br> > Lista de Disciplinas, Orientações Orientadora, Rubricas, Conteúdos disciplina TCC, comunicação em Forums, Bibliografia do TCC > | < Objetivos do projeto, justificativas, cronograma, estratégias definidas <br> Gestão de Projeto Acadêmico <br> > Objetivos, justificativas, cronograma, estratégias definidas > | < Referências bibliográficas, orientações da disciplina, embasamento teórico <br> Pesquisa Acadêmica de Data Science <br> > Exploração de bibliotecas, análise de dados, resultados experimentais > | < Templates ABNT, organização do Word, vídeo final, link para capa <br> Desenvolvimento Acadêmico <br> > YouTube, Teams, OneDrive > |\n\n| ^ | ^ | ^ | ^ |\n| v | v | v | v |\n\n#### Equipe Developers\n| Gestão Acadêmica (Doc) | Gestão de Projetos (Proposta) | Pesquisa Científica | Desenvolvimento de Software |\n| :---: | :---: | :---: | :---: |\n| < Documentação técnica, fóruns SkLearn, Docs compartilhados <br> Gestão de Documentação de Software <br> > Documentação técnica, fóruns SkLearn, Docs compartilhados > | < Planos de sprints, backlog, integração acadêmica, coordenação técnica <br> Gestão de Projeto de Software <br> > Planos de sprints, backlog, integração acadêmica, coordenação técnica > | < Exploração de bibliotecas, análise de dados, resultados experimentais <br> Pesquisa científica de Data Science <br> > Exploração de bibliotecas, análise de dados, resultados experimentais > | < Docs SkLearn <br> Desenvolvimento de Software <br> > GitHub, Streamlit Community Cloud, Teams, Documentação e Fóruns de Bibliotecas (sklearn, streamlit) > |\n\n*Conexões Horizontais e Verticais representadas pelas setas de Input (<) e Output (>) dentro de cada célula e entre as linhas de equipes.*")
