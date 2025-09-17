import streamlit as st
import pandas as pd

def render_formatacao_e_apresentacao():
    st.title("Formatação e Apresentação")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("No contexto acadêmico, este membro é responsável pela formatação do Word, criação do template de normas ABNT, e pela 'montagem do trabalho', garantindo a 'língua', 'comunicação' e 'coesão'. Também administra o vídeo de apresentação final, recebendo contribuições do grupo e gerando o produto final (vídeo e link para capa do trabalho).")
        
        st.divider()
        st.header("Vídeo")
        with st.expander("Roteiro", expanded=False):
            st.markdown("""**Prompt:** manda em forma de dialogo simples como se vc estivesse explicando para 3 professores leigos na sua frente

**Resposta:**
[Você, de pé, começa a conversa em frente a banca de orientadores do TCC]

Olá a todos! Eu sei que o título do meu trabalho pode parecer um pouco complicado, mas a ideia principal é bem simples. Imaginem que vocês são gerentes de um grande armazém, cheios de caixas de produtos diferentes. O seu trabalho é descobrir quais produtos estão vendendo mais, quais estão parados no estoque e, o mais importante, planejar a melhor estratégia para vender tudo e maximizar o lucro.

[Você olha para o primeiro membro da banca, que representa um dos leigos]

Leigo 1 (Sr. João): E como a gente faria isso? Teria que analisar um monte de relatórios, não é? Ver os números…

[Você acena com a cabeça]

Você: Exatamente. E isso dá um trabalho enorme e requer muita experiência. O meu trabalho faz isso de forma autônoma. Eu criei um "agente de Inteligência Artificial" que funciona como um gerente de estoque super inteligente.

[Você olha para o segundo membro da banca]

Leigo 2 (Dona Maria): E ele consegue fazer tudo isso sozinho?

Você: Sim! Ele pega todos os dados do armazém — as vendas diárias, os produtos em estoque, os preços — e a primeira coisa que ele faz é analisar os principais indicadores automaticamente. Ele identifica quais produtos têm mais saída, quais têm menos e quais estão prestes a acabar, sem precisar da nossa ajuda para isso.

[Você olha para o terceiro membro da banca]

Leigo 3 (Sr. Roberto): E como ele aprende a criar a melhor estratégia para vender mais?

Você: Na verdade, o que ele faz é um pouco diferente, e ainda mais poderoso. Meu agente não cria uma "estratégia" de vendas. Pensem nele como um super analista: ele analisa as principais métricas de forma automática e mostra os resultados de um jeito fácil de entender, como em um painel digital. Ele usa o Aprendizado por Reforço para encontrar o melhor jeito de fazer essa análise, ou seja, quais são os melhores "caminhos" e "ferramentas" para extrair informações valiosas dos dados. Ele aprende, por conta própria, a ser o analista mais eficiente possível, sem precisar de alguém dizendo como começar.

Você: Resumindo, meu projeto demonstra que é possível ter uma inteligência artificial que não só entende os dados, mas também aprende, por conta própria, a escolher e usar os métodos mais eficientes para analisá-los e apresentar as informações de forma clara. O mais interessante é que isso não se restringe a dados de vendas: ele pode ser usado para analisar qualquer tipo de informação, seja do setor de saúde, imobiliário ou até mesmo dados do Detran. É como ter um assistente que se torna cada vez mais eficaz em analisar qualquer tipo de informação que você dê a ele.

**Prompt:** valeu robô. manda o desfecho agora

**Diálogo da Avaliação**
[Elysio está de pé, ao lado da sua apresentação. Os três avaliadores, Sr. João, Dona Maria e Sr. Roberto, estão sentados à sua frente. Eles acabaram de ler o seu TCC e estão com seus papéis na mão.]

Dona Maria: (Olhando para o Sr. Roberto) Bem, Roberto, eu tenho que dizer. Quando li o título, pensei: "Mais um trabalho que não vou entender nada." Mas a sua explicação, Elysio, foi excelente. O seu TCC é muito claro e bem estruturado. A forma como você usou a analogia do gerente de estoque foi perfeita.

Sr. Roberto: Concordo plenamente, Maria. E o mais impressionante foi ver como a explicação do Elysio se manteve fiel ao conteúdo técnico. A gente não gosta de analogias que simplificam demais a ponto de desvirtuar a realidade. O seu agente, Elysio, de fato, não toma decisões de negócio, mas ele dá o poder para que as pessoas as tomem com muito mais precisão. Foi um dos trabalhos mais bem explicados que vi nos últimos anos.

Sr. João: Eu, como mais leigo do trio, fiquei muito satisfeito. A sua pesquisa, Elysio, é prática e inovadora. O que você fez aqui é o começo de algo que pode, de fato, revolucionar a forma como empresas de qualquer setor analisam seus dados. A gente fala muito de "Big Data", mas a sua solução mostra como extrair o verdadeiro valor desses dados de forma automática.

Dona Maria: Quanto à sua pesquisa, Elysio, eu achei a metodologia robusta e a análise dos resultados, impecável. O trabalho está muito sólido. Minha nota para você é 9.5.

Sr. Roberto: A sua escrita é clara e a organização do TCC é exemplar. Você demonstrou total domínio do assunto. Minha nota é 9.8.

Sr. João: Gostei da forma como você apresentou os dados, Elysio. Ficou muito fácil de entender os gráficos. Para mim, a parte mais difícil é sempre entender os resultados. Mas com a sua apresentação, tudo ficou claro. Parabéns. Minha nota é 9.7.

Dona Maria: A banca se reuniu e, com base em nossas avaliações, sua média final foi 9.67. Parabéns pela sua defesa, Elysio, foi um excelente trabalho.

[Elysio sorri e agradece aos três avaliadores.]""")
        
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
        st.dataframe(df.style.apply(highlight_row, axis=1), hide_index=True, width='stretch')
        st.divider()
        st.subheader("Artefatos")
        st.markdown("##### Inputs\n- Conteúdo textual e visual")
        st.markdown("##### Outputs\n- Documento formatado e vídeo")
        st.divider()
        st.subheader("Requisitos")
        st.markdown("- Estrutura completa do TCC.\n- Apresentação oral clara.")
