<div align="center">

# **Análise de Desempenho de Agente de Inteligência Artificial Autônomo na Utilização da Suíte Scikit-learn em Projetos de Aprendizado de Máquinas (AutoML)**

</div>

---

## 1. 📜 Descrição

<p align="justify"> Uma pesquisa científica que investiga a viabilidade de usar um agente de Aprendizado por Reforço (RL) para aprender a selecionar e sequenciar autonomamente as ferramentas da biblioteca Scikit-learn, construindo um fluxo de trabalho de pré-processamento e modelagem para maximizar uma métrica de performance. </p>

---

## 2. 📝 Resumo

<p align="justify"> Este trabalho tem como objetivo analisar o desempenho de agentes de inteligência artificial autônomos na utilização da suíte Scikit-learn em projetos de aprendizado de máquina no contexto de AutoML (Automated Machine Learning). A crescente demanda por soluções capazes de automatizar etapas como a preparação de dados, a seleção de algoritmos e o ajuste de hiperparâmetros evidencia a relevância de ferramentas que reduzam a dependência de conhecimento técnico especializado, otimizando tempo e recursos. Nesse cenário, os agentes autônomos se destacam por sua capacidade de realizar tarefas complexas de forma automática, possibilitando maior eficiência na construção, avaliação e validação de modelos. </p>

**PALAVRAS-CHAVE**: Inteligência Artificial, Agentes Autônomos, Machine Learning.

---

## 3. 🎯 Objetivos

### 3.1. Objetivo Geral

<p align="justify"> Analisar o desempenho de agentes de Inteligência Artificial Autônomos na utilização da suíte Scikit-learn para tarefas de aprendizado de máquina em projetos AutoML. </p>

### 3.2. Objetivos Específicos

* 3.2.1. Investigar a aplicabilidade de agentes autônomos em diferentes contextos de aprendizado de máquina.
* 3.2.2. Comparar os resultados obtidos por AutoML com agentes autônomos em relação a modelos configurados manualmente.
* 3.2.3. Analisar métricas de desempenho como acurácia, precisão, recall, F1-score, tempo de execução e custo computacional.
* 3.2.4. Avaliar a escalabilidade da abordagem em diferentes conjuntos de dados.
* 3.2.5. Identificar desafios, limitações e possíveis melhorias no uso de agentes autônomos com Scikit-learn.

---

## 4. Justificativa e Delimitação do Problema

<p align="justify"> O campo do Aprendizado de Máquina (Machine Learning – ML) caracteriza-se por sua amplitude e complexidade. A construção de um pipeline de ML, que envolve etapas como o pré-processamento de dados, a seleção de algoritmos, a otimização de hiperparâmetros e a avaliação de modelos, demanda elevado nível de conhecimento técnico e considerável investimento de tempo. Essa complexidade configura-se como uma barreira de entrada, dificultando que profissionais e pesquisadores, mesmo aqueles que dispõem de dados relevantes, utilizem de maneira eficaz as potencialidades do ML. </p>

<p align="justify"> A automação desses processos, conhecida como AutoML (Automated Machine Learning), surge como uma alternativa para democratizar o acesso a técnicas de aprendizado de máquina. No entanto, grande parte das ferramentas de AutoML atualmente disponíveis ainda se fundamenta em estratégias exaustivas, como a busca em grade, que são computacionalmente onerosas e pouco eficientes. Nesse sentido, observa-se uma lacuna na literatura quanto ao uso de abordagens mais inteligentes e adaptativas para a construção de pipelines em ML. </p>

<p align="justify"> Este Trabalho de Conclusão de Curso propõe-se a preencher essa lacuna por meio da investigação da viabilidade de uma abordagem inovadora: a utilização de um agente de Inteligência Artificial Autônomo baseado em Aprendizado por Reforço (Reinforcement Learning – RL). O RL é uma subárea da Inteligência Artificial que se dedica ao estudo de como agentes inteligentes devem tomar decisões sequenciais em um ambiente, de modo a maximizar uma recompensa cumulativa. </p>

<p align="justify"> Com base nessa perspectiva, busca-se empregar um agente capaz de aprender, por meio de tentativa e erro, a construir pipelines de ML de forma autônoma e otimizada, selecionando os recursos mais adequados da biblioteca Scikit-learn. Além de demonstrar a viabilidade técnica dessa abordagem, o estudo também pretende analisar a estratégia aprendida pelo agente, o que pode gerar contribuições relevantes para a ciência de dados, ao revelar combinações e sequências de ferramentas que poderiam não ser facilmente identificadas por um cientista humano. </p>

<p align="justify"> A relevância do presente trabalho manifesta-se em três dimensões principais. No âmbito acadêmico, a pesquisa contribui para o avanço do estado da arte em AutoML, ao explorar uma metodologia ainda pouco empregada, com potencial para gerar publicações científicas e apresentações em eventos especializados. No campo da inovação tecnológica, os resultados podem fundamentar o desenvolvimento de ferramentas mais eficientes e inteligentes, superando as limitações das abordagens atualmente utilizadas. Por fim, no aspecto mercadológico, a automação de tarefas complexas em ML apresenta elevado potencial de aplicação comercial, atraindo o interesse de empresas de tecnologia, pesquisadores e profissionais de ciência de dados. </p>

---

## 5. 📚 Fundamentação Teórica

<p align="justify"> A fundamentação teórica aborda conceitos fundamentais sobre agentes inteligentes, sistemas autônomos, AutoML e a biblioteca Scikit-learn. </p>

<p align="justify"> Autores como Russell e Norvig (2020) apresentam os fundamentos da Inteligência Artificial, destacando a evolução dos sistemas baseados em agentes. No campo do aprendizado de máquina, Pedregosa et al. (2011) descrevem o papel do Scikit-learn como uma ferramenta essencial, consolidada pela sua simplicidade e eficiência. </p>

<p align="justify"> Já Feurer et al. (2019) analisam o Auto-sklearn, uma extensão voltada à automação de experimentos em ML, que combina meta-aprendizado e otimização bayesiana para seleção de modelos e ajuste de hiperparâmetros. </p>

<p align-justify"> O AutoML pode ser definido como o conjunto de técnicas que visam automatizar todo ou parte do ciclo de vida do aprendizado de máquina, desde a seleção de algoritmos até a engenharia de hiperparâmetros e avaliação de desempenho. Ele possibilita que profissionais com diferentes níveis de conhecimento em estatística e ciência de dados desenvolvam soluções robustas sem necessidade de expertise avançada em ML. </p>

<p align="justify"> O Scikit-learn, por sua vez, é uma biblioteca open-source em Python, amplamente utilizada em tarefas de classificação, regressão, clustering e redução de dimensionalidade. Seu ecossistema fornece uma interface unificada para treinamento, validação cruzada, métricas e integração de pipelines, sendo considerado padrão de referência para experimentação acadêmica e aplicações industriais. </p>

<p align="justify"> Quando associado a frameworks de AutoML, o Scikit-learn se torna ainda mais poderoso, viabilizando a construção de pipelines automatizados que reduzem esforço humano e aumentam a reprodutibilidade dos experimentos. </p>

---

## 6. 🛠️ Metodologia

<p align="justify"> A metodologia proposta neste trabalho compreende as seguintes etapas: </p>

6.1. **Seleção de dados**: escolha de conjuntos de dados padrão de uso consolidado na literatura de aprendizado de máquina.

6.2. **Implementação de agente autônomo**: configuração de um pipeline AutoML que utilize a suíte Scikit-learn.

6.3. **Configuração manual**: treinamento de modelos equivalentes de forma manual, incluindo ajuste de hiperparâmetros.

6.4. **Comparação de desempenho**: aplicação de métricas padronizadas, como acurácia, precisão, recall, F1-score, tempo de execução e custo computacional.

6.5. **Discussão dos resultados**: identificação de ganhos, limitações e possíveis melhorias.

---

## 7. 📅 Cronograma

| Quinzenas | Início | Atividade | Vencimento das atividades | Carência |
| :---: | :---: | :---: | :---: | :---: |
| Quinzena 1 | 11/08/2025 | | | |
| Quinzena 2 | 25/08/2025 | | | |
| Quinzena 3 | 08/09/2025 | Primeira entrega | 16/09/2025 às 23:59 | 21/09/2025 às 23:59 |
| Quinzena 4 | 22/09/2025 | | | |
| Quinzena 5 | 06/10/2025 | | | |
| Quinzena 6 | 20/10/2025 | Segunda entrega | 28/10/2025 às 23:59 | 03/11/2025 às 23:59 |
| Quinzena 7 | 03/11/2025 | Terceira entrega | 11/11/2025 às 23:59 | 16/11/2025 às 23:59 |

---

## 8. 💡 Resultados Esperados

<p align="justify"> Espera-se que os agentes de inteligência artificial autônomos apresentem desempenho competitivo em relação às abordagens manuais, especialmente em termos de tempo de execução e automação do processo de modelagem. Entretanto, é possível que surjam desafios relacionados à interpretabilidade dos modelos gerados e à demanda por maior capacidade computacional. Tais aspectos serão analisados com vistas a fornecer uma visão crítica sobre o uso de AutoML em diferentes contextos. </p>

---

## 9. 🤔 Considerações Finais

<p align="justify"> Este trabalho propõe uma análise sistemática sobre a aplicabilidade de agentes autônomos em projetos de aprendizado de máquina com a suíte Scikit-learn, dentro do contexto de AutoML. Os resultados almejam contribuir para a compreensão das vantagens e limitações do uso de tais sistemas em ambientes acadêmicos e corporativos, além de abrir espaço para pesquisas futuras sobre a integração de AutoML e agentes inteligentes. </p>

---

## 10. 💻 Softwares, Tecnologias e Linguagens Utilizadas

| Nome | Descrição | Categoria | Ícone |
| :--- | :--- | :--- | :---: |
| Python | Linguagem de programação de alto nível, interpretada, de script, imperativa, orientada a objetos, funcional, de tipagem dinâmica e forte. | Linguagem de Programação | ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) |
| Scikit-learn | Biblioteca de aprendizado de máquina de código aberto para a linguagem de programação Python. | Biblioteca | ![Scikit-Learn](https://camo.githubusercontent.com/118144d81ec86f9f76fcf7d90a624757d151d145f4e451ae16811bf1a04831e7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5363696b69745f4c6561726e2d6461726b677265656e3f7374796c653d666f722d7468652d6261646765266c6f676f3d7363696b69746c6561726e266c6f676f436f6c6f723d7768697465) |
| Pandas | Biblioteca de software criada para a linguagem de programação Python para manipulação e análise de dados. | Biblioteca | ![Pandas](https://camo.githubusercontent.com/3d7a922256075509efa5008b7a785e9081b3573faa16478ec77cc374aa2233c0/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50616e6461732d6461726b677265656e3f7374796c653d666f722d7468652d6261646765266c6f676f3d70616e646173266c6f676f436f6c6f723d7768697465) |
| Numpy | Biblioteca para a linguagem de programação Python, que suporta o processamento de grandes, matrizes e matrizes multidimensionais, juntamente com uma grande coleção de funções matemáticas de alto nível para operar nessas matrizes. | Biblioteca | ![Numpy](https://camo.githubusercontent.com/f05a73e44899b0602c2aef5a6574c157b69eda7b24ecb7bde1a5251cb5bf4e5b/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f4e756d70792d6461726b677265656e3f7374796c653d666f722d7468652d6261646765266c6f676f3d6e756d7079266c6f676f436f6c6f723d7768697465) |
| Plotly | Biblioteca de software de código aberto para a criação de gráficos interativos. | Biblioteca | ![Plotly](https://camo.githubusercontent.com/815be29d999a5ce84e8223aa3d1df10a63d1c58387dbd0da393919a8ac0fb56e/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f506c6f746c792d6461726b677265656e3f7374796c653d666f722d7468652d6261646765266c6f676f3d706c6f746c79266c6f676f436f6c6f723d7768697465) |
| Streamlit | Framework de aplicativo da web de código aberto para criar e compartilhar aplicativos da web de ciência de dados e aprendizado de máquina. | Framework | ![Streamlit](https://camo.githubusercontent.com/93279e9bb44216a85c3b5a145f0a53d2fde7b354b489a4e90f37dcf304f4e80c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f53747265616d6c69742d6461726b677265656e3f7374796c653d666f722d7468652d6261646765266c6f676f3d73747265616d6c6974266c6f676f436f6c6f723d7768697465) |
| Jupyter | Projeto de código aberto e uma comunidade cujo objetivo é desenvolver software de código aberto, padrões abertos e serviços para computação interativa em dezenas de linguagens de programação. | Ferramenta | ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) |
| Git | Sistema de controle de versão distribuído gratuito e de código aberto projetado para lidar com tudo, desde projetos pequenos a muito grandes, com velocidade e eficiência. | Ferramenta | ![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white) |
| GitHub | Plataforma de hospedagem de código-fonte e arquivos com controle de versão usando o Git. | Plataforma | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white) |

---

## 11. 📊 Detalhes da Proposta

| Atributo | Avaliação | Pontuação |
| :--- | :--- | :---: |
| Viabilidade | ██░░░ | 2/5 |
| Apelo | █████ | 5/5 |
| Tendência | █████ | 5/5 |
| Inovação | █████ | 5/5 |
| Contribuição Social | ████░ | 4/5 |
| Relação com BCD | █████ | 5/5 |
| Adequação ao TCC | █████ | 5/5 |
| Potencial | █████ | 5/5 |
| Facilidade de Aquisição | █████ | 5/5 |
| Qualidade dos Dados | ████░ | 4/5 |
| Nível de Atualização | █████ | 5/5 |

---

## 12. 📑 Rúbricas de Avaliação por Entrega

### 12.1. Entrega 1 - Projeto

#### 12.1.1. Tecnológica

*   **Descrição:** Habilidades para utilizar ferramentas tecnológicas na solução de um determinado problema.
*   **Critérios:**
    *   PESQUISAR PRINCIPAIS BASES CIENTÍFICAS E BIBLIOTECAS VIRTUAIS
    *   PESQUISAR (NAS BASES) POR LIVROS E ARTIGOS QUE AUXILIEM NA SOLUÇÃO DE DETERMINADO PROBLEMA
    *   PESQUISAR FERRAMENTAS DE GERENCIAMENTO DE ARQUIVOS E REFERÊNCIAS, SELECIONAR UMA E UTILIZÁ-LA
*   **Avaliação:**
    *   **0% a 45%:** O grupo não acessa bases científicas, ferramentas de gerenciamento de arquivos e referências.
    *   **46% a 70%:** O grupo acessa poucas bases científicas, ferramentas de gerenciamento de arquivos e de referências.
    *   **71% a 100%:** O grupo acessa as principais bases científicas e bibliotecas virtuais na busca de livros e artigos que auxiliem na solução de determinado problema. Utiliza ferramentas de gerenciamento de arquivos e de referências.

### 12.2. Entrega 2 - Desenvolvimento

#### 12.2.1. Tecnológica

*   **Descrição:** Habilidades para utilizar ferramentas tecnológicas na solução de um determinado problema.
*   **Critérios:**
    *   Utilizar bases científicas e bibliotecas virtuais na busca de livros e artigos que auxiliem na solução de determinado problema.
    *   Utilizar ferramentas de gerenciamento de arquivos e de referências.
*   **Avaliação:**
    *   **0% a 45%:** O grupo não acessa bases científicas, ferramentas de gerenciamento de arquivos e referências.
    *   **46% a 70%:** O grupo acessa poucas bases científicas, ferramentas de gerenciamento de arquivos e de referências.
    *   **71% a 100%:** O grupo acessa as principais bases científicas e bibliotecas virtuais na busca de livros e artigos que auxiliem na solução de determinado problema. Utiliza ferramentas de gerenciamento de arquivos e de referências.

### 12.3. Banca de Avaliação

#### 12.3.1. Estrutura do TCC

*   **Descrição:** Descreve claramente e de maneira completa todos os tópicos solicitados.
*   **Avaliação:**
    *   **0% a 45%:** Descreve menos da metade dos tópicos solicitados. Nota-se baixo empenho em buscar informações solicitadas.
    *   **46% a 70%:** Descreve claramente os tópicos solicitados, mas deixa de tratar alguns aspectos solicitados.
    *   **71% a 100%:** Descreve claramente e de maneira completa todos os tópicos solicitados.

---

## 13. 🎓 Sobre a Univesp

<p align="justify"> A Universidade Virtual do Estado de São Paulo (Univesp) é uma instituição de ensino superior pública e gratuita, credenciada pelo Conselho Estadual de Educação e pelo Ministério da Educação, com sede na cidade de São Paulo. A Univesp é uma universidade pública que tem como objetivo a formação de profissionais em cursos de graduação e pós-graduação, na modalidade a distância. </p>

---

## 14. 📜 Sobre o TCC

<p align="justify"> Este trabalho é um Trabalho de Conclusão de Curso (TCC) do curso de Bacharelado em Ciência de Dados da Universidade Virtual do Estado de São Paulo (Univesp). </p>

---

## 15. 👨‍💻 Integrantes

* Cintia Aguena Zanetti
* Edna Aparecida Nascimento
* Elysio Damasceno da Silva Neto
* Gustavo Daniel Martinez Madeira
* José Donizete De Lima
* Pascoal Fernandes Neto
* Wesllei Moreira de Sousa Sabino
* Wilson Matos de Carvalho Neto

---

## 16. 📚 Referências

* 16.1. ABNT – Associação Brasileira de Normas Técnicas. NBR 14724: Informação e documentação. Trabalhos Acadêmicos - Apresentação. Rio de Janeiro: ABNT, 2002.
* 16.2. FEURER, M. et al. Auto-sklearn 2.0: Hands-free AutoML via Meta-Learning. In: Advances in Neural Information Processing Systems, 2019.
* 16.3. PEDREGOSA, F. et al. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, v. 12, p. 2825–2830, 2011.
* 16.4. RUSSELL, S.; NORVIG, P. Artificial Intelligence: A Modern Approach. 4. ed. New Jersey: Pearson, 2020.
* 16.5. ZOPH, B.; LE, Q. V. Neural Architecture Search with Reinforcement Learning. In: International Conference on Learning Representations (ICLR), 2017.