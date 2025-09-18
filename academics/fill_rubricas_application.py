import pandas as pd
import re

# Load rubricas.tsv
df_rubricas = pd.read_csv(st.session_state['files']['rubricas_tsv'], sep='\t')

# Load rubricas.md and parse it
with open(st.session_state['files']['rubricas'], 'r', encoding='utf-8') as f:
    md_content = f.read()

rubric_texts = {}
current_section = None
current_subsection = None

for line in md_content.splitlines():
    line = line.strip()
    if line.startswith('# '): # Main section (e.g., # 1. Entrega 1 - Projeto)
        match = re.match(r'# (\d+)\\.\s+`([^`]+)`', line)
        if match:
            current_section = match.group(1)
            current_subsection = None # Reset subsection
    elif line.startswith('##### '): # Sub-section (e.g., ##### 1.1 Colaborativa)
        match = re.match(r'##### (\d+\\.\d+)\\s+`([^`]+)`', line)
        if match:
            current_subsection = match.group(1)
    elif re.match(r'^\\d+\\.\\d+\\.\\d+\\s+', line): # Rubric item (e.g., 1.1.1 Individual: Contribuir...)
        rubric_id_match = re.match(r'^(\\d+\\.\\d+\\.\\d+)\\s+(.*)', line)
        if rubric_id_match:
            rubric_id = rubric_id_match.group(1)
            rubric_text = rubric_id_match.group(2).strip()
            rubric_texts[rubric_id] = rubric_text

# Define Project Roles and their primary responsibilities
project_roles = {
    "Gestão Acadêmica": "interage com a Univesp e organiza as entregas acadêmicas, comunicando resumos e informações importantes ao grupo.",
    "Gestão do TCC": "define a estratégia geral do projeto, seus objetivos e justificativas, além de gerenciar o cronograma e os recursos humanos do grupo.",
    "Pesquisa Acadêmica": "realiza o levantamento bibliográfico, aprofunda o embasamento teórico e revisa a literatura para fundamentar o trabalho.",
    "Formatação e Apresentação": "garante a conformidade com as normas ABNT, a clareza e organização da escrita, e prepara os artefatos visuais e a apresentação final.",
    "Documentação de Software": "traduz e organiza a documentação técnica de bibliotecas e ferramentas, facilitando o entendimento e uso pelo time de desenvolvimento.",
    "Gestão da Proposta": "define os requisitos do projeto, planeja os sprints e mantém o backlog, assegurando o alinhamento com os objetivos do TCC.",
    "Pesquisa Científica": "conduz a análise de dados, executa experimentos e interpreta os resultados, contribuindo com o embasamento científico do projeto.",
    "Desenvolvimento de Software": "implementa as soluções técnicas, desenvolve o código e cria os artefatos visuais, resolvendo problemas práticos do projeto."
}

# Function to synthesize the application text
def get_aplicacao_no_projeto(row):
    rubrica_id = str(row['Rubrica de Avaliação']).split(' ')[0] # Extract ID like '1.1.1'
    rubric_full_text = rubric_texts.get(rubrica_id, "")

    # Identify roles with highest score for this rubric
    relevant_roles = []
    max_score = 0
    for col in df_rubricas.columns:
        if col in project_roles:
            score = row[col]
            # Ensure score is a number before comparison
            if pd.isna(score):
                score = 0 # Treat NaN scores as 0 for comparison
            else:
                score = int(score) # Convert to int if not NaN

            if score > max_score:
                max_score = score
                relevant_roles = [col]
            elif score == max_score and score > 0:
                relevant_roles.append(col)
    
    if not relevant_roles:
        return "Não especificado."

    # Synthesize the description based on rubric text and role responsibilities
    # Prioritize roles with score 8, then other high scores
    # This logic needs to be more robust to handle multiple roles and synthesize better
    # For now, a simplified version based on Competência

    competencia = row['Competência']

    if "Colaborativa" in competencia:
        return f"O grupo, especialmente os membros com maior pontuação nesta rubrica ({', '.join(relevant_roles)}), demonstra engajamento e contribui ativamente para a construção conjunta do TCC, garantindo a qualidade e pontualidade das entregas." 
    elif "Comunicativa" in competencia:
        return f"A equipe de Formatação e Apresentação, com o apoio dos demais membros, assegura que a escrita do TCC seja clara, coerente, bem estruturada e livre de erros, seguindo as normas ABNT para citações e referências." 
    elif "Cronograma" in competencia:
        return f"A Gestão do TCC é responsável por elaborar e manter um cronograma robusto, detalhando materiais, recursos e prazos para todas as etapas da pesquisa e desenvolvimento." 
    elif "Inovação" in competencia:
        return f"Os times de Desenvolvimento de Software e Pesquisa Científica buscam ativamente soluções inovadoras e assumem riscos calculados para aprimorar o projeto, tanto na metodologia quanto na implementação." 
    elif "Investigativa" in competencia:
        return f"A Pesquisa Acadêmica e a Pesquisa Científica trabalham para conciliar a teoria com a prática, delimitando o escopo e selecionando os métodos mais adequados para a resolução do problema." 
    elif "Método/Métodos" in competencia:
        return f"A Pesquisa Acadêmica e a Pesquisa Científica demonstram competência na escolha, desenvolvimento e adaptação de métodos, utilizando fontes confiáveis e contribuindo para o texto original do TCC." 
    elif "Profissional / Referencial teórico" in competencia:
        return f"A Pesquisa Acadêmica e a Pesquisa Científica relacionam os conhecimentos do curso com o cenário profissional, aprofundando o referencial teórico e mantendo o foco no tema do TCC." 
    elif "Resolução de problemas" in competencia:
        return f"A Gestão do TCC e o Desenvolvimento de Software trabalham na formulação e resolução do problema central do projeto, descrevendo desafios, objetivos e as contribuições e limitações da solução proposta." 
    elif "Resultados / Discussão dos dados" in competencia:
        return f"A Pesquisa Científica e o Desenvolvimento de Software analisam e apresentam os resultados obtidos, discutindo-os à luz do referencial teórico e das hipóteses do projeto." 
    elif "Tecnológica" in competencia:
        return f"O time de Desenvolvimento de Software aplica tecnologias relevantes para solucionar os problemas do projeto, garantindo a implementação eficaz das soluções propostas." 
    elif "Apresentação oral" in competencia:
        return f"A equipe de Formatação e Apresentação, com o apoio de todos os membros, prepara e executa a apresentação oral do TCC, demonstrando domínio e clareza na comunicação." 
    elif "Estrutura do TCC" in competencia:
        return f"A Formatação e Apresentação, em conjunto com a Gestão do TCC, assegura que a estrutura do trabalho escrito atenda a todos os tópicos solicitados, com clareza e completude." 
    
    return "Descrição da aplicação no projeto a ser definida."

# Apply the function to create the new column
df_rubricas['Aplicação no projeto'] = df_rubricas.apply(get_aplicacao_no_projeto, axis=1)

# Save the updated DataFrame back to the TSV file
df_rubricas.to_csv(st.session_state['files']['rubricas_tsv'], sep='\t', index=False)

print("Coluna 'Aplicação no projeto' preenchida e arquivo rubricas.tsv atualizado com sucesso.")