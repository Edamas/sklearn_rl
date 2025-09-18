import pandas as pd
from datetime import datetime, timedelta

# Define the start date of the project
project_start_date = datetime(2025, 8, 11)

# Define the 8 functions
functions = [
    "Gestão Acadêmica", "Gestão do TCC", "Pesquisa Acadêmica", "Formatação e Apresentação",
    "Documentação de Software", "Gestão da Proposta", "Pesquisa Científica", "Desenvolvimento de Software"
]

# Data for the cronograma
cronograma_data = []

# Quinzena 1 (11/08/2025 - 24/08/2025): Iniciação e Planejamento do Grupo
quinzena_start = project_start_date
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Organização de documentos de referência e comunicação inicial com a Univesp.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Definição da estrutura inicial do projeto e planejamento de recursos humanos.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Levantamento de temas e referências bibliográficas iniciais para propostas.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Análise de templates e requisitos de formatação para o TCC.",
    "Responsável": "Wesllei"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Levantamento de ferramentas e padrões de documentação para o projeto.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Coleta e organização de propostas de TCC e requisitos iniciais.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Análise de competições e oportunidades de pesquisa em IA.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 1",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Configuração do ambiente de desenvolvimento e versionamento (Git).",
    "Responsável": "Elysio"
})

# Quinzena 2 (25/08/2025 - 07/09/2025): Seleção da Proposta e Início do Desenvolvimento
quinzena_start = project_start_date + timedelta(days=14)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Acompanhamento da eleição da proposta e comunicação de prazos.",
    "Responsável": "Edna"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Finalização da seleção da proposta e alinhamento de objetivos.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Aprofundamento da pesquisa bibliográfica para a proposta selecionada.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Início da estruturação do documento base do TCC.",
    "Responsável": "Wesllei"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Início da documentação de requisitos e arquitetura do protótipo.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Detalhamento da proposta selecionada e criação do backlog inicial.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Definição de métricas e estratégias de avaliação para o protótipo.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 2",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Implementação do protótipo v1.0 (agente inicial) e refatoração de features.",
    "Responsável": "Elysio"
})

# Quinzena 3 (08/09/2025 - 21/09/2025): Desenvolvimento do Protótipo e Refatoração
quinzena_start = project_start_date + timedelta(days=28)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Monitoramento de prazos de entrega e comunicação com a orientação.",
    "Responsável": "Edna"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Revisão dos objetivos e escopo do protótipo.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Revisão da fundamentação teórica para o protótipo atual.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Formatação e revisão do relatório do protótipo.",
    "Responsável": "Wesllei"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Atualização da documentação técnica do protótipo.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Gerenciamento do backlog e planejamento de sprints para o protótipo.",
    "Responsável": "Elysio"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Análise de resultados preliminares do protótipo.",
    "Responsável": "Todos"
})
cronograma_data.append({
    "Quinzena": "Quinzena 3",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Desenvolvimento e refatoração do protótipo (v1.1 a v1.4), incluindo melhorias de UI e hiperparâmetros.",
    "Responsável": "Elysio"
})

# Quinzena 4 (22/09/2025 - 05/10/2025): Implementação de Agentes Avançados e Experimentação
quinzena_start = project_start_date + timedelta(days=42)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Preparação para a próxima entrega acadêmica e alinhamento com a orientação.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Revisão do plano de projeto e alinhamento com os objetivos de agentes avançados.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Pesquisa aprofundada sobre Agentes Ponderados, Meta-Aprendizado e Otimização Bayesiana.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Definição de padrões para documentação de experimentos e resultados.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Documentação detalhada dos Agentes 2 e 3 (arquitetura, implementação).",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Planejamento de sprints para implementação e experimentação dos novos agentes.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Definição de metodologia experimental e seleção de datasets para os novos agentes.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 4",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Implementação dos Agentes 2 (Aleatório Ponderado) e 3 (Meta-Aprendizado).",
    "Responsável": ""
})

# Quinzena 5 (06/10/2025 - 19/10/2025): Otimização e Análise de Resultados
quinzena_start = project_start_date + timedelta(days=56)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Acompanhamento do progresso das atividades e comunicação de eventuais desvios.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Avaliação do progresso em relação aos objetivos e ajuste de rota, se necessário.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Pesquisa sobre Algoritmos Genéticos e Otimização Bayesiana para agentes.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Elaboração de templates para relatórios de experimentos e gráficos.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Documentação detalhada dos Agentes 4 e 5 (arquitetura, implementação).",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Gerenciamento do backlog de experimentos e análise de resultados preliminares.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Execução e monitoramento dos experimentos com os Agentes 2 e 3.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 5",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Implementação dos Agentes 4 (Algoritmo Genético) e 5 (Otimização Bayesiana).",
    "Responsável": ""
})

# Quinzena 6 (20/10/2025 - 02/11/2025): Redação Final e Preparação da Apresentação
quinzena_start = project_start_date + timedelta(days=70)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Revisão de requisitos para a entrega final do TCC.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Revisão geral do conteúdo do TCC e alinhamento com a banca.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Revisão da fundamentação teórica e contribuição para a seção de discussão.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Formatação final do TCC (ABNT) e preparação de slides para apresentação.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Consolidação da documentação técnica do projeto.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Finalização do backlog e garantia de que todos os requisitos foram atendidos.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Análise final dos resultados dos experimentos e redação da seção de resultados.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 6",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Finalização de artefatos visuais e código para a apresentação.",
    "Responsável": ""
})

# Quinzena 7 (03/11/2025 - 16/11/2025): Revisão Final e Entrega
quinzena_start = project_start_date + timedelta(days=84)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Submissão final do TCC e acompanhamento do processo.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Revisão final e aprovação do documento completo do TCC.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Revisão final da bibliografia e citações.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Revisão final da formatação, coesão e vídeo de apresentação.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Revisão final da documentação do software.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Verificação final de conformidade com os requisitos do projeto.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Revisão final da análise de resultados e conclusões.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 7",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Preparação final do ambiente para demonstração e apresentação.",
    "Responsável": ""
})

# Quinzena 8 (17/11/2025 - 30/11/2025): Apresentação e Pós-Entrega
quinzena_start = project_start_date + timedelta(days=98)
quinzena_end = quinzena_start + timedelta(days=13)
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão Acadêmica",
    "Tarefa": "Acompanhamento da apresentação e feedback da banca.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão do TCC",
    "Tarefa": "Participação na apresentação e defesa do TCC.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Acadêmica",
    "Tarefa": "Suporte na defesa teórica do TCC.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Formatação e Apresentação",
    "Tarefa": "Suporte técnico durante a apresentação.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Documentação de Software",
    "Tarefa": "Atualização da documentação pós-feedback.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Gestão da Proposta",
    "Tarefa": "Avaliação pós-projeto e lições aprendidas.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Pesquisa Científica",
    "Tarefa": "Discussão dos resultados com a banca.",
    "Responsável": ""
})
cronograma_data.append({
    "Quinzena": "Quinzena 8",
    "Início": quinzena_start,
    "Fim": quinzena_end,
    "Função": "Desenvolvimento de Software",
    "Tarefa": "Demonstração do agente e funcionalidades.",
    "Responsável": ""
})

# Create DataFrame
cronograma_df = pd.DataFrame(cronograma_data)

# Format dates to 'YYYY-MM-DD' for TSV
cronograma_df['Início'] = cronograma_df['Início'].dt.strftime('%Y-%m-%d')
cronograma_df['Fim'] = cronograma_df['Fim'].dt.strftime('%Y-%m-%d')

# Save the updated cronograma.tsv
cronograma_path = "D:\PROGRAMACAO\sklearn_rl\docs\cronograma.tsv"
cronograma_df.to_csv(cronograma_path, sep='\t', index=False)

print("Cronograma.tsv atualizado com o novo plano de gestão de projetos detalhado.")