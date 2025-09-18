import pandas as pd
from pathlib import Path

# Load existing data
try:
    df_existing = pd.read_csv('docs/registro_de_atividades.tsv', sep='\t', encoding='utf-8')
except FileNotFoundError:
    df_existing = pd.DataFrame(columns=['Data', 'Canal', 'Evento', 'Responsável', 'Função', 'Observações'])

# New activities to add
new_activities_data = [
    {"Data": "11/08/2025 09:00", "Canal": "Interno", "Evento": "Início do Projeto TCC - Sklearn RL Agent", "Responsável": "Elysio", "Função": "Gestão do TCC", "Observações": "Definição inicial de escopo e equipe."},
    {"Data": "15/08/2025 10:00", "Canal": "Interno", "Evento": "Criação da estrutura base da aplicação Streamlit", "Responsável": "Elysio", "Função": "Desenvolvimento de Software", "Observações": "Criação de `app.py`, `menu.py`, `functions.py`."},
    {"Data": "18/08/2025 11:00", "Canal": "Interno", "Evento": "Configuração inicial de arquivos de dados", "Responsável": "Elysio", "Função": "Gestão da Proposta", "Observações": "Criação de `files.tsv` e `registro_de_atividades.tsv`."},
    {"Data": "20/08/2025 12:00", "Canal": "Interno", "Evento": "Criação de documentos acadêmicos iniciais", "Responsável": "Elysio", "Função": "Formatação e Apresentação", "Observações": "Criação de `docs/relatorio_1_projeto.tsv` e `docs/formatacao.tsv`."},
    {"Data": "18/09/2025 06:12", "Canal": "Git", "Evento": "Implementação da página 'Formatação e Apresentação'", "Responsável": "Elysio", "Função": "Formatação e Apresentação", "Observações": "Inclusão da seção de relatório e estilos ABNT."},
    {"Data": "18/09/2025 15:00", "Canal": "Interno", "Evento": "Atualização de Artefatos e Requisitos em todas as páginas", "Responsável": "Elysio", "Função": "Documentação de Software", "Observações": "Revisão e detalhamento de inputs/outputs e requisitos do TCC/Proposta."},
    {"Data": "18/09/2025 16:00", "Canal": "Interno", "Evento": "Remoção de mensagens de seleção `st.info`", "Responsável": "Elysio", "Função": "Desenvolvimento de Software", "Observações": "Otimização da interface do usuário para reduzir poluição visual."}
]
df_new_activities = pd.DataFrame(new_activities_data)

# Concatenate and remove duplicates (in case of re-running)
df_combined = pd.concat([df_existing, df_new_activities]).drop_duplicates(subset=['Data', 'Evento'], keep='last')

# Convert 'Data' column to datetime for proper sorting
df_combined['Data'] = pd.to_datetime(df_combined['Data'], format='%d/%m/%Y %H:%M', errors='coerce')

# Sort by date
df_combined = df_combined.sort_values(by='Data', ascending=True)

# Convert 'Data' back to string format for saving
df_combined['Data'] = df_combined['Data'].dt.strftime('%d/%m/%Y %H:%M')

# Save the updated DataFrame
df_combined.to_csv('docs/registro_de_atividades.tsv', sep='\t', index=False, encoding='utf-8')

print("Registro de atividades atualizado com sucesso.")
