import pandas as pd
from datetime import timedelta

# Paths to the TSV files
registro_path = "D:\\PROGRAMACAO\\sklearn_rl\\docs\\registro_de_atividades.tsv"
cronograma_path = "D:\\PROGRAMACAO\\sklearn_rl\\docs\\cronograma.tsv"

# Load DataFrames
registro_df = pd.read_csv(registro_path, sep='\t')
cronograma_df = pd.read_csv(cronograma_path, sep='\t')

# Convert 'Data' column to datetime in registro_df
registro_df['Data'] = pd.to_datetime(registro_df['Data'], format='%d/%m/%Y %H:%M')

# Convert 'Início' column to datetime in cronograma_df
cronograma_df['Início'] = pd.to_datetime(cronograma_df['Início'], format='%d/%m/%Y')

# Calculate 'Fim' date for each quinzena (14 days after 'Início')
cronograma_df['Fim'] = cronograma_df['Início'] + timedelta(days=13)

# Keywords to identify significant activities
significant_keywords = [
    "Criação do grupo", "Primeira reunião", "Final das inscrições",
    "Convocação", "Criação da Equipe", "Adicionado Arquivo", "Atualização",
    "Enquete", "Postagem sobre discussão", "Preenchimento dos campos faltantes"
]

# Function to summarize activities for a given quinzena
def summarize_activities(start_date, end_date):
    activities_in_period = registro_df[
        (registro_df['Data'] >= start_date) & (registro_df['Data'] <= end_date)
    ]
    
    summaries = []
    for _, row in activities_in_period.iterrows():
        event = row['Evento']
        observacoes = row['Observações']
        
        # Check if the event or observation contains any significant keywords
        is_significant = any(keyword.lower() in event.lower() or \
                             (isinstance(observacoes, str) and keyword.lower() in observacoes.lower())
                             for keyword in significant_keywords)
        
        if is_significant:
            summaries.append(f"{row['Data'].strftime('%d/%m')}: {event}")
            
    if summaries:
        return "; ".join(summaries)
    return ""

# Update 'Atividade' column in cronograma_df
for index, row in cronograma_df.iterrows():
    start = row['Início']
    end = row['Fim']
    cronograma_df.loc[index, 'Atividade'] = summarize_activities(start, end)

# Drop the temporary 'Fim' column
cronograma_df = cronograma_df.drop(columns=['Fim'])

# Save the updated cronograma.tsv
cronograma_df.to_csv(cronograma_path, sep='\t', index=False)

print("Cronograma.tsv atualizado com sucesso!")
