

import pandas as pd

registro_path = "D:\\PROGRAMACAO\\sklearn_rl\\docs\\registro_de_atividades.tsv"
registro_df = pd.read_csv(registro_path, sep='\t')

def assign_function(row):
    evento = str(row['Evento']).lower()
    observacoes = str(row['Observações']).lower()
    responsavel = str(row['Responsável']).lower()

    if "univesp" in responsavel or "fórum ava" in evento or "inscrições" in evento or "orientadora" in evento or "disciplinas" in observacoes or "tcc" in evento and "grupo" not in evento:
        return "Gestão Acadêmica"
    if "criação do grupo" in evento or "criação da equipe" in evento or "convocação" in evento or "primeira reunião" in evento or "tabela de rh" in evento or "planner" in evento or "discussão de propostas" in evento or "enquete" in evento:
        return "Gestão do TCC"
    if "fontes bibliográficas" in evento or "pesquisar" in observacoes or "temas" in observacoes or "fundamentação" in observacoes:
        return "Pesquisa Acadêmica"
    if "tcc anteriores" in observacoes or "formatação" in observacoes or "layout" in observacoes or "resumos da quinzena" in observacoes:
        return "Formatação e Apresentação"
    if "documentação" in evento or "arquivos" in evento and ("cbo" in observacoes or "ppc" in observacoes or "perfis" in observacoes or "áreas de atuação" in observacoes):
        return "Documentação de Software"
    if "propostas" in evento or "escopo" in evento or "planner" in evento and "propostas" in observacoes:
        return "Gestão da Proposta"
    if "competições em ia" in observacoes or "científica" in evento or "agente" in observacoes:
        return "Pesquisa Científica"
    if "desenvolvimento" in evento or "implementação" in evento or "código" in observacoes or "artefatos" in observacoes:
        return "Desenvolvimento de Software"
    
    return "Não Atribuído" # Default if no clear match

registro_df['Função'] = registro_df.apply(assign_function, axis=1)

registro_df.to_csv(registro_path, sep='\t', index=False)

print("Coluna 'Função' adicionada e registro_de_atividades.tsv atualizado com sucesso.")

