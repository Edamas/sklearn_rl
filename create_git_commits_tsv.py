import pandas as pd

git_log_output = """64215f745f38cf00fd41975bb0abbbae615ee2e5	Edamas	18/09/2025 06:36	feat: Adiciona expander à visualização do relatório formatado
b595d4be564d963ffe39c7140780cb68f540c374	Edamas	18/09/2025 06:27	fix: Adiciona relatorio_1_projeto e formatacao a files.tsv
2c501c2caa7128b6589a46a55317bc73e446f029	Edamas	18/09/2025 06:22	feat: Adiciona visualização de relatório formatado e atualiza estilos ABNT
bf29a8d962802231864156d22748ed3c12d05fd3	Edamas	18/09/2025 06:12	feat: Implementa seção de relatório e estilos ABNT e corrige seleção de temas de TCC
89d5550b5e4e323de561feb602f88cb488adac34	Edamas	18/09/2025 05:49	feat: Adiciona referências bibliográficas da disciplina de TCC
ed4c8e8b5cea8e172d4d91f9f98167eb918e6f17	Edamas	18/09/2025 05:41	fix: Adiciona função get_card_style e corrige FileNotFoundError
5d9111aa86f0b91a4a2d4f19a5f00bb739eedbae	Edamas	18/09/2025 05:39	fix: Usa barras normais em files.tsv para compatibilidade com Linux
857299ad22c1ce9b4f315bf224cc5389b6479d57	Edamas	18/09/2025 05:37	fix: Atualiza files.tsv para corrigir KeyError
a6c1307c1fde8468dfa914a4ca6f82215c884e70	Edamas	18/09/2025 05:35	feat: Adiciona seção de análise de temas de TCCs
33dbed11e25988cf97f26737fdb9438fa6da8555	Edamas	17/09/2025 19:15	feat: Add organogram and rubrics
8742f74caeaa483057836b269650b92a6bc4c355	Edamas	16/09/2025 21:18	Refactor: Content distribution, rubric updates, and bug fixes.
0d50c51fc1487895d1c7e4223d9a14b5aff17353	Edamas	16/09/2025 19:25	Protótipo v. 1.4
69e88ac039dc0bc35d09329a156237a7a009ab92	Edamas	14/09/2025 19:49	refactor: Reestruturação do projeto e melhoria da UI de features
826570054f3302209ce2194e2ccf59eb8ba66973	Edamas	12/09/2025 09:41	feat: Melhora a geração de hiperparâmetros e a exibição de resultados
c81923fa977b19cfa1f4aad65e631908baf1f5c0	Edamas	12/09/2025 09:26	protótipo 1.1
949b433117a02fc2d25c1ff8ae43dc729cc4464e	Edamas	10/09/2025 16:02	feat(data): Processa PatchExtractor e atualiza parâmetros
e779f924c333a415c066f3650f3366e542399ca0	Edamas	10/09/2025 13:01	chore(debug): Adiciona verificação para depurar erro de configuração
d1fe1ee11fba19869b78467f57264cd69a6b8eac	Edamas	10/09/2025 12:59	fix(app): Re-implementa manuseio robusto de caminhos para implantação
a051d420394162b549209df51a17515c4a94d363	Edamas	10/09/2025 12:58	fix(core): Corrige o manuseio de caminhos de arquivo para implantação
19ea5f26c4cb0e0eba97797bca0f9347579ed217	Edamas	10/09/2025 12:43	feat: Implementa protótipo v1.0
8dab055708bf044a198c91181b9f30cbe1cc00ef	Edamas	05/09/2025 19:17	fix: Adicionar dependências ausentes ao requirements.txt
0ee8ae4a66bbc1eab3b2e15a6134196cb5d0acfc	Edamas	05/09/2025 19:08	chore: Desvincular diretório __pycache__
0eb841e37a950d69e59df77303fd3a5371c737ef	Edamas	05/09/2025 19:07	fix(agent): Refatora e corrige a lógica de seleção de estimadores
e7485f95ac1c28bb195addfb2685b97d0aaea93a	Edamas	05/09/2025 18:05	implementação inicial do agente
c68533dfa1076609bccd9728d6ace359f6586627	Edamas	04/09/2025 23:21	fix: Remove redundant graph import from app.py. The plot_sklearn_graph function and its associated logic have been moved to datasets_app.py. This commit removes the now-redundant import statement from app.py to resolve ModuleNotFoundError issues.
d3b7365af072f053f06b3cf6f13d477a6b1523a5	Edamas	04/09/2025 22:56	feat: Refactor pipeline builder and enhance dataset analysis: Pipeline builder integrated into Datasets page. Enhanced dataset analysis with consolidated metrics and general metrics. Flexible dataset configuration with multi-select target and dynamic feature selection. Groundwork for visual pipeline construction laid. Aims to improve UX, streamline workflow, and provide robust foundation.
a739d903fe03c17ab0105787b425fa56d99ca059	Edamas	03/09/2025 09:25	Atualizando datasets e readme
b3e3352e108a5abbe74c2756ef80fe0bcf4f0367	Edamas	03/09/2025 08:04	Atualizando requirements.txt
bd534e8009d28e7962b57445f33cf2a8e1df03f3	Edamas	03/09/2025 08:01	Atualizando requirements.txt
bb6f60cce24012076539aa16379574995ea42367	Edamas	03/09/2025 08:00	Atualizando requirements.txt
1ec39a1db11481e9fd2e41357256712da7074063	Edamas	03/09/2025 06:30	Atualizando requirements.txt
af873f155392885bbfd5eb7f920dd416d0f10de1	Edamas	03/09/2025 06:24	Retirando método obsoleto all_estimators
35484f1daa881b4e816f5b906cef52ecf86b3d7c	Edamas	03/09/2025 06:21	Datasets adicionados
2fe1f9e3ff75d823c5388cf3be93efabd780808d	Edamas	03/09/2025 05:14	Exploração dos métodos do Scikit-Learn
c083a1f535ad220d6314b5faf663d2862c27c142	Edamas	03/09/2025 03:09	Atualização do header
47d6e3f0c67fec0283cddcea4a448335bd12f007	Edamas	03/09/2025 03:03	Atualização de rubricas.md
f3c631d6aad01ba15e81ffd585fc810095ab4874	Edamas	03/09/2025 02:53	Add new application files, update documentation, and ignore __pycache__
d2b9be4d8b84b39ead77ce04dbb5141d2112ddc9	Edamas	03/09/2025 00:16	Update documentation and format TCC
9779373ebe900013af4150eb0c268727b59e7f82	Edamas	03/09/2025 00:02	Initial commit"""

commits_data = []
for line in git_log_output.strip().split('\n'):
    parts = line.split('\t')
    if len(parts) == 4:
        commits_data.append({
            "Hash": parts[0],
            "Autor": parts[1],
            "Data": parts[2],
            "Mensagem": parts[3]
        })

df_commits = pd.DataFrame(commits_data)
df_commits.to_csv('docs/git_commits.tsv', sep='\t', index=False, encoding='utf-8')

print("Arquivo git_commits.tsv criado com sucesso.")
