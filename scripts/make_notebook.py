"""Gera notebook com implementações visíveis, não apenas imagens de código."""
from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
source_data = (ROOT / "src/data.py").read_text()
source_model = (ROOT / "src/model.py").read_text()


def function(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(source, node)


cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Ocorrência SP · Roubo ou furto com MLP
**Global Solution · Aprendizado Profundo com Redes Neurais Convolucionais**

| Integrante | RM |
|---|---|
| Arthur Casqueira Berthi | 568017 |
| Agatha Carolina | 568352 |
| João Dyonisio | 567087 |

**Objetivo:** classificar o tipo de uma ocorrência já registrada ou hipotética da capital paulista. A saída não é uma previsão de criminalidade futura nem uma decisão jurídica.

Este notebook executa três MLPs reais e preserva as saídas. O relatório e a aplicação usam os mesmos resultados. Para executar novamente: extraia o projeto, instale `requirements.txt` e abra este arquivo pelo Jupyter na pasta `notebooks/`.
""")
md("""
## 1. Fonte, recorte e decisões de projeto

Fonte pública efetivamente utilizada: [Crime Data in Brazil](https://www.kaggle.com/datasets/inquisitivecrow/crime-data-in-brazil), de Marco Zanchi (`inquisitivecrow`), arquivo **BO_2016.csv**, versão 1. O autor atribui a origem à SSP-SP e disponibiliza `RDO_methodology.txt` e `RDO_column_description.csv`. Os dois documentos acompanham o projeto em `docs/`. Consulta em 12/11/2026. Licença declarada no catálogo: GPL 2, registrada em `docs/source_metadata.json` e acompanhada de `LICENSE`.

A base é uma cópia histórica de dados RDO, diferente dos arquivos atuais de celulares/veículos da [SSP-SP](https://www.ssp.sp.gov.br/estatistica/consultas). O download direto do portal estava indisponível neste ambiente. O ano de 2016 é permitido pelo enunciado e permite demonstrar o fluxo completo, mas **não sustenta conclusões sobre São Paulo atual**.

**Atenção à geografia:** o dicionário descreve `CIDADE` como cidade de registro. A metodologia informa que o cidadão pode registrar um fato fora de sua circunscrição. Assim, o recorte usa `NOME_DEPARTAMENTO_CIRC = DECAP` e as oito `NOME_SECCIONAL_CIRC` da capital, que representam o local do fato. Não filtramos por delegacia de registro.

**Unidade de análise:** um BO identificado pela chave `(ID_DELEGACIA, ANO_BO, NUM_BO)`, conforme a metodologia RDO. Linhas com pessoas, objetos e naturezas podem repetir o mesmo boletim. Antes de amostrar, excluímos BOs com naturezas múltiplas, contextos/regiões conflitantes e registros sem chave válida. Consolidamos repetições e depois sorteamos 20.000 BOs, com semente 42 e proporção natural das classes.
""")
code("""
from pathlib import Path
import os, sys, json, hashlib, zipfile
from sklearn.model_selection import train_test_split
ROOT = Path.cwd().resolve()
if not (ROOT / 'src').is_dir():
    ROOT = ROOT.parent
assert (ROOT / 'src/data.py').exists(), 'Abra o notebook dentro do projeto extraído.'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display, Image, Markdown
from src.data import (KEY, REGIONS, CONTEXTS, LABELS, FEATURES, SEED,
                      SOURCE, DOWNLOAD, sha256, download_raw)
print('Python:', sys.version.split()[0])
print('NumPy:', np.__version__, '| pandas:', pd.__version__, '| scikit-learn:', sklearn.__version__)
print('Semente:', SEED)
""")
md("""
### Rótulo e entradas

| Papel | Campo da base | Transformação e motivo |
|---|---|---|
| Alvo | `RUBRICA` | Correspondência exata de Furto (art. 155) e Roubo (art. 157). Furto = 0, Roubo = 1. Excluímos outras naturezas, qualificadas e atos infracionais. |
| Entrada 1 | `NOME_SECCIONAL_CIRC` | Oito regiões administrativas da ocorrência. Localização agregada, sem endereços ou coordenadas. |
| Entrada 2 | `CONDUTA` | Dez contextos de ambiente/situação geral. Mapeamento explícito, inspecionável no código. |

**CONDUTA exige cuidado:** trata-se de um campo cadastral da ocorrência, que pode trazer classificações operacionais. O agrupamento conserva apenas ambientes/contextos e exclui “Saidinha de Banco”, cuja semântica sugere roubo. Nenhum campo com violência, ameaça, texto jurídico, `RUBRICA`, `DESDOBRAMENTO`, `FLAG_STATUS`, identificador ou dado pessoal entra na MLP. Em uma expansão do projeto, é recomendável coletar um tipo de local independente da natureza jurídica e testar a estabilidade do resultado sem `CONDUTA`.

Utilizamos apenas registros consumados (`FLAG_STATUS = C`). Campos ausentes de contexto recebem “Não informado”. Essa regra é fixa e não depende do teste. No recorte final, não há contexto ausente. Não usamos dia da semana nem horário porque o arquivo escolhido não oferece esses campos com precisão suficiente para essa decisão.
""")
code("""
print('Regiões de circunscrição:', REGIONS)
display(pd.DataFrame(CONTEXTS.items(), columns=['CONDUTA original', 'contexto de entrada']))
print('Entradas permitidas:', FEATURES)
assert 'RUBRICA' not in FEATURES and 'alvo' not in FEATURES
""")
md("""
## 2. Preparação implementada

O código abaixo é a mesma função de `src/data.py`. Na primeira execução, o notebook baixa a origem pública e gera a amostra de 20.000 BOs localmente. O download do ZIP tem cerca de 42 MB e o CSV descompactado cerca de 319 MB. As próximas execuções podem reutilizar a amostra local. Para forçar a preparação novamente, altere `REBUILD_FROM_SOURCE` para `True`.

Os arquivos com registros e previsões por ocorrência não acompanham a distribuição pública. O código, o modelo treinado e os resultados agregados permitem reproduzir e avaliar o experimento.

O hash SHA-256 do ZIP original e a contagem em cada filtro estão em `data/processed/auditoria.json`. O identificador exportado é um hash da chave do BO, usado somente para auditar a divisão dos conjuntos.
""")
code(function(source_data, "prepare_data"))
code("""
REBUILD_FROM_SOURCE = False
sample_path = ROOT / 'data/processed/amostra_capital_2016.csv'
if REBUILD_FROM_SOURCE or not sample_path.exists():
    frame, audit = prepare_data(download_raw())
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(sample_path, index=False)
    (sample_path.parent / 'auditoria.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
else:
    frame = pd.read_csv(sample_path)
    audit = json.loads((ROOT / 'data/processed/auditoria.json').read_text(encoding='utf-8'))
assert len(frame) == 20_000 and frame.id_grupo.nunique() == len(frame)
display(pd.DataFrame([(k, v) for k, v in audit.items() if isinstance(v, int)], columns=['Etapa', 'Quantidade']))
print('SHA-256 do ZIP:', audit['raw_sha256'])
print('Dimensões da amostra local:', frame.shape)
print('Entradas utilizadas:', FEATURES)
""")
md("""
## 3. Exploração

A distribuição de classes é praticamente equilibrada. A amostra preserva essa distribuição sem superamostragem ou balanceamento artificial. O cruzamento a seguir é descritivo e não define regras manuais para o classificador. Repetições de combinações de entradas são esperadas: boletins diferentes podem ocorrer na mesma região e no mesmo contexto.
""")
code("""
display(frame.alvo.map({0:'Furto',1:'Roubo'}).value_counts().rename_axis('Classe').to_frame('Boletins'))
display(pd.crosstab(frame.contexto, frame.alvo.map({0:'Furto',1:'Roubo'})))
print('Ausentes nas entradas:', frame[FEATURES].isna().sum().to_dict())
""")
md("""
## 4. Separação e pré-processamento sem vazamento

A divisão estratificada reserva 70% para treino, 15% para validação e 15% para teste. Como consolidamos uma linha por BO antes dessa divisão, o mesmo boletim não aparece em dois conjuntos. Uma asserção verifica a separação pelas chaves em hash.

O encoder **one-hot** aprende suas categorias apenas no treino. Não impomos uma ordem numérica a regiões ou contextos. As 18 entradas binárias já estão na mesma escala, dispensando padronização. Categorias externas ao treino podem ser codificadas como zeros com `handle_unknown='ignore'`; o formulário publicado oferece somente as categorias conhecidas e exibe o suporte de cada combinação no treino.
""")
code(function(source_data, "split_data"))
code("""
from sklearn.preprocessing import OneHotEncoder
train, val, test = split_data(frame)
summary = pd.DataFrame({name: d.alvo.value_counts().sort_index() for name,d in [('Treino',train),('Validação',val),('Teste',test)]})
summary.index = ['Furto', 'Roubo']
display(summary)
encoder_preview = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoder_preview.fit(train[FEATURES])
print('Dimensão one-hot:', len(encoder_preview.get_feature_names_out()))
print('Ajuste do encoder: apenas os', len(train), 'boletins de treino.')
""")
md(r"""
## 5. MLP: funcionamento e configuração

Para cada camada oculta, a rede calcula $h=\mathrm{ReLU}(XW+b)$, com $\mathrm{ReLU}(z)=\max(0,z)$. A saída única usa $p=\sigma(z)=1/(1+e^{-z})$ para representar a probabilidade atribuída à classe **Roubo**. A de **Furto** é $1-p$. O limiar fixo de decisão é 0,5.

A função de perda é a entropia cruzada binária, $-\frac{1}{n}\sum[y\log p+(1-y)\log(1-p)]$, com regularização L2 dos pesos no treinamento. Os gráficos mostram somente a parcela de entropia cruzada. Adam ajusta os pesos usando gradientes e momentos. ReLU permite aprender relações não lineares entre as categorias, enquanto sigmoid corresponde à classificação binária.

Todas as configurações usam Adam, taxa inicial **0,001**, **batch size 128**, até **80 épocas**, `shuffle=True` e semente **42**. A validação externa controla a parada: sem melhora de pelo menos **0,0001** na perda por **12 épocas**, o treino termina e restaura o checkpoint escolhido. `partial_fit` executa uma época e preserva o estado do otimizador entre chamadas. Não criamos outra validação interna.

Treinar uma MLP de camadas densas é coerente com os dados tabulares do desafio. Não há imagens nem convoluções nesse recorte.
""")
code(source_model.split("def metrics(")[0])
code(function(source_model, "metrics"))
md("""
## 6. Três arquiteturas

| Arquitetura | Camadas ocultas | Ativação | L2 (alpha) | Hipótese |
|---|---|---|---|---|
| MLP A | 16 | ReLU | 0,0001 | Referência simples, com menos parâmetros. |
| MLP B | 32, 16 | ReLU | 0,0001 | Capacidade maior para interações entre entradas. |
| MLP C | 64, 32, 16 | ReLU | 0,01 | Mais profundidade e regularização maior. |

Mantemos entradas e conjuntos constantes. Mudanças simultâneas de profundidade e L2 em C impedem atribuir a diferença a um único fator, portanto a comparação é entre configurações completas. O desenho não é um estudo causal de hiperparâmetros.
""")
code(function(source_model, "train_one"))
md("""
## 7. Seleção pela validação e avaliação final

Cada arquitetura fornece seu checkpoint pela perda de validação. Entre esses checkpoints, selecionamos o maior **F1 macro da validação**. O desempate considera menor perda e menos parâmetros. Precision, recall e F1 macro calculam a métrica para cada classe e depois fazem a média simples, dando peso igual a furto e roubo.

O teste fica reservado até essa seleção. Não geramos uma tabela de teste para as três redes. Apenas a rede selecionada e a referência que sempre estima a classe majoritária são avaliadas no teste final.
""")
code(function(source_model, "run_experiments"))
code("""
bundle, report, histories, membership, predictions = run_experiments(frame)
comparison = pd.DataFrame([{'Modelo':r['name'], 'Camadas':str(r['hidden_layers']), 'Parâmetros':r['parameters'],
    'Época escolhida':r['best_epoch'], 'Épocas executadas':r['epochs_run'],
    'Acurácia':r['validation']['accuracy'], 'Precision macro':r['validation']['precision_macro'],
    'Recall macro':r['validation']['recall_macro'], 'F1 macro':r['validation']['f1_macro']}
    for r in report['configs']])
display(comparison.round(4))
print('Selecionado somente pela validação:', report['selected'])
""")
code("""
display(pd.DataFrame({'MLP selecionada': report['test'], 'Classe majoritária': report['baseline']['test']}))
display(pd.DataFrame(report['per_class_test']).T)
print('Matriz de confusão [linhas reais; colunas estimadas; ordem Furto, Roubo]:')
print(np.array(report['confusion_matrix']))
""")
md("""
## 8. Curvas, matriz de confusão e interpretação dos resultados

Os gráficos usam todas as épocas executadas. A linha tracejada identifica o checkpoint restaurado, que pode preceder a parada. Uma perda de treino cada vez menor acompanhada de piora na validação sugeriria overfitting. Perdas elevadas e semelhantes podem refletir capacidade insuficiente ou falta de informação útil nas entradas, sem que este experimento consiga distinguir essas causas.
""")
code("""
from src.plots import make_figures
make_figures(frame, report, histories, ROOT / 'reports/figures')
display(Image(filename=str(ROOT / 'reports/figures/exploracao.png')))
display(Image(filename=str(ROOT / 'reports/figures/comparacao_curvas.png')))
display(Image(filename=str(ROOT / 'reports/figures/curvas_modelo_selecionado.png')))
display(Image(filename=str(ROOT / 'reports/figures/matriz_confusao.png')))
selected = next(r for r in report['configs'] if r['name']==report['selected'])
gap = selected['training']['accuracy'] - selected['validation']['accuracy']
display(Markdown(f"**Análise:** {report['selected']} apresenta acurácia de treino de {selected['training']['accuracy']:.2%} e validação de {selected['validation']['accuracy']:.2%}, diferença de {gap*100:.2f} pontos percentuais. Não há gap grande de generalização nessa divisão. O desempenho moderado pode refletir o limite de informação das duas entradas. As diferenças de F1 entre as três redes são pequenas e não demonstram superioridade estatística. No teste final, a acurácia é {report['test']['accuracy']:.2%} e o F1 macro é {report['test']['f1_macro']:.4f}."))
""")
md("""
## 9. Exportação e integração web

Salvamos o modelo Python junto do encoder em `models/mlp_bundle.joblib` e exportamos as mesmas matrizes e vieses para `web/model.json`. A função JavaScript em `web/inference.js` executa one-hot, multiplicações de matrizes, ReLU e sigmoid. Nenhum número de saída é escolhido manualmente.

A aplicação não recebe dados pessoais, não exige API paga e executa a classificação no navegador. O teste de integração compara Python e JavaScript em **todas as 80 combinações conhecidas**, além de uma entrada desconhecida. A tolerância numérica é de $10^{-12}$.
""")
code(function(source_model, "export_model"))
code(function(source_model, "save_outputs"))
code("""
save_outputs(bundle, report, histories, membership, predictions)
from scripts.build_web import build_web
build_web()
example = pd.DataFrame([{'regiao':'Norte','contexto':'Transeunte'}])
proba = bundle['model'].predict_proba(bundle['encoder'].transform(example[FEATURES]))[0]
print('Exemplo hipotético:', example.to_dict('records')[0])
print(f'Probabilidades da classe: Furto={proba[0]:.4%}, Roubo={proba[1]:.4%}')
print('Interpretação: dado que uma ocorrência aconteceu, estas são as probabilidades das classes.')
""")
code("""
import subprocess
verification = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                              cwd=ROOT, text=True, capture_output=True)
print(verification.stdout + verification.stderr)
assert verification.returncode == 0
""")
md("""
## 10. Conclusões, limitações e uso de IA generativa

A MLP aprende associações nos dados tabulares e supera a referência de classe majoritária neste recorte. A arquitetura mais profunda não apresentou ganho relevante. A rede menor tem 321 parâmetros e permite uma aplicação leve. A escolha decorre da validação, sem busca de hiperparâmetros no teste.

**Limitações:** base de 2016, cobertura e preenchimento heterogêneos, subnotificação, rótulos administrativos, duas entradas agregadas, possíveis diferenças de preenchimento de `CONDUTA` por natureza, ausência de calibração de probabilidades e uma única semente/divisão. Não usamos variáveis de violência ou ameaça, então o modelo não faz enquadramento jurídico. Os totais de BOs não equivalem à estatística criminal oficial, como informa a metodologia RDO.

**Próximos experimentos:** usar base mais recente com tipo de local independente da natureza, fazer ablação sem `CONDUTA`, repetir sementes e validar estabilidade temporal. Essas extensões exigem novo protocolo com teste futuro reservado, sem reutilizar este teste para decidir alterações.

**IA generativa:** ChatGPT/Codex auxiliou na escrita do código, preparação da interface, documentação e relatório. Os registros e os resultados exibidos são provenientes do processamento e treinamento executados. A equipe deve revisar o notebook e explicar as decisões, a chave de deduplicação, a codificação, a função de perda e os limites de interpretação. O guia `docs/GUIA_APRESENTACAO.md` apoia essa revisão.

**Execução web:** `python -m http.server 8000 --directory web`, depois abra `http://localhost:8000`. O endereço publicado está no README. `docs/CHECKLIST_ENTREGA.md` relaciona os materiais e o estado de publicação.
""")

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {"kernelspec": {"display_name":"Python 3", "language":"python", "name":"python3"},
               "language_info":{"name":"python","version":"3.12"}}
nbf.write(nb, ROOT / "notebooks/GlobalSolution_MLP_SP.ipynb")
print('Notebook criado:', len(cells), 'células')
