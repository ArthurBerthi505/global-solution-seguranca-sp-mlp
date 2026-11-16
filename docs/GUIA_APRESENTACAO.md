# Guia de revisão e apresentação

O grupo deve executar a aplicação e revisar o notebook antes da entrega. Este guia explica as decisões sem presumir que essa revisão já ocorreu.

## Sugestão de divisão da fala

| Integrante | Slides | Pontos para estudar |
|---|---|---|
| Arthur Casqueira Berthi | 1 a 3 | Objetivo, fonte, geografia do fato, chave do BO e preparação. |
| Agatha Carolina | 4 a 6 | One-hot, neurônios, funções de ativação, perda, Adam e experimentos. |
| João Dyonisio | 7 a 10 | Validação/teste, métricas, gráficos, interface e limitações. |

Essa é uma sugestão de organização. Todos devem compreender a solução inteira.

## Perguntas e respostas para revisão

**O que o modelo prevê?** A classe de uma ocorrência, condicionada a ela já ter acontecido ou ser um exemplo hipotético. A probabilidade de Roubo não é a probabilidade de acontecer um crime.

**Por que dados de 2016?** O enunciado permite escolher o ano. A coleção pública histórica ofereceu dados e documentação para executar o projeto após falha no download direto do portal. A limitação temporal está explícita. Não alegamos retratar a cidade atual.

**Por que não filtrar CIDADE = São Paulo?** No dicionário ela é o local do registro. Um cidadão pode registrar um fato fora da região em que aconteceu. O filtro usa DECAP e as oito seccionais `_CIRC`, que descrevem a circunscrição do fato.

**Por que remover duplicatas antes de dividir?** O mesmo BO pode gerar várias linhas. Se essas linhas fossem para treino e teste, os conjuntos deixariam de ser independentes por ocorrência. A chave reúne delegacia, ano e número do BO.

**Como evitamos vazamento?** RUBRICA define apenas o alvo. As entradas são região e contexto. Excluímos naturezas múltiplas e o contexto “Saidinha de Banco”. O encoder aprende só no treino. Validação escolhe checkpoint e rede. O teste fica reservado. CONDUTA ainda é um campo cadastral, então sua independência de preenchimento em relação à natureza é uma limitação que merece ablação em trabalhos futuros.

**Por que 18 entradas para só duas variáveis?** One-hot cria uma posição para cada categoria observada no treino: oito regiões e dez contextos. Um exemplo ativa uma região e um contexto. Nenhuma ordem numérica artificial existe entre eles.

**O que faz um neurônio?** Calcula uma soma ponderada das entradas, adiciona um viés e passa o resultado por uma ativação. A camada oculta usa ReLU para permitir relações não lineares. A saída sigmoid fica entre zero e um.

**Por que um neurônio de saída para duas classes?** A saída representa p(Roubo). A de Furto é seu complemento. O limiar 0,5 é fixo.

**O que é entropia cruzada?** Uma perda que penaliza probabilidades incompatíveis com o rótulo real, especialmente previsões erradas muito confiantes. Adam modifica os pesos para reduzir a perda, usando gradientes e momentos.

**Como chegamos a 321 parâmetros?** Entrada 18 para camada 16: 18×16 pesos e 16 vieses. Camada 16 para saída 1: 16 pesos e 1 viés. Total 321.

**Por que não usar scaler?** As variáveis finais são binárias, todas em 0 ou 1. Não há uma grandeza numérica contínua com escala diferente.

**Por que três arquiteturas?** Para comparar capacidade e regularização, como exige o enunciado. A usa 16 neurônios; B usa 32 e 16; C usa 64, 32 e 16, com L2 maior. O resultado não mostrou ganho relevante de profundidade.

**Por que A se C tem acurácia de validação maior?** O critério definido foi F1 macro. A teve 0,7205 e C 0,7204, uma diferença muito pequena. A é a escolhida por esse critério e possui menos parâmetros. Não há evidência de superioridade estatística.

**O que é F1 macro?** Calculamos F1 separadamente para Furto e Roubo e tiramos a média simples. Assim, cada classe tem o mesmo peso.

**O que a matriz mostra?** Dos 1.513 furtos do teste, a rede acertou 1.150 e estimou 363 como roubo. Dos 1.487 roubos, acertou 1.000 e estimou 487 como furto. Ao todo, 2.150 de 3.000 acertos, ou 71,67%.

**Existe overfitting?** O gap entre acurácias de treino e validação no checkpoint é de 0,37 ponto percentual, sem sinal forte nessa divisão. O teto moderado pode ser limite das entradas. Não concluímos a causa apenas pelas curvas.

**Por que não testar as três redes no teste e escolher a melhor?** Isso usaria o teste como validação, deixando a avaliação final otimista. Escolhemos com validação e depois testamos somente a rede selecionada.

**A web usa uma simulação?** Ela usa as mesmas matrizes e vieses exportados do modelo treinado. O teste de integração compara Python e JavaScript em 80 combinações conhecidas e uma desconhecida.

**Como a IA generativa ajudou?** Na escrita do código, interface, documentação e relatório. Os dados vieram de uma fonte pública e as métricas vieram da execução real. A equipe responde pela revisão e compreensão.

## Roteiro breve da demonstração

1. Mostre a mensagem de interpretação na tela inicial.
2. Clique em Transporte coletivo: Centro + Transporte coletivo estima Furto, aproximadamente 97,1%.
3. Clique em Transeunte: Norte + Transeunte estima Roubo, aproximadamente 77,4%.
4. Explique que as porcentagens se referem às classes de uma ocorrência, não ao risco do local.
5. Abra Experimentos e resultados e mostre a seleção na validação e o teste final.
6. Abra Dados e método para mostrar fonte histórica, circunscrição e limitações.
