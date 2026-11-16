# Proveniência dos dados

Coleção: **Crime Data in Brazil**. Publicador: **Marco Zanchi**, usuário Kaggle `inquisitivecrow`.

URL: https://www.kaggle.com/datasets/inquisitivecrow/crime-data-in-brazil

Arquivo: `BO_2016.csv`, obtido pelo endpoint de arquivo específico registrado no README. Ano de referência 2016, versão 1 da coleção, consulta em 12/11/2026.

O publicador relata que solicitou dados à SSP-SP por pedido de informação. A coleção inclui metodologia do sistema Registro Digital de Ocorrências e dicionário. Os documentos de origem estão em `RDO_methodology.txt` e `RDO_column_description.csv`, com conteúdo preservado e codificação convertida para UTF-8.

Licença declarada no catálogo: **GPL 2**. A cópia da licença acompanha o projeto em `LICENSE`. O arquivo `source_metadata.json` registra os metadados públicos da coleção. Os dados foram filtrados e agrupados, portanto o CSV em `data/processed/` é um derivado, não o arquivo original da fonte.

O campo `CIDADE` é cidade do registro no dicionário. O filtro usa departamento e seccional de circunscrição. A chave de deduplicação é exatamente a descrita na metodologia: ID_DELEGACIA, ANO_BO e NUM_BO. Não devemos somar linhas brutas como se fossem ocorrências independentes. Os totais desta análise não substituem os indicadores oficiais da SSP.

Os 20.000 registros derivados não incluem nome, RG, placa, logradouro, número do imóvel, coordenadas ou qualquer atributo pessoal. O hash da chave é utilizado somente para auditar a separação dos conjuntos, sem participar da MLP.

O ano de referência é antigo. A licença, a escolha da base e essa limitação temporal devem permanecer documentadas em qualquer publicação do derivado.

A distribuição pública não contém registros e previsões por ocorrência. Esses arquivos são gerados localmente pelo notebook ou por `scripts/train.py`, a partir da origem pública documentada. O repositório inclui resultados agregados e o modelo treinado.
