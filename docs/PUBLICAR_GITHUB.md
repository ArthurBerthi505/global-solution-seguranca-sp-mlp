# Repositório publicado no GitHub

Repositório público: https://github.com/ArthurBerthi505/global-solution-seguranca-sp-mlp

Aplicação funcional: https://seguranca-sp-mlp-gs568017.tiledtoffee16.chatgpt.site

## Clonar e executar

```powershell
git clone https://github.com/ArthurBerthi505/global-solution-seguranca-sp-mlp.git
cd global-solution-seguranca-sp-mlp
python -m http.server 8000 --directory web
```

Abra http://localhost:8000. Para reproduzir o treinamento e executar o notebook, siga o README.

## Enviar alterações posteriores

Depois de revisar e validar suas mudanças, use na cópia clonada:

```powershell
git add .
git commit -m "Descreva a alteracao realizada"
git push origin main
```

O `.gitignore` exclui dados brutos, registros e previsões por ocorrência, ambiente virtual e arquivos temporários. Preserve no repositório o código, o modelo, o notebook executado, os resultados agregados e o PDF de dez slides. Para gerar os CSVs por ocorrência na sua máquina, execute `python scripts/train.py`. Os totais e resultados são históricos de 2016, conforme a documentação.

O link da aplicação já está no README. A hospedagem atual é independente do GitHub, portanto enviar um commit não altera automaticamente a aplicação online. Para trocar de hospedagem, publique o conteúdo de `web/` preservando seus caminhos relativos e atualize o endereço no README.
