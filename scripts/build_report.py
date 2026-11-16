"""Relatório técnico em PDF, formato 16:9 e exatamente dez slides."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/Relatorio_GlobalSolution_MLP_SP.pdf"
r = json.loads((ROOT / "reports/metrics.json").read_text())
a = json.loads((ROOT / "data/processed/auditoria.json").read_text())
sel = next(x for x in r["configs"] if x["name"]==r["selected"])
W, H = 960, 540
NAVY, TEAL, MUTED, LINE = "#172D43", "#087F8C", "#5D7283", "#D9E3E8"
from matplotlib import font_manager
FONT_DIR = Path(font_manager.findfont('DejaVu Sans')).parent
pdfmetrics.registerFont(TTFont('DejaVu',str(FONT_DIR/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(FONT_DIR/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
c = canvas.Canvas(str(OUTPUT), pagesize=(W,H))
c.setTitle('Global Solution - Classificação de ocorrências com MLP')
c.setAuthor('Arthur Casqueira Berthi; Agatha Carolina; João Dyonisio')


def text(s,x,y,size=18,color=NAVY,bold=False):
    c.setFont('DejaVu-Bold' if bold else 'DejaVu',size)
    c.setFillColor(colors.HexColor(color))
    c.drawString(x,y,s)


def p(s,x,y,width=864,size=18,color=NAVY,leading=None):
    style = ParagraphStyle('body',fontName='DejaVu',fontSize=size,
                           leading=leading or size*1.45,textColor=colors.HexColor(color))
    paragraph = Paragraph(s,style)
    pw,ph = paragraph.wrap(width,H)
    if y-ph < 42:
        raise ValueError(f'Texto ultrapassa rodapé: {s[:80]}, y final = {y-ph}')
    paragraph.drawOn(c,x,y-ph)
    return y-ph


def header(title,number):
    c.setFillColor(colors.white);c.rect(0,0,W,H,fill=1,stroke=0)
    text('OCORRÊNCIA SP',48,510,10,TEAL,True)
    text(title,48,459,30,NAVY,True)
    c.setStrokeColor(colors.HexColor(LINE));c.setLineWidth(1);c.line(48,438,912,438)
    text('Global Solution · Aprendizado Profundo · FIAP',48,20,9,MUTED)
    text(f'{number:02d} / 10',858,20,9,MUTED)


def table(rows, widths, x=48,y=410,font=16,selected=None):
    t = Table(rows,colWidths=widths,hAlign='LEFT')
    style = [('FONTNAME',(0,0),(-1,-1),'DejaVu'),('FONTSIZE',(0,0),(-1,-1),font),
             ('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'),('TEXTCOLOR',(0,0),(-1,-1),colors.HexColor(NAVY)),
             ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EDF3F6')),
             ('BOTTOMPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),12),
             ('LINEBELOW',(0,0),(-1,-1),.5,colors.HexColor(LINE)),
             ('VALIGN',(0,0),(-1,-1),'MIDDLE')]
    if selected is not None:
        style += [('BACKGROUND',(0,selected),(-1,selected),colors.HexColor('#E4F2F2')),
                  ('FONTNAME',(0,selected),(-1,selected),'DejaVu-Bold')]
    t.setStyle(TableStyle(style))
    _,height=t.wrap(W,H);t.drawOn(c,x,y-height)
    return y-height


def image(path,x,y,width,height):
    c.drawImage(ImageReader(str(path)),x,y,width=width,height=height,preserveAspectRatio=True,anchor='c',mask='auto')


def end(): c.showPage()
def percent(x):return f'{100*x:.2f}%'.replace('.',',')

# 1. Projeto e equipe
c.setFillColor(colors.HexColor(NAVY));c.rect(0,0,W,H,fill=1,stroke=0)
text('GLOBAL SOLUTION · FIAP',52,489,12,'#85D6D0',True)
text('Segurança pública',52,399,43,'#FFFFFF',True)
text('inteligente em São Paulo',52,343,38,'#FFFFFF',True)
p('Classificação de ocorrências de <b>roubo ou furto</b><br/>com uma rede neural MLP',54,294,820,24,'#D5E6EE')
p('Arthur Casqueira Berthi · RM 568017<br/>Agatha Carolina · RM 568352<br/>João Dyonisio · RM 567087',54,167,820,17,'#FFFFFF')
text('Aprendizado Profundo com Redes Neurais Convolucionais',54,49,13,'#B9CEDA')
text('01 / 10',858,20,9,'#B9CEDA');end()

# 2. Problema e fonte
header('Problema, fonte e recorte',2)
p('<b>Objetivo:</b> estimar a classe de um fato já registrado ou hipotético. A saída não calcula risco futuro de crime.',48,416,size=19)
p('<b>Base:</b> Crime Data in Brazil, Marco Zanchi, arquivo BO_2016.csv. Cópia histórica de dados RDO atribuídos à SSP-SP. Ano de referência: <b>2016</b>.',48,342,size=18)
p(f'<b>Recorte do fato:</b> DECAP e oito seccionais de circunscrição da capital. De {a["raw_rows"]:,} linhas brutas, {a["capital_rows"]:,} pertencem a esse recorte e {a["eligible_unique_bulletins"]:,} BOs ficam elegíveis após os filtros.'.replace(',','.'),48,265,size=18)
p('<b>Amostra:</b> 20.000 boletins únicos. Entradas: região de circunscrição e contexto de CONDUTA. Localização agregada e ambiente permitem testar associações sem dados pessoais.',48,173,size=18)
p('<link href="https://www.kaggle.com/datasets/inquisitivecrow/crime-data-in-brazil" color="#087F8C">Fonte e download: kaggle.com/datasets/inquisitivecrow/crime-data-in-brazil</link><br/><link href="https://www.ssp.sp.gov.br/estatistica/consultas" color="#087F8C">Referência institucional: ssp.sp.gov.br/estatistica/consultas</link> · Acesso em 12/11/2026',48,87,size=10.5,leading=15)
end()

# 3. Preparação
header('Preparação e isolamento dos conjuntos',3)
rows=[['Etapa','Decisão aplicada'],['Geografia','Circunscrição do fato. CIDADE é local do registro.'],['Alvo','Furto = 0 e Roubo = 1, com natureza exata e status C.'],['Consolidação','Chave: delegacia + ano + número do boletim.'],['Consistência','Excluir naturezas múltiplas e entradas conflitantes.'],['Entradas','8 regiões + 10 contextos, codificados por one-hot.']]
table(rows,[160,704],font=14,y=416)
p('<b>Ausências:</b> contexto faltante recebe “Não informado”. Não houve casos na amostra. As entradas binárias dispensam scaler. Encoder ajustado somente no treino.',48,157,size=16)
p('<b>Treino 14.000</b> · <b>Validação 3.000</b> · <b>Teste 3.000</b><br/>Semente 42, amostragem e divisão estratificadas, sem BOs compartilhados.',48,93,size=16)
end()

# 4. Rede selecionada
header('Arquitetura da MLP selecionada',4)
table([['Camada','Dimensão','Função'],['Entrada','18 valores','8 regiões e 10 contextos, em one-hot'],['Oculta densa','16 neurônios','ReLU: max(0, XW + b)'],['Saída densa','1 neurônio','Sigmoid: probabilidade atribuída a Roubo']],[220,170,474],font=16,y=414)
p('<b>321 parâmetros treináveis:</b> (18 × 16 + 16) + (16 × 1 + 1).<br/>A rede aprende pesos e vieses para combinar as entradas.',48,208,size=19)
p('<b>Decisão:</b> p(Roubo) ≥ 0,5 estima Roubo. Caso contrário, Furto.<br/>A probabilidade atribuída a Furto é 1 − p(Roubo).',48,126,size=18)
p('Duas classes não exigem dois neurônios de saída. Uma sigmoid é suficiente para classificação binária.',48,62,size=12,color=MUTED)
end()

# 5. Treinamento
header('Configuração do treinamento',5)
table([['Componente','Configuração e motivo'],['Perda','Entropia cruzada binária + penalização L2.'],['Otimizador','Adam, taxa inicial 0,001.'],['Mini-batch','128 boletins. Embaralhamento a cada época.'],['Duração','Até 80 épocas, semente 42.'],['Parada','12 épocas sem melhora de 0,0001 na perda de validação.']],[210,654],font=16,y=414)
p('<b>Checkpoint:</b> restaurar os pesos escolhidos pela perda de validação.<br/><b>Modelo A:</b> checkpoint na época 48, parada na época 60.',48,139,size=18)
p('partial_fit executa uma época e preserva o estado Adam. O teste não controla a parada nem a escolha da rede.',48,63,size=12,color=MUTED)
end()

# 6. Experimentos
header('Três arquiteturas e suas hipóteses',6)
table([['Modelo','Camadas ocultas','L2 (alpha)','Parâmetros'],['MLP A','16','0,0001','321'],['MLP B','32, 16','0,0001','1.153'],['MLP C','64, 32, 16','0,01','3.841']],[170,285,205,204],font=18,y=414)
p('<b>A:</b> referência simples, com uma camada oculta.<br/><b>B:</b> maior capacidade para interações de região e contexto.<br/><b>C:</b> maior profundidade e L2 mais forte para conter overfitting.',48,231,size=19)
p('As três configurações compartilham dados, divisão, one-hot, ReLU, sigmoid, Adam e limiar. Em C, profundidade e L2 mudam juntas: não isolamos o efeito de cada fator.',48,113,size=16,color=MUTED)
end()

# 7. Resultados comparativos
header('Comparação na validação e escolha',7)
rows=[['Modelo','Acurácia','Precision','Recall','F1 macro']]
for m in r['configs']:
 v=m['validation'];rows.append([m['name'],percent(v['accuracy']),percent(v['precision_macro']),percent(v['recall_macro']),percent(v['f1_macro'])])
table(rows,[160,176,176,176,176],font=17,y=415,selected=1)
p('<b>Seleção: MLP A</b>, pelo maior F1 macro na validação.<br/>A e C quase empatam. A também tem menos parâmetros. Uma divisão não comprova superioridade estatística.',48,224,size=18)
p(f'<b>Teste final da MLP A:</b> acurácia {percent(r["test"]["accuracy"])}, precision {percent(r["test"]["precision_macro"])}, recall {percent(r["test"]["recall_macro"])} e F1 {percent(r["test"]["f1_macro"])}.',48,128,size=18)
p('Precision, recall e F1 usam média macro. Referência de classe majoritária no teste: acurácia 50,43% e F1 macro 33,53%.',48,62,size=12,color=MUTED)
end()

# 8. Evidências
header('Curvas e matriz de confusão',8)
image(ROOT/'reports/figures/curvas_modelo_selecionado.png',39,214,575,210)
image(ROOT/'reports/figures/matriz_confusao.png',633,177,285,257)
p('<b>Treino:</b> 72,50% de acurácia. <b>Validação:</b> 72,13%.<br/>Gap de 0,37 ponto percentual no checkpoint: sem sinal forte de overfitting nesta divisão.',48,187,width=548,size=16)
p('<b>Teste:</b> 2.150 acertos / 3.000 BOs.<br/>Recall F/R: 76,01% / 67,25%.',632,154,width=280,size=14)
p('O desempenho moderado em treino e validação pode refletir informação limitada nas duas entradas. Aumentar a rede não garante resolver esse limite.',48,87,size=14.5,color=MUTED)
end()

# 9. Aplicação real
header('Aplicação web integrada ao modelo',9)
shot=ROOT/'reports/figures/interface_classificacao.jpg'
if shot.exists(): image(shot,42,84,596,339)
else: p('Captura da interface indisponível.',48,400,width=560,size=20)
p('<b>Formulário</b><br/>Região e contexto conhecidos no treino.<br/><br/><b>Inferência real</b><br/>One-hot, pesos, ReLU e sigmoid em JavaScript.<br/><br/><b>Saída</b><br/>Classe, duas probabilidades e suporte no treino.',658,410,width=250,size=16)
p('<link href="https://seguranca-sp-mlp-gs568017.tiledtoffee16.chatgpt.site" color="#087F8C">seguranca-sp-mlp-gs568017.tiledtoffee16.chatgpt.site</link><br/>Python e JavaScript conferidos nas 80 combinações conhecidas, com tolerância de 10⁻¹².',48,77,size=11,leading=16)
end()

# 10. Conclusões e IA
header('Conclusões, limites e uso de IA',10)
p('<b>Resultado:</b> uma MLP funcional, três configurações comparadas e inferência web com os mesmos pesos do treinamento. A rede simples teve o maior F1 de validação.',48,411,size=19)
p('<b>Limitações:</b> dados de 2016, subnotificação e preenchimento desigual. Duas entradas não descrevem violência ou ameaça. CONDUTA pode sofrer influência cadastral da natureza. Probabilidades sem calibração e uma única semente/divisão.',48,319,size=18)
p('<b>Uso adequado:</b> demonstração acadêmica de classificação. Os BOs não equivalem à estatística oficial. O modelo não estima chance de crime nem faz enquadramento jurídico.',48,204,size=18)
p('<b>IA generativa:</b> ChatGPT/Codex apoiou código, interface e documentação. Os resultados vêm de execução real. O grupo deve revisar e explicar as decisões e os testes antes da apresentação.',48,113,size=17)
end()
c.save()
print(OUTPUT)
