"use strict";
let model, metadata;
const $ = id => document.getElementById(id);
const pct = value => (value * 100).toLocaleString("pt-BR", {minimumFractionDigits: 1, maximumFractionDigits: 1}) + "%";

document.querySelectorAll("[data-tab]").forEach(button => {
  button.addEventListener("click", () => {
    document.querySelectorAll("[data-tab]").forEach(tab => {
      const active = tab === button;
      tab.classList.toggle("active", active);
      tab.setAttribute("aria-selected", String(active));
    });
    document.querySelectorAll(".tab-panel").forEach(panel => { panel.hidden = panel.id !== button.dataset.tab; });
    history.replaceState(null, "", "#" + button.dataset.tab);
  });
});

function showResult() {
  if(!model) return;
  const input = {regiao: $("regiao").value, contexto: $("contexto").value};
  if(!model.features.every((feature, i) => model.categories[i].includes(input[feature]))) {
    $("loading").textContent = "Selecione as duas características da ocorrência.";
    return;
  }
  const p = MLP.predict(model, input);
  $("empty-state").hidden = true;
  $("result").hidden = false;
  $("predicted-class").textContent = p.classe;
  $("input-summary").textContent = `${input.regiao} · ${input.contexto}`;
  $("prob-furto").textContent = pct(p.furto);
  $("prob-roubo").textContent = pct(p.roubo);
  $("bar-furto").style.width = p.furto*100+"%";
  $("bar-roubo").style.width = p.roubo*100+"%";
  const maxP = Math.max(p.furto, p.roubo);
  $("interpretation").textContent = `Considerando que uma ocorrência aconteceu com essas características, a rede atribui ${pct(maxP)} à classe “${p.classe.toLowerCase()}”.`;
  const support = metadata.training_support.find(r => r.regiao === input.regiao && r.contexto === input.contexto);
  const n = support ? support.count : 0;
  $("support").textContent = `No treino, ${n.toLocaleString("pt-BR")} boletins têm essa combinação de entradas.`;
  $("uncertainty").textContent = n < 30 ? "Combinação pouco representada no treinamento: interprete o resultado com cautela." : maxP < 0.6 ? "As probabilidades estão próximas. A rede encontrou pouca distinção entre as classes neste contexto." : "O resultado reflete padrões do histórico, com margem de erro.";
  $("loading").textContent = "Classificação concluída no navegador.";
}
document.querySelector(".brand").addEventListener("click", () => $("tab-classificar").click());
$("form").addEventListener("submit", event => {event.preventDefault();showResult();});
document.querySelectorAll("[data-example]").forEach(button => button.addEventListener("click", () => {
  $("regiao").value = button.dataset.example === "transporte" ? "Centro" : "Norte";
  $("contexto").value = button.dataset.example === "transporte" ? "Transporte coletivo" : "Transeunte";
  showResult();
}));

async function init() {
  try {
    const responses = await Promise.all([fetch("model.json"), fetch("metadata.json")]);
    if(responses.some(r => !r.ok)) throw new Error("Não foi possível carregar os arquivos do modelo.");
    [model, metadata] = await Promise.all(responses.map(r => r.json()));
    if(model.format !== "sklearn-mlp-binary-v1") throw new Error("Versão de modelo incompatível.");
    model.features.forEach((feature, i) => {
      const select = $(feature);
      select.replaceChildren(new Option("Selecione…", ""), ...model.categories[i].map(value => new Option(value, value)));
      select.disabled = false;
    });
    $("classify").disabled = false;
    document.querySelectorAll("[data-example]").forEach(button => {button.disabled = false;});
    $("loading").textContent = `${model.selected} carregada · pesos reais do treinamento`;
    $("test-accuracy").textContent = pct(model.metrics.accuracy);
    const comparison = $("comparison");
    metadata.report.configs.forEach(config => {
      const row = document.createElement("tr");
      if(config.name === model.selected) row.className = "selected";
      const cells = [config.name === model.selected ? config.name + " · selecionada" : config.name,
        config.hidden_layers.join(" → "), pct(config.validation.accuracy), pct(config.validation.precision_macro),
        pct(config.validation.recall_macro), pct(config.validation.f1_macro), String(config.best_epoch)];
      cells.forEach(value => {const td = document.createElement("td");td.textContent = value;row.appendChild(td);});
      comparison.appendChild(row);
    });
    $("selection-note").textContent = `${model.selected}: maior F1 macro de validação. As diferenças são pequenas e não demonstram superioridade estatística. Cada checkpoint minimiza a perda de validação com tolerância de 0,0001.`;
    const t = metadata.report.test;
    $("test-metrics").textContent = `Acurácia: ${pct(t.accuracy)} · Precision macro: ${pct(t.precision_macro)} · Recall macro: ${pct(t.recall_macro)} · F1 macro: ${pct(t.f1_macro)}.`;
    $("baseline-metrics").textContent = `Referência que sempre estima a classe majoritária: acurácia de ${pct(metadata.report.baseline.test.accuracy)} e F1 macro de ${pct(metadata.report.baseline.test.f1_macro)} no teste.`;
    $("overfitting-note").textContent = metadata.overfitting_note;
    const initial = location.hash.slice(1);
    const tab = [...document.querySelectorAll("[data-tab]")].find(t => t.dataset.tab === initial);
    if(tab) tab.click();
  } catch(error) {
    $("loading").classList.add("error");
    $("loading").textContent = error.message + " Abra a aplicação por um servidor HTTP e tente recarregar.";
  }
}
init();
