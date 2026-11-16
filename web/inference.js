/* Inferência da MLP exportada do scikit-learn. Sem dependências externas. */
(function(root) {
  "use strict";
  function encode(model, input) {
    return model.features.flatMap((name, k) =>
      model.categories[k].map(value => value === input[name] ? 1 : 0));
  }
  function sigmoid(value) {
    if (value >= 0) return 1 / (1 + Math.exp(-value));
    const e = Math.exp(value);
    return e / (1 + e);
  }
  function predict(model, input) {
    if(model.format !== "sklearn-mlp-binary-v1") throw new Error("Formato de modelo incompatível.");
    let a = encode(model, input);
    for(let layer = 0; layer < model.weights.length; layer++) {
      const w = model.weights[layer], b = model.biases[layer];
      const last = layer === model.weights.length - 1;
      a = b.map((bias, j) => {
        let z = bias;
        for(let i=0; i<a.length; i++) z += a[i] * w[i][j];
        return last ? sigmoid(z) : Math.max(0, z);
      });
    }
    const roubo = a[0];
    return {furto: 1-roubo, roubo, classe: roubo >= model.threshold ? "Roubo" : "Furto"};
  }
  const api = {encode, predict};
  if(typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MLP = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
