/* Upload pipeline off the main thread: validate CSVs -> pool answers -> embed -> neighbours -> heads. */
importScripts("core.js");

let D = null; // {enc, emb: {fused, bitsq}, theta: properties.json theta block, heads: property heads}
let index = null;

function progress(stage, frac) {
  postMessage({ type: "progress", stage, frac });
}

onmessage = (ev) => {
  const msg = ev.data;
  if (msg.type === "init") {
    D = msg.data;
    index = { fused: BP.normRows(D.emb.fused), bitsq: BP.normRows(D.emb.bitsq) };
    return;
  }
  if (msg.type !== "run") return;
  try {
    const enc = D.enc;
    progress("Checking the bits CSV", 0.02);
    const bits = BP.parseBits(msg.bitsText, enc.core_qids, enc.pool_qids);
    let answers = null;
    if (msg.answersText) {
      answers = BP.parseAnswers(msg.answersText, enc.core_qids, enc.pool_qids, 768, (f) =>
        progress("Reading the answer embeddings", 0.05 + 0.75 * f)
      );
    }
    const errors = [];
    bits.errors.forEach((e) => errors.push(`Bits CSV: ${e}`));
    if (answers) answers.errors.forEach((e) => errors.push(`Answers CSV: ${e}`));
    const warnings = bits.warnings.map((w) => `Bits CSV: ${w}`);
    if (answers) answers.warnings.forEach((w) => warnings.push(`Answers CSV: ${w}`));
    if (errors.length) {
      postMessage({ type: "invalid", errors, warnings });
      return;
    }
    progress("Pooling answers", 0.82);
    let text = null;
    if (answers) text = BP.poolText(answers.vectors, enc.pool_qids).vec;
    const space = text ? "fused" : "bitsq";
    progress(`Placing the model (${space === "fused" ? "BULLPEN-fused" : "BULLPEN-bits+q"})`, 0.88);
    const z = BP.embed(enc.spaces[space], bits.bits, bits.observed, text);
    progress("Searching the 260 paper models", 0.94);
    const neighbours = BP.knn(index[space], z, 10, -1);
    const preds = BP.predictAll(enc.heads[space], z);
    // θ(m) from the bits alone; the property heads read the fused embedding, so they need answers
    const th = BP.thetaFoldIn(D.theta.irt, bits.bits, bits.observed);
    const theta = { z: (th - D.theta.irt.mu) / D.theta.irt.sd, pct: BP.percentileOf(D.theta.sorted_theta, th) };
    const props = space === "fused" ? BP.predictAll(D.heads, z) : null;
    progress("Done", 1);
    postMessage({
      type: "result",
      space,
      z,
      neighbours,
      preds,
      theta,
      props,
      warnings,
      nObserved: bits.nObserved,
      nAnswers: answers ? answers.nPool : 0,
    });
  } catch (err) {
    postMessage({ type: "invalid", errors: [`The page hit an unexpected error: ${err.message}`], warnings: [] });
  }
};
