/* BULLPEN demo: the encoder fold-in, prediction heads, neighbour search and CSV checks.
 * Pure functions, no DOM. Mirrors site/export.py:embed_linear and Head.predict exactly.
 * Loaded by the page as a global (BP) and by node for the parity test (module.exports). */
(function (root) {
  "use strict";

  // Solve (A) x = b for a small symmetric positive-definite A (Gaussian elimination, partial pivoting).
  function solve(A, b) {
    const n = b.length;
    const M = A.map((row, i) => row.concat([b[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        if (f === 0) continue;
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  // ridge: argmin ||W_j s - y||^2 + lam ||s||^2 over the observed rows j of W
  function ridge(W, j, y, lam) {
    const d = W[0].length;
    const A = Array.from({ length: d }, () => new Array(d).fill(0));
    const r = new Array(d).fill(0);
    for (let t = 0; t < j.length; t++) {
      const w = W[j[t]];
      for (let a = 0; a < d; a++) {
        r[a] += w[a] * y[t];
        for (let c = a; c < d; c++) A[a][c] += w[a] * w[c];
      }
    }
    for (let a = 0; a < d; a++) {
      A[a][a] += lam;
      for (let c = 0; c < a; c++) A[a][c] = A[c][a];
    }
    return solve(A, r);
  }

  function zscore(s, b) {
    return s.map((v, i) => (v - b.zmu[i]) / b.zsd[i]);
  }

  /* Place one model in an exported PLS-SVD space.
   * bits: array over the pool questions (spec order) or null; observed: same-length booleans
   * (null = all observed); text: pooled 768-d answer vector or null. */
  function embed(spec, bits, observed, text) {
    const out = [];
    let j = [], yc = [];
    if (bits) {
      for (let q = 0; q < bits.length; q++) {
        if (observed && !observed[q]) continue;
        j.push(q);
        yc.push(bits[q] - spec.mu_b[q]);
      }
    }
    for (const b of spec.blocks) {
      let s;
      if (b.channel === "bits") {
        if (!j.length) { out.push(...b.zmean); continue; }
        s = ridge(b.W, j, yc, spec.lam);
      } else if (b.channel === "qtext") {
        if (!j.length) { out.push(...b.zmean); continue; }
        const d = b.c.length;
        s = new Array(d).fill(0);
        for (let t = 0; t < j.length; t++) {
          const g = b.G[j[t]];
          for (let a = 0; a < d; a++) s[a] += yc[t] * g[a];
        }
        for (let a = 0; a < d; a++) s[a] = s[a] / j.length - b.c[a];
      } else {
        if (!text) { out.push(...b.zmean); continue; }
        const d = b.W[0].length;
        s = new Array(d).fill(0);
        for (let i = 0; i < text.length; i++) {
          const x = text[i] - b.mean[i];
          const w = b.W[i];
          for (let a = 0; a < d; a++) s[a] += x * w[a];
        }
      }
      out.push(...zscore(s, b));
    }
    return out;
  }

  function predict(head, z) {
    let lin = head.intercept;
    for (let i = 0; i < z.length; i++) lin += ((z[i] - head.mu[i]) / head.sd[i]) * head.coef[i];
    return head.kind === "ridge" ? lin : 1 / (1 + Math.exp(-lin));
  }

  function predictAll(heads, z) {
    const out = {};
    for (const t of Object.keys(heads)) out[t] = predict(heads[t], z);
    return out;
  }

  // Flatten [[...],...] rows into a Float32Array with unit-norm rows for cosine search.
  function normRows(rows) {
    const n = rows.length, d = rows[0].length, F = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += rows[i][k] * rows[i][k];
      s = Math.sqrt(s) || 1;
      for (let k = 0; k < d; k++) F[i * d + k] = rows[i][k] / s;
    }
    return { F, n, d };
  }

  function knn(index, z, k, exclude) {
    const { F, n, d } = index;
    let s = 0;
    for (let a = 0; a < d; a++) s += z[a] * z[a];
    s = Math.sqrt(s) || 1;
    const best = []; // [sim, i], kept sorted descending, length <= k
    for (let i = 0; i < n; i++) {
      if (i === exclude) continue;
      let c = 0;
      const o = i * d;
      for (let a = 0; a < d; a++) c += F[o + a] * z[a];
      c /= s;
      if (best.length < k || c > best[best.length - 1][0]) {
        let p = best.length < k ? best.length : k - 1;
        best[p] = [c, i];
        while (p > 0 && best[p][0] > best[p - 1][0]) {
          [best[p], best[p - 1]] = [best[p - 1], best[p]];
          p--;
        }
      }
    }
    return best.map(([sim, i]) => ({ i, sim }));
  }

  // Mean of the given answer vectors over the pool questions, renormalised to unit length
  // (bullpen.data.text_banks.pooled_model_text).
  function poolText(vectors, poolQids) {
    let sum = null, n = 0;
    for (const q of poolQids) {
      const v = vectors.get(q);
      if (!v) continue;
      if (!sum) sum = new Float64Array(v.length);
      for (let i = 0; i < v.length; i++) sum[i] += v[i];
      n++;
    }
    if (!n) return { vec: null, n: 0 };
    let s = 0;
    for (let i = 0; i < sum.length; i++) { sum[i] /= n; s += sum[i] * sum[i]; }
    s = Math.max(Math.sqrt(s), 1e-12);
    return { vec: Array.from(sum, (v) => v / s), n };
  }

  // PCA view: (z - mean) @ components^T, first k components (site/export.py:pca_projection)
  function project(proj, z, k) {
    const n = k || proj.components.length;
    const out = new Array(n).fill(0);
    for (let c = 0; c < n; c++) {
      const w = proj.components[c];
      let s = 0;
      for (let i = 0; i < z.length; i++) s += (z[i] - proj.mean[i]) * w[i];
      out[c] = s;
    }
    return out;
  }

  // ---------------------------------------------------------------- CSV checks
  const MIN_POOL = 50; // fewest pool questions an upload must cover
  const MAX_ERRORS = 8;

  function splitLines(text) {
    return text.replace(/^﻿/, "").split(/\r?\n/);
  }

  function header(line) {
    return line.split(",").map((h) => h.trim().toLowerCase());
  }

  function finish(errors, warnings, extra) {
    const shown = errors.slice(0, MAX_ERRORS);
    if (errors.length > MAX_ERRORS) shown.push(`... and ${errors.length - MAX_ERRORS} more errors`);
    return Object.assign({ ok: errors.length === 0, errors: shown, warnings }, extra);
  }

  /* bits CSV: header `question_id,correct`; correct in {0,1}, blank = not answered. */
  function parseBits(text, coreQids, poolQids) {
    const errors = [], warnings = [];
    const lines = splitLines(text);
    const h = header(lines[0] || "");
    if (h[0] !== "question_id" || h[1] !== "correct" || h.length !== 2) {
      return finish([`line 1: header must be exactly "question_id,correct" (got "${(lines[0] || "").slice(0, 60)}")`], [], {});
    }
    const core = new Set(coreQids), seen = new Map();
    for (let n = 1; n < lines.length; n++) {
      const line = lines[n].trim();
      if (!line) continue;
      const f = line.split(",");
      if (f.length !== 2) { errors.push(`line ${n + 1}: expected 2 fields, got ${f.length}`); continue; }
      const q = f[0].trim(), v = f[1].trim();
      if (!core.has(q)) { errors.push(`line ${n + 1}: unknown question_id "${q}" (not one of the 1,181 core questions)`); continue; }
      if (seen.has(q)) { errors.push(`line ${n + 1}: question_id "${q}" repeated (first on line ${seen.get(q).line})`); continue; }
      if (v !== "" && v !== "0" && v !== "1") { errors.push(`line ${n + 1}: correct must be 0, 1 or blank, got "${v}"`); continue; }
      seen.set(q, { line: n + 1, v });
    }
    const bits = new Array(poolQids.length).fill(0), observed = new Array(poolQids.length).fill(false);
    let nObs = 0;
    poolQids.forEach((q, i) => {
      const r = seen.get(q);
      if (r && r.v !== "") { bits[i] = Number(r.v); observed[i] = true; nObs++; }
    });
    if (!errors.length && nObs < MIN_POOL) errors.push(`only ${nObs} of the ${poolQids.length} pool questions have a 0/1 answer; at least ${MIN_POOL} are needed`);
    if (!errors.length && nObs < poolQids.length) warnings.push(`${poolQids.length - nObs} of ${poolQids.length} pool questions are blank or absent; the bits are placed from the ${nObs} observed ones (ridge fold-in)`);
    const nonPool = [...seen.keys()].filter((q) => !poolQids.includes(q)).length;
    if (nonPool) warnings.push(`${nonPool} rows are core questions outside the 900-question pool the encoder reads; they are ignored`);
    return finish(errors, warnings, { bits, observed, nObserved: nObs });
  }

  /* answers CSV: header `question_id,e0,...,e767`; one BGE-base answer embedding per row. */
  function parseAnswers(text, coreQids, poolQids, dim, onProgress) {
    const errors = [], warnings = [];
    const lines = splitLines(text);
    const h = header(lines[0] || "");
    const want = ["question_id"].concat(Array.from({ length: dim }, (_, i) => `e${i}`));
    if (h.length !== want.length || h.some((x, i) => x !== want[i])) {
      return finish([`line 1: header must be "question_id,e0,...,e${dim - 1}" (${dim + 1} columns; got ${h.length})`], [], {});
    }
    const core = new Set(coreQids), vectors = new Map();
    let offNorm = 0;
    for (let n = 1; n < lines.length; n++) {
      if (onProgress && n % 100 === 0) onProgress(n / lines.length);
      const line = lines[n].trim();
      if (!line) continue;
      const f = line.split(",");
      const q = f[0].trim();
      if (f.length !== dim + 1) { errors.push(`line ${n + 1}: expected ${dim + 1} fields, got ${f.length}`); continue; }
      if (!core.has(q)) { errors.push(`line ${n + 1}: unknown question_id "${q}"`); continue; }
      if (vectors.has(q)) { errors.push(`line ${n + 1}: question_id "${q}" repeated`); continue; }
      const v = new Float64Array(dim);
      let bad = -1, s = 0;
      for (let i = 0; i < dim; i++) {
        const x = Number(f[i + 1]);
        if (f[i + 1].trim() === "" || !Number.isFinite(x)) { bad = i; break; }
        v[i] = x; s += x * x;
      }
      if (bad >= 0) { errors.push(`line ${n + 1}: e${bad} is not a number ("${f[bad + 1]}")`); continue; }
      if (Math.abs(Math.sqrt(s) - 1) > 0.05) offNorm++;
      vectors.set(q, v);
    }
    const nPool = poolQids.filter((q) => vectors.has(q)).length;
    if (!errors.length && nPool < MIN_POOL) errors.push(`only ${nPool} of the ${poolQids.length} pool questions have an answer embedding; at least ${MIN_POOL} are needed`);
    if (offNorm) warnings.push(`${offNorm} rows are not unit length; BGE answer embeddings are L2-normalised, so check the pooling (mean pooling, then L2 norm)`);
    if (!errors.length && nPool < poolQids.length) warnings.push(`${poolQids.length - nPool} pool questions have no answer embedding; the mean is taken over the ${nPool} present`);
    return finish(errors, warnings, { vectors, nPool });
  }

  const api = { project, solve, ridge, embed, predict, predictAll, normRows, knn, poolText, parseBits, parseAnswers, MIN_POOL };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BP = api;
})(typeof self !== "undefined" ? self : this);
