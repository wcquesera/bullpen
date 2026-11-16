/* BULLPEN demo UI: explorer over the 260 paper models, and the CSV upload. */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else n.setAttribute(k, v);
    }
    for (const c of kids) if (c != null) n.append(c);
    return n;
  };
  const SPACE_NAME = { fused: "BULLPEN-fused", bitsq: "BULLPEN-bits+q" };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  let ENC, MODELS, EMB, EXAMPLES, INDEX;

  const name = (id) => id.replace("__", "/");

  // ------------------------------------------------------------ formatting
  function fmtYear(v) {
    const y = Math.floor(v);
    const m = Math.min(11, Math.max(0, Math.floor((v - y) * 12)));
    return `${MONTHS[m]} ${y}`;
  }
  function sig(v, n = 2) {
    return Number(v.toPrecision(n)).toString();
  }
  function pct(v) {
    const c = Math.min(1, Math.max(0, v));
    return `${(100 * c).toFixed(1)}%` + (c !== v ? "*" : "");
  }
  const FORMAT = {
    log_params: { pred: (v) => `${sig(10 ** v)}B`, truth: (v) => `${sig(10 ** v)}B` },
    release_year: { pred: fmtYear, truth: fmtYear },
    is_chat: { pred: (v) => `P(chat) ${v.toFixed(2)}`, truth: (v) => (v >= 0.5 ? "chat" : "base") },
    log_n_tok: { pred: (v) => v.toFixed(2), truth: (v) => v.toFixed(2) },
    refusal_rate: { pred: pct, truth: pct },
    hedge_rate: { pred: pct, truth: pct },
    oll_v2: { pred: (v) => v.toFixed(1), truth: (v) => v.toFixed(1) },
  };
  const NOTE = {
    log_params: "parameters",
    release_year: "Hugging Face creation date",
    is_chat: "label from the model name",
    log_n_tok: "log answer length in tokens",
    refusal_rate: "share of answers that refuse",
    hedge_rate: "share of answers that hedge",
    oll_v2: "average score",
  };

  // ------------------------------------------------------------ rendering
  function roleBadge(role) {
    if (role === "heldout") return el("span", { class: "badge held", text: "held out" });
    if (role === "train") return el("span", { class: "badge train", text: "training" });
    return el("span", { class: "badge up", text: "your upload" });
  }

  function neighbourTable(neigh) {
    const tb = el("tbody");
    neigh.forEach(({ i, sim }, r) => {
      const m = MODELS[i];
      tb.append(
        el("tr", {},
          el("td", { class: "num", text: String(r + 1) }),
          el("td", {}, el("a", { href: `#m=${encodeURIComponent(m.id)}`, text: name(m.id) })),
          el("td", { text: m.family || "—" }),
          el("td", {}, roleBadge(m.role)),
          el("td", { class: "num", text: sim.toFixed(3) }))
      );
    });
    return el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ...["#", "model", "publisher family", "split", "cosine"].map((h) => el("th", { text: h })))),
      tb);
  }

  function predictionTable(space, preds, truth) {
    const tb = el("tbody");
    let clamped = false;
    for (const [task, spec] of Object.entries(ENC.tasks)) {
      const h = ENC.heads[space][task];
      if (!h) continue;
      const f = FORMAT[task];
      const p = f.pred(preds[task]);
      if (p.endsWith("*")) clamped = true;
      const t = truth && truth[task] != null ? f.truth(truth[task]) : null;
      tb.append(
        el("tr", {},
          el("td", {}, el("div", { text: spec.label }), el("div", { class: "sub", text: NOTE[task] })),
          el("td", { class: "num strong", text: p }),
          el("td", { class: "num", text: t ?? "—" }),
          el("td", { class: "num sub", text: String(h.n_train) }))
      );
    }
    const tbl = el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ...["task", "predicted", truth ? "true" : "true (held-out only)", "head fit on"].map((h) => el("th", { text: h })))),
      tb);
    const wrap = el("div", {}, tbl);
    if (clamped) wrap.append(el("p", { class: "sub", text: "* a ridge prediction outside [0, 1], shown clamped." }));
    wrap.append(el("p", { class: "sub", text: "Answer length, hedging and refusal: the author responses report that each label's own defining formula, computed directly from the answers, beats the embedding (general response, row 4). Size, release date and base vs chat are metadata checks rather than use cases." }));
    return wrap;
  }

  function resultBlock({ title, badge, space, neigh, preds, truth, story, notes }) {
    const card = el("section", { class: "card" });
    const head = el("div", { class: "card-head" }, el("h3", { text: title }), badge);
    card.append(head);
    if (story) {
      card.append(el("div", { class: "story" },
        el("div", { class: "kind", text: story.kind }),
        el("h4", { text: story.title }),
        el("p", { text: story.text })));
    }
    (notes || []).forEach((n) => card.append(el("p", { class: "note", text: n })));
    const grid = el("div", { class: "grid2" },
      el("div", {}, el("h4", { text: `Nearest 10 of the 260 paper models (${SPACE_NAME[space]}, cosine)` }), neighbourTable(neigh)),
      el("div", {}, el("h4", { text: "Predicted properties" }), predictionTable(space, preds, truth)));
    card.append(grid);
    return card;
  }

  // ------------------------------------------------------------ explorer
  function showModel(id) {
    const i = MODELS.findIndex((m) => m.id === id);
    if (i < 0) return;
    const m = MODELS[i];
    $("#pick").value = name(m.id);
    const z = EMB.fused[i];
    const neigh = BP.knn(INDEX.fused, z, 10, i);
    const preds = BP.predictAll(ENC.heads.fused, z);
    const story = EXAMPLES.find((e) => e.id === m.id);
    const notes = [];
    if (m.role === "train") notes.push("A training model in this fold: the encoder and heads were fitted with it, so true values are withheld and its predictions are in-sample.");
    else notes.push("Held out in this fold: neither the encoder nor the heads saw this model. It is placed from its own answers by the fold-in, as an upload is.");
    $("#explorer-out").replaceChildren(resultBlock({
      title: name(m.id), badge: roleBadge(m.role), space: "fused", neigh, preds,
      truth: m.role === "heldout" ? m.truth : null, story, notes,
    }));
    document.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c.dataset.id === m.id));
  }

  function setupExplorer() {
    const dl = $("#model-list");
    MODELS.map((m) => m.id).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }))
      .forEach((id) => dl.append(el("option", { value: name(id) })));
    const chips = $("#chips");
    EXAMPLES.forEach((e) => {
      const c = el("button", { class: "chip", type: "button" }, el("b", { text: e.kind }), ` ${name(e.id)}`);
      c.dataset.id = e.id;
      c.onclick = () => (location.hash = `m=${encodeURIComponent(e.id)}`);
      chips.append(c);
    });
    $("#pick").addEventListener("change", (ev) => {
      const id = ev.target.value.replace("/", "__");
      const hit = MODELS.find((m) => m.id === id || m.id === ev.target.value);
      if (hit) location.hash = `m=${encodeURIComponent(hit.id)}`;
    });
    const route = () => {
      const m = /m=([^&]+)/.exec(location.hash);
      showModel(m ? decodeURIComponent(m[1]) : EXAMPLES[0].id);
    };
    window.addEventListener("hashchange", route);
    route();
  }

  // ------------------------------------------------------------ upload
  function setupUpload() {
    const worker = new Worker("js/worker.js");
    worker.postMessage({ type: "init", data: { enc: ENC, emb: EMB } });
    const bar = $("#bar"), stage = $("#stage"), out = $("#upload-out"), go = $("#run");
    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === "progress") {
        bar.style.width = `${Math.round(100 * msg.frac)}%`;
        stage.textContent = msg.stage;
        return;
      }
      go.disabled = false;
      const warn = (msg.warnings || []).map((w) => el("li", { text: w }));
      if (msg.type === "invalid") {
        stage.textContent = "The upload did not pass validation";
        bar.style.width = "0%";
        out.replaceChildren(el("div", { class: "errors" },
          el("h4", { text: "Fix these and upload again" }),
          el("ul", {}, ...msg.errors.map((e) => el("li", { text: e }))),
          warn.length ? el("ul", { class: "warn" }, ...warn) : null));
        return;
      }
      const notes = [
        `Placed with ${SPACE_NAME[msg.space]} from ${msg.nObserved} pool-question bits` +
          (msg.nAnswers ? ` and ${msg.nAnswers} answer embeddings.` : " (no answers CSV, so the bits-only encoder)."),
        "Predictions are validated in distribution only. The submission scores the heads on held-out models from the 260 open-weight models. The author responses add nine closed Claude and GPT models placed without refitting, for which heads trained only on open models rank hedging and answer length almost perfectly (Spearman .97). Other settings are untested; a model unlike the 260 can be misplaced (see the SmolLM2 example).",
      ];
      const card = resultBlock({ title: "Your model", badge: roleBadge("upload"), space: msg.space, neigh: msg.neighbours, preds: msg.preds, truth: null, notes });
      if (warn.length) card.insertBefore(el("ul", { class: "warn" }, ...warn), card.children[1]);
      out.replaceChildren(card);
    };
    go.onclick = async () => {
      const fb = $("#bits-file").files[0], fa = $("#ans-file").files[0];
      if (!fb) {
        out.replaceChildren(el("div", { class: "errors" }, el("p", { text: "A bits CSV is required; the answers CSV is optional." })));
        return;
      }
      go.disabled = true;
      bar.style.width = "1%";
      stage.textContent = "Reading files";
      out.replaceChildren();
      const [bitsText, answersText] = await Promise.all([fb.text(), fa ? fa.text() : Promise.resolve(null)]);
      worker.postMessage({ type: "run", bitsText, answersText });
    };
  }

  // ------------------------------------------------------------ boot
  async function boot() {
    const get = (f) => fetch(`data/${f}`).then((r) => {
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r.json();
    });
    try {
      const [enc, models, fused, bitsq, ex] = await Promise.all(
        ["encoders.json", "models.json", "emb_fused.json", "emb_bitsq.json", "examples.json"].map(get));
      ENC = enc; MODELS = models.models; EXAMPLES = ex.examples;
      EMB = { fused, bitsq };
      INDEX = { fused: BP.normRows(fused) };
      $("#n-train").textContent = enc.protocol.n_train;
      $("#n-held").textContent = enc.protocol.n_heldout;
      setupExplorer();
      setupUpload();
      document.body.classList.add("ready");
    } catch (err) {
      $("#explorer-out").replaceChildren(el("div", { class: "errors", text: `Could not load the site data: ${err.message}` }));
    }
  }
  boot();
})();
