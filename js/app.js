/* BULLPEN demo, 3-pane UI: input (left) -> embedding space (middle) -> inspector (right). */
(function () {
  "use strict";
  const $ = (s) => document.querySelector(s);
  const $$ = (s) => [...document.querySelectorAll(s)];
  const el = (tag, attrs = {}, ...kids) => {
    const n = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") n.className = v;
      else if (k === "text") n.textContent = v;
      else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
      else n.setAttribute(k, v);
    }
    for (const c of kids) if (c != null) n.append(c);
    return n;
  };
  const SPACE_NAME = { fused: "BULLPEN-fused", bitsq: "BULLPEN-bits+q" };
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  // Swiss theme (see css/style.css): black on white, one red accent
  const THEME = {
    ink: "#111", paper: "#fff", train: "#b3b3b3", held: "#111", accent: "#e30613", rule: "#e2e2e2",
    muted: "#767676", font: '"Helvetica Neue", Helvetica, Inter, Arial, sans-serif',
  };
  const GROUPS = {
    behavior: {
      title: "Behavior",
      tasks: ["log_n_tok", "refusal_rate", "hedge_rate"],
      notes: [
        "Author responses, general response row 4: for answer length, hedging and refusal, each label's own defining formula, computed from the answers, beats the embedding.",
      ],
    },
    capability: {
      title: "Capability",
      tasks: ["oll_v2"],
      notes: ["The Open LLM Leaderboard v2 head is fitted on the training models that have a board score. The count is under 'head fit on'."],
    },
    metadata: {
      title: "Metadata",
      tasks: ["log_params", "release_year", "is_chat"],
      notes: [
        "Author responses: size, release date and base vs chat are \"metadata checks rather than use cases\".",
        "The base-vs-chat label comes from the model name.",
      ],
    },
  };

  const S = { space: "fused", mode: "2d", path: "examples", sel: null, tab: "overview", upload: null };
  let ENC, MODELS, EMB, PROJ, EXAMPLES, INDEX, plot;

  const name = (id) => id.replace("__", "/");

  // ------------------------------------------------------------ formatting
  function fmtYear(v) {
    const y = Math.floor(v);
    const m = Math.min(11, Math.max(0, Math.floor((v - y) * 12)));
    return `${MONTHS[m]} ${y}`;
  }
  const sig = (v, n = 2) => Number(v.toPrecision(n)).toString();
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

  // ------------------------------------------------------------ model state
  const isUpload = (sel) => sel === "upload";
  function zOf(sel) {
    if (sel == null) return null;
    if (isUpload(sel)) return S.upload && S.upload.space === S.space ? S.upload.z : null;
    return EMB[S.space][sel];
  }
  function neighboursOf(sel) {
    const z = zOf(sel);
    return z ? BP.knn(INDEX[S.space], z, 10, isUpload(sel) ? -1 : sel) : [];
  }
  function predsOf(sel) {
    const z = zOf(sel);
    return z ? BP.predictAll(ENC.heads[S.space], z) : null;
  }

  // ------------------------------------------------------------ middle pane
  function renderPlot() {
    const P = PROJ[S.space];
    plot.setData(MODELS.map((m, i) => ({ p: P.coords[i], kind: m.role })));
    const up = S.upload && S.upload.space === S.space ? BP.project(P, S.upload.z, 3) : null;
    plot.setUpload(up);
    plot.setSelection(zOf(S.sel) ? S.sel : null, neighboursOf(S.sel).map((n) => n.i));
    renderLegend();
    const k = S.mode === "3d" ? 3 : 2;
    const evr = P.evr.slice(0, k).reduce((a, b) => a + b, 0);
    const clipped = P.coords.filter((c) => plot.isClipped(c, k)).length;
    const lines = [
      `PCA of the 260 ${SPACE_NAME[S.space]} placements. ${k} components keep ${(100 * evr).toFixed(0)}% of the variance.`,
      `Each axis is scaled to its 2nd to 98th percentile range. ${clipped} models fall farther out and are pinned to the frame.`,
      "Distances in the view are approximate. Red links mark the nearest 10 by cosine in the full 32-d space.",
      (S.mode === "3d" ? "Drag to rotate" : "Drag to pan") + ". Scroll to zoom. Click a point to inspect it.",
    ];
    $("#caption").replaceChildren(...lines.map((t) => el("li", { text: t })));
  }

  function renderLegend() {
    const item = (color, label, shape = "dot") => el("span", { class: "lg" }, el("i", { class: shape, style: `background:${color}` }), label);
    const items = [item(THEME.train, "training"), item(THEME.held, "held out"), item(THEME.accent, "selected"), item(THEME.accent, "nearest 10", "ring")];
    if (S.upload) items.push(item(THEME.accent, S.upload.space === S.space ? "your model" : `your model (in ${SPACE_NAME[S.upload.space]})`, "square"));
    $("#legend").replaceChildren(...items);
  }

  // ------------------------------------------------------------ right pane
  function roleBadge(role) {
    if (role === "heldout") return el("span", { class: "tag held", text: "held out" });
    if (role === "train") return el("span", { class: "tag", text: "training" });
    return el("span", { class: "tag up", text: "your upload" });
  }

  function predictionTable(tasks, preds, truth) {
    const tb = el("tbody");
    let clamped = false;
    for (const task of tasks) {
      const h = ENC.heads[S.space][task];
      if (!h) continue;
      const f = FORMAT[task];
      const p = f.pred(preds[task]);
      if (p.endsWith("*")) clamped = true;
      const t = truth && truth[task] != null ? f.truth(truth[task]) : null;
      tb.append(el("tr", {},
        el("td", {}, el("div", { text: ENC.tasks[task].label }), el("div", { class: "sub", text: NOTE[task] })),
        el("td", { class: "num strong", text: p }),
        el("td", { class: "num", text: t ?? "—" }),
        el("td", { class: "num sub", text: String(h.n_train) })));
    }
    const wrap = el("div", {}, el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ...["task", "predicted", truth ? "true" : "true (held-out only)", "head fit on"].map((x) => el("th", { text: x })))),
      tb));
    if (clamped) wrap.append(el("p", { class: "sub", text: "* A ridge prediction outside [0, 1], shown clamped." }));
    return wrap;
  }

  function neighbourTable(neigh) {
    const tb = el("tbody");
    neigh.forEach(({ i, sim }, r) => {
      const m = MODELS[i];
      tb.append(el("tr", { class: "clickable", onclick: () => select(i) },
        el("td", { class: "num", text: String(r + 1) }),
        el("td", {}, el("a", { href: `#m=${encodeURIComponent(m.id)}`, text: name(m.id) })),
        el("td", { text: m.family || "—" }),
        el("td", {}, roleBadge(m.role)),
        el("td", { class: "num", text: sim.toFixed(3) })));
    });
    return el("table", { class: "tbl" },
      el("thead", {}, el("tr", {}, ...["#", "model", "publisher family", "split", "cosine"].map((x) => el("th", { text: x })))),
      tb);
  }

  const ul = (items, cls = "plain") => el("ul", { class: cls }, ...items.map((t) => el("li", { text: t })));

  function uploadNotes() {
    const u = S.upload;
    return [
      `Placed with ${SPACE_NAME[u.space]} from ${u.nObserved} pool-question bits` +
        (u.nAnswers ? ` and ${u.nAnswers} answer embeddings.` : ". No answers CSV, so the bits-only encoder."),
      "Predictions are validated in distribution only.",
      "The submission scores the heads on held-out models from the 260 open-weight models.",
      "The author responses add nine closed Claude and GPT models placed without refitting. Heads trained only on open models rank their hedging and answer length almost perfectly (Spearman .97).",
      "Other settings are untested. A model unlike the 260 can be misplaced; see the SmolLM2 example.",
    ];
  }

  function renderInspector() {
    const head = $("#insp-head"), body = $("#insp-body");
    $$("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === S.tab));
    if (S.sel == null) {
      head.replaceChildren();
      body.replaceChildren(el("p", { class: "sub", text: "Click a point or pick an example." }));
      return;
    }
    const up = isUpload(S.sel);
    const m = up ? null : MODELS[S.sel];
    head.replaceChildren(el("h3", { text: up ? "Your model" : name(m.id) }), roleBadge(up ? "upload" : m.role),
      el("span", { class: "sub", text: SPACE_NAME[S.space] }));
    const z = zOf(S.sel);
    if (!z) {
      body.replaceChildren(el("div", { class: "note" }, ul([`Your model was placed with ${SPACE_NAME[S.upload.space]}.`, "Switch the space in the middle pane to see it."])));
      return;
    }
    const preds = predsOf(S.sel);
    const truth = !up && m.role === "heldout" ? m.truth : null;
    const out = [];
    if (S.tab === "overview") {
      const story = up ? null : EXAMPLES.find((e) => e.id === m.id);
      if (story) out.push(el("div", { class: "story" }, el("div", { class: "kind", text: story.kind }), el("h4", { text: story.title }), ul(story.points)));
      if (up) {
        out.push(el("div", { class: "note" }, ul(uploadNotes())));
        if (S.upload.warnings.length) out.push(ul(S.upload.warnings, "plain warn"));
      } else if (m.role === "train") out.push(el("div", { class: "note" }, ul(["Training model in this fold. The encoder and heads were fitted with it.", "True values are withheld. Predictions are in-sample."])));
      else out.push(el("div", { class: "note" }, ul(["Held out in this fold. The encoder and heads never saw it.", "Placed from its own answers by the fold-in, the same way an upload is."])));
      out.push(el("h4", { text: "Predicted properties" }), predictionTable(Object.keys(ENC.tasks), preds, truth));
      if (story && S.space !== "fused") out.push(el("p", { class: "sub", text: "The example text describes the BULLPEN-fused view." }));
    } else if (S.tab === "neighbours") {
      out.push(el("h4", { text: `Nearest 10 of the 260 paper models (${SPACE_NAME[S.space]}, cosine)` }), neighbourTable(neighboursOf(S.sel)));
    } else {
      const g = GROUPS[S.tab];
      out.push(el("h4", { text: g.title }), predictionTable(g.tasks, preds, truth), el("div", { class: "note" }, ul(g.notes)));
    }
    body.replaceChildren(...out);
  }

  // ------------------------------------------------------------ selection
  function select(sel, { fromHash = false } = {}) {
    S.sel = sel;
    if (!fromHash && !isUpload(sel) && sel != null) history.replaceState(null, "", `#m=${encodeURIComponent(MODELS[sel].id)}`);
    if (isUpload(sel)) history.replaceState(null, "", "#upload");
    $("#pick").value = sel == null || isUpload(sel) ? "" : name(MODELS[sel].id);
    $$(".example").forEach((c) => c.classList.toggle("on", !isUpload(sel) && sel != null && c.dataset.id === MODELS[sel].id));
    renderPlot();
    renderInspector();
  }
  const byId = (id) => MODELS.findIndex((m) => m.id === id);

  // ------------------------------------------------------------ left pane: upload
  const STEP_OF = (stage) =>
    /bits CSV|answer embeddings/.test(stage) ? "validate" : /Pooling/.test(stage) ? "pool" : /Placing/.test(stage) ? "foldin" : /Searching|Done/.test(stage) ? "search" : null;

  function setSteps(active, { error = false, skipPool = false, done = false } = {}) {
    const order = ["validate", "pool", "foldin", "search"];
    const k = order.indexOf(active);
    $$("#pipeline li").forEach((li) => {
      const j = order.indexOf(li.dataset.step);
      li.className = done || j < k ? "done" : j === k ? (error ? "error" : "active") : "";
      if (li.dataset.step === "pool" && skipPool && (done || j < k)) li.className = "skipped";
    });
  }

  function setupUpload() {
    const worker = new Worker("js/worker.js");
    worker.postMessage({ type: "init", data: { enc: ENC, emb: EMB } });
    const bar = $("#bar"), stage = $("#stage"), msgBox = $("#upload-msg"), go = $("#run");
    let hasAnswers = false;
    worker.onmessage = (ev) => {
      const msg = ev.data;
      if (msg.type === "progress") {
        bar.style.width = `${Math.round(100 * msg.frac)}%`;
        stage.textContent = msg.stage;
        const step = STEP_OF(msg.stage);
        if (step) setSteps(step, { skipPool: !hasAnswers, done: msg.stage === "Done" });
        return;
      }
      go.disabled = false;
      if (msg.type === "invalid") {
        stage.textContent = "The upload did not pass validation";
        bar.style.width = "0%";
        setSteps("validate", { error: true });
        msgBox.replaceChildren(el("div", { class: "errors" },
          el("h4", { text: "Fix these and upload again" }),
          el("ul", {}, ...msg.errors.map((e) => el("li", { text: e }))),
          msg.warnings.length ? el("ul", { class: "warn" }, ...msg.warnings.map((w) => el("li", { text: w }))) : null));
        return;
      }
      setSteps("search", { done: true, skipPool: !hasAnswers });
      S.upload = msg;
      msgBox.replaceChildren(el("div", { class: "ok" },
        el("b", { text: `Embedded with ${SPACE_NAME[msg.space]}.` }),
        ul(["Shown as a red square in the middle pane.", `Nearest: ${name(MODELS[msg.neighbours[0].i].id)} (cosine ${msg.neighbours[0].sim.toFixed(3)}).`])));
      setSpace(msg.space, false);
      S.tab = "overview";
      select("upload");
    };
    go.onclick = async () => {
      const fb = $("#bits-file").files[0], fa = $("#ans-file").files[0];
      if (!fb) {
        msgBox.replaceChildren(el("div", { class: "errors" }, el("p", { text: "A bits CSV is required. The answers CSV is optional." })));
        return;
      }
      hasAnswers = !!fa;
      $("#foldin-space").textContent = fa ? "BULLPEN-fused" : "BULLPEN-bits+q";
      go.disabled = true;
      bar.style.width = "1%";
      stage.textContent = "Reading files";
      setSteps("validate");
      msgBox.replaceChildren();
      const [bitsText, answersText] = await Promise.all([fb.text(), fa ? fa.text() : Promise.resolve(null)]);
      worker.postMessage({ type: "run", bitsText, answersText });
    };
  }

  // ------------------------------------------------------------ wiring
  function setSpace(space, redraw = true) {
    S.space = space;
    $$("#seg-space button").forEach((b) => b.classList.toggle("on", b.dataset.space === space));
    if (redraw) { renderPlot(); renderInspector(); }
  }

  function setPath(path) {
    S.path = path;
    $$("#seg-path button").forEach((b) => {
      const on = b.dataset.path === path;
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", String(on));
    });
    $("#path-examples").hidden = path !== "examples";
    $("#path-upload").hidden = path !== "upload";
  }

  function setupControls() {
    $$("#seg-space button").forEach((b) => (b.onclick = () => setSpace(b.dataset.space)));
    $$("#seg-mode button").forEach((b) => (b.onclick = () => {
      S.mode = b.dataset.mode;
      $$("#seg-mode button").forEach((x) => x.classList.toggle("on", x === b));
      plot.setMode(S.mode);
      renderPlot();
    }));
    $$("#seg-path button").forEach((b) => (b.onclick = () => setPath(b.dataset.path)));
    $("#reset-view").onclick = () => plot.resetView();
    $$("#tabs button").forEach((b) => (b.onclick = () => { S.tab = b.dataset.tab; renderInspector(); }));
    $$("[data-open]").forEach((b) => (b.onclick = () => document.getElementById(b.dataset.open).showModal()));
    $$("dialog [data-close]").forEach((b) => (b.onclick = () => b.closest("dialog").close()));
    $$("dialog").forEach((d) => d.addEventListener("click", (e) => { if (e.target === d) d.close(); }));

    const dl = $("#model-list");
    MODELS.map((m) => m.id).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: "base" }))
      .forEach((id) => dl.append(el("option", { value: name(id) })));
    $("#pick").addEventListener("change", (ev) => {
      const v = ev.target.value;
      const i = MODELS.findIndex((m) => m.id === v.replace("/", "__") || m.id === v);
      if (i >= 0) select(i);
    });
    const chips = $("#chips");
    EXAMPLES.forEach((e) => {
      const c = el("button", { class: "example", type: "button", onclick: () => { setSpace("fused", false); S.tab = "overview"; select(byId(e.id)); } },
        el("b", { text: e.kind }), el("span", { text: name(e.id) }));
      c.dataset.id = e.id;
      chips.append(c);
    });

    const tip = $("#tooltip");
    plot.onHover = (hit, x, y) => {
      if (hit == null) { tip.hidden = true; return; }
      const sims = new Map(neighboursOf(S.sel).map((n) => [n.i, n.sim]));
      if (isUpload(hit)) tip.replaceChildren(el("b", { text: "Your model" }));
      else {
        const m = MODELS[hit];
        tip.replaceChildren(el("b", { text: name(m.id) }), el("div", { text: `${m.family || "—"} · ${m.role === "heldout" ? "held out" : "training"}` }),
          sims.has(hit) ? el("div", { text: `cosine to selection ${sims.get(hit).toFixed(3)}` }) : null);
      }
      tip.style.left = `${x + 14}px`; tip.style.top = `${y + 10}px`;
      tip.hidden = false;
    };
    plot.onClick = (hit) => select(hit);

    window.addEventListener("hashchange", () => {
      const mm = /m=([^&]+)/.exec(location.hash);
      if (mm) { const i = byId(decodeURIComponent(mm[1])); if (i >= 0 && i !== S.sel) select(i, { fromHash: true }); }
    });
  }

  async function boot() {
    const get = (f) => fetch(`data/${f}`).then((r) => {
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r.json();
    });
    try {
      const [enc, models, fused, bitsq, proj, ex] = await Promise.all(
        ["encoders.json", "models.json", "emb_fused.json", "emb_bitsq.json", "projections.json", "examples.json"].map(get));
      ENC = enc; MODELS = models.models; EXAMPLES = ex.examples; PROJ = proj;
      EMB = { fused, bitsq };
      INDEX = { fused: BP.normRows(fused), bitsq: BP.normRows(bitsq) };
      $("#n-train").textContent = enc.protocol.n_train;
      $("#n-held").textContent = enc.protocol.n_heldout;
      plot = new ScatterPlot($("#plot"), THEME);
      window.BPPlot = plot;  // used by the screenshot script to find points
      setupControls();
      setupUpload();
      if (location.hash === "#upload") setPath("upload");
      const mm = /m=([^&]+)/.exec(location.hash);
      const start = mm ? byId(decodeURIComponent(mm[1])) : byId(EXAMPLES[0].id);
      select(start >= 0 ? start : byId(EXAMPLES[0].id), { fromHash: true });
      document.body.classList.add("ready");
    } catch (err) {
      $("#insp-body").replaceChildren(el("div", { class: "errors", text: `Could not load the site data: ${err.message}` }));
      throw err;
    }
  }
  boot();
})();
