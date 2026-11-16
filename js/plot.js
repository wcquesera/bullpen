/* Canvas scatter for the embedding view: 2D (pan, zoom) or 3D (drag to rotate, zoom), hover and
 * click picking, a selected point linked to its neighbours, and one extra "upload" point.
 *
 * Display scaling: each axis is mapped to [-1, 1] by its 2nd and 98th percentiles over the 260
 * models, and values outside are clipped to the edge. The view trades exact geometry for
 * density; neighbours are computed elsewhere, in the full embedding. No dependencies. */
(function (root) {
  "use strict";

  const LO_Q = 0.02, HI_Q = 0.98, CLIP = 1.06;

  function quantile(sorted, q) {
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  class ScatterPlot {
    constructor(canvas, theme) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.t = theme;
      this.onHover = () => {};
      this.onClick = () => {};
      this.pts = [];
      this.upload = null;
      this.sel = null;
      this.neigh = [];
      this.hover = null;
      this.mode = "2d";
      this.resetView(false);
      this.screen = [];
      this.uploadScreen = null;
      this._bind();
      new ResizeObserver(() => this.draw()).observe(canvas);
    }

    // points: [{p: [x, y, z], kind: "train" | "heldout"}]
    setData(points) {
      this.pts = points;
      this.scale = [0, 1, 2].map((k) => {
        const v = points.map((q) => q.p[k]).sort((a, b) => a - b);
        const lo = quantile(v, LO_Q), hi = quantile(v, HI_Q);
        return { mid: (lo + hi) / 2, half: (hi - lo) / 2 || 1 };
      });
      this.norm = points.map((q) => this._norm(q.p));
      this.draw();
    }
    _norm(p) {
      return this.scale.map((s, k) => Math.max(-CLIP, Math.min(CLIP, (p[k] - s.mid) / s.half)));
    }
    isClipped(p, dims = 3) {
      return this.scale.slice(0, dims).some((s, k) => Math.abs((p[k] - s.mid) / s.half) > CLIP);
    }
    setMode(mode) { this.mode = mode; this.resetView(); }
    setSelection(sel, neigh) { this.sel = sel; this.neigh = neigh || []; this.draw(); }
    setUpload(p) { this.upload = p ? this._norm(p) : null; this.draw(); }
    resetView(redraw = true) {
      this.yaw = 0.55; this.pitch = 0.3; this.zoom = 1; this.pan = [0, 0];
      if (redraw) this.draw();
    }

    // normalised coords -> [screen x, screen y, depth in -1..1]
    _project(n) {
      const w = this.c.clientWidth, h = this.c.clientHeight, pad = 26;
      let [x, y, z] = n;
      if (this.mode === "2d") {
        const sx = (w / 2 - pad) * this.zoom / CLIP, sy = (h / 2 - pad) * this.zoom / CLIP;
        return [w / 2 + this.pan[0] + sx * x, h / 2 + this.pan[1] - sy * y, 0];
      }
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
      [x, z] = [cy * x + sy * z, -sy * x + cy * z];
      const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      [y, z] = [cp * y - sp * z, sp * y + cp * z];
      // fill the pane: scale to the short side, stretch the long side up to 1.35x
      const s = ((Math.min(w, h) / 2 - pad) * this.zoom) / (CLIP * 1.22);
      const ax = Math.min(1.35, Math.max(1, w / h)), ay = Math.min(1.35, Math.max(1, h / w));
      const persp = 1 / (1 - 0.12 * z / CLIP);
      return [w / 2 + this.pan[0] + ax * s * x * persp, h / 2 + this.pan[1] - ay * s * y * persp, z / (CLIP * 1.73)];
    }

    screenOf(i) {
      const v = i === "upload" ? this.uploadScreen : this.screen[i];
      if (!v) return null;
      const r = this.c.getBoundingClientRect();
      return { x: r.left + v[0], y: r.top + v[1] };
    }

    draw() {
      const c = this.c, ctx = this.ctx, t = this.t, dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (!w || !h || !this.pts.length) return;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      this.screen = this.norm.map((n) => this._project(n));
      this.uploadScreen = this.upload ? this._project(this.upload) : null;
      this.mode === "3d" ? this._cube() : this._frame();

      const selXY = this.sel === "upload" ? this.uploadScreen : this.sel != null ? this.screen[this.sel] : null;
      if (selXY) {
        ctx.strokeStyle = t.accent; ctx.lineWidth = 1; ctx.globalAlpha = 0.8;
        for (const j of this.neigh) {
          const q = this.screen[j];
          ctx.beginPath(); ctx.moveTo(selXY[0], selXY[1]); ctx.lineTo(q[0], q[1]); ctx.stroke();
        }
        ctx.globalAlpha = 1;
      }
      const order = this.pts.map((_, i) => i);
      if (this.mode === "3d") order.sort((a, b) => this.screen[a][2] - this.screen[b][2]);
      const nset = new Set(this.neigh);
      const base = this.mode === "3d" ? 4.6 : 4.6;
      for (const i of order) {
        const [x, y, d] = this.screen[i];
        if (i === this.sel || nset.has(i)) continue; // drawn on top below
        const r = this.mode === "3d" ? base * (0.8 + 0.35 * (d + 1) / 2) : base;
        ctx.globalAlpha = this.mode === "3d" ? 0.55 + 0.45 * (d + 1) / 2 : 1;
        ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = this.pts[i].kind === "heldout" ? t.held : t.train;
        ctx.fill();
        if (i === this.hover) { ctx.globalAlpha = 1; ctx.lineWidth = 1.5; ctx.strokeStyle = t.ink; ctx.stroke(); }
      }
      ctx.globalAlpha = 1;
      for (const j of this.neigh) {
        const [x, y] = this.screen[j];
        ctx.beginPath(); ctx.arc(x, y, base + 1.4, 0, 2 * Math.PI);
        ctx.fillStyle = t.paper; ctx.fill();
        ctx.lineWidth = 2; ctx.strokeStyle = t.accent; ctx.stroke();
        if (j === this.hover) { ctx.lineWidth = 1.5; ctx.strokeStyle = t.ink; ctx.beginPath(); ctx.arc(x, y, base + 3.2, 0, 2 * Math.PI); ctx.stroke(); }
      }
      if (this.sel != null && this.sel !== "upload" && this.screen[this.sel]) {
        const [x, y] = this.screen[this.sel];
        ctx.beginPath(); ctx.arc(x, y, base + 2.6, 0, 2 * Math.PI);
        ctx.fillStyle = t.accent; ctx.fill(); ctx.lineWidth = 1.5; ctx.strokeStyle = t.ink; ctx.stroke();
      }
      if (this.uploadScreen) {
        const [x, y] = this.uploadScreen, s = this.sel === "upload" ? 7.5 : 6;
        ctx.fillStyle = t.accent; ctx.fillRect(x - s, y - s, 2 * s, 2 * s);
        ctx.lineWidth = 1.5; ctx.strokeStyle = t.ink; ctx.strokeRect(x - s, y - s, 2 * s, 2 * s);
      }
    }

    _label(text, x, y, align = "left") {
      const ctx = this.ctx;
      ctx.fillStyle = this.t.muted; ctx.font = `10px ${this.t.font}`; ctx.textAlign = align;
      ctx.fillText(text, x, y); ctx.textAlign = "left";
    }

    _frame() {
      const ctx = this.ctx, w = this.c.clientWidth, h = this.c.clientHeight;
      const o = this._project([0, 0, 0]);
      ctx.strokeStyle = this.t.rule; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, Math.round(o[1]) + 0.5); ctx.lineTo(w, Math.round(o[1]) + 0.5);
      ctx.moveTo(Math.round(o[0]) + 0.5, 0); ctx.lineTo(Math.round(o[0]) + 0.5, h); ctx.stroke();
      this._label("PC1", w - 8, Math.round(o[1]) - 6, "right");
      this._label("PC2", Math.round(o[0]) + 6, 14);
    }

    _cube() {
      // three hairline axes through the centre (no bounding box: less ink, nothing runs off the pane)
      const ctx = this.ctx, e = CLIP;
      ctx.strokeStyle = this.t.rule; ctx.lineWidth = 1;
      const ax = [[[e, 0, 0], "PC1"], [[0, e, 0], "PC2"], [[0, 0, e], "PC3"]];
      ctx.beginPath();
      for (const [v] of ax) {
        const a = this._project(v.map((x) => -x)), b = this._project(v);
        ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
      for (const [v, lab] of ax) { const p = this._project(v); this._label(lab, p[0] + 4, p[1] - 4); }
    }

    _pick(mx, my) {
      let best = null, bd = 10 * 10;
      if (this.uploadScreen) {
        const d = (this.uploadScreen[0] - mx) ** 2 + (this.uploadScreen[1] - my) ** 2;
        if (d < bd) { bd = d; best = "upload"; }
      }
      this.screen.forEach(([x, y], i) => {
        const d = (x - mx) ** 2 + (y - my) ** 2;
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    }

    _bind() {
      const c = this.c;
      let drag = null;
      c.addEventListener("pointerdown", (e) => {
        drag = { x: e.clientX, y: e.clientY, moved: false };
        c.setPointerCapture(e.pointerId);
      });
      c.addEventListener("pointermove", (e) => {
        const r = c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
        if (drag) {
          const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
          if (drag.moved) {
            if (this.mode === "3d") {
              this.yaw += dx * 0.01;
              this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.01));
            } else { this.pan[0] += dx; this.pan[1] += dy; }
            drag.x = e.clientX; drag.y = e.clientY;
            this.draw();
            return;
          }
        }
        const hit = this._pick(mx, my);
        if (hit !== this.hover) { this.hover = hit; this.draw(); }
        c.style.cursor = hit != null ? "pointer" : this.mode === "3d" ? "grab" : "move";
        this.onHover(hit, mx, my);
      });
      c.addEventListener("pointerup", (e) => {
        if (drag && !drag.moved) {
          const r = c.getBoundingClientRect();
          const hit = this._pick(e.clientX - r.left, e.clientY - r.top);
          if (hit != null) this.onClick(hit);
        }
        drag = null;
      });
      c.addEventListener("pointerleave", () => { this.hover = null; this.onHover(null); this.draw(); });
      c.addEventListener("wheel", (e) => {
        e.preventDefault();
        this.zoom = Math.max(0.4, Math.min(12, this.zoom * Math.exp(-e.deltaY * 0.0015)));
        this.draw();
      }, { passive: false });
    }
  }
  root.ScatterPlot = ScatterPlot;
})(typeof self !== "undefined" ? self : this);
