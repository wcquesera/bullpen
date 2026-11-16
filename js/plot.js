/* Canvas scatter for the embedding view: 2D (pan, zoom) or 3D (drag to rotate, zoom), hover and
 * click picking, a selected point with links to its neighbours, and one extra "upload" point.
 * No dependencies. Exposes window.ScatterPlot. */
(function (root) {
  "use strict";

  class ScatterPlot {
    constructor(canvas, { onHover, onClick } = {}) {
      this.c = canvas;
      this.ctx = canvas.getContext("2d");
      this.onHover = onHover || (() => {});
      this.onClick = onClick || (() => {});
      this.pts = [];          // [{p: [x, y, z], color}]
      this.upload = null;     // {p: [x, y, z]}
      this.sel = null;        // index or "upload"
      this.neigh = [];        // indices linked to the selection
      this.hover = null;
      this.mode = "3d";
      this.yaw = 0.6; this.pitch = 0.35; this.zoom = 1; this.pan = [0, 0];
      this.screen = [];       // per point: [sx, sy, depth]
      this.uploadScreen = null;
      this._bind();
      new ResizeObserver(() => this.draw()).observe(canvas);
    }

    setData(points) {
      this.pts = points;
      const all = points.map((q) => q.p);
      const lo = [0, 1, 2].map((k) => Math.min(...all.map((p) => p[k])));
      const hi = [0, 1, 2].map((k) => Math.max(...all.map((p) => p[k])));
      this.center = lo.map((v, k) => (v + hi[k]) / 2);
      this.radius = Math.max(...hi.map((v, k) => v - lo[k])) / 2 || 1;
      this.draw();
    }
    setMode(mode) { this.mode = mode; this.zoom = 1; this.pan = [0, 0]; this.draw(); }
    setSelection(sel, neigh) { this.sel = sel; this.neigh = neigh || []; this.draw(); }
    setUpload(p) { this.upload = p ? { p } : null; this.draw(); }
    resetView() { this.yaw = 0.6; this.pitch = 0.35; this.zoom = 1; this.pan = [0, 0]; this.draw(); }

    // data coords -> [screen x, screen y, depth] in CSS pixels
    _project(p) {
      const w = this.c.clientWidth, h = this.c.clientHeight;
      const s = (0.42 * Math.min(w, h) * this.zoom) / this.radius;
      let x = p[0] - this.center[0], y = p[1] - this.center[1], z = 0;
      if (this.mode === "3d") {
        z = p[2] - this.center[2];
        const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
        [x, z] = [cy * x + sy * z, -sy * x + cy * z];
        const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
        [y, z] = [cp * y - sp * z, sp * y + cp * z];
      }
      return [w / 2 + this.pan[0] + s * x, h / 2 + this.pan[1] - s * y, z / this.radius];
    }

    screenOf(i) {
      const v = i === "upload" ? this.uploadScreen : this.screen[i];
      if (!v) return null;
      const r = this.c.getBoundingClientRect();
      return { x: r.left + v[0], y: r.top + v[1] };
    }

    draw() {
      const c = this.c, ctx = this.ctx, dpr = window.devicePixelRatio || 1;
      const w = c.clientWidth, h = c.clientHeight;
      if (!w || !h || !this.pts.length) return;
      if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
        c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      this.screen = this.pts.map((q) => this._project(q.p));
      this.uploadScreen = this.upload ? this._project(this.upload.p) : null;
      if (this.mode === "3d") this._axes();
      else this._grid();

      const selXY = this.sel === "upload" ? this.uploadScreen : this.sel != null ? this.screen[this.sel] : null;
      if (selXY) {
        ctx.strokeStyle = "rgba(230,126,34,0.55)"; ctx.lineWidth = 1.2;
        for (const j of this.neigh) {
          const t = this.screen[j];
          ctx.beginPath(); ctx.moveTo(selXY[0], selXY[1]); ctx.lineTo(t[0], t[1]); ctx.stroke();
        }
      }
      const order = this.pts.map((_, i) => i);
      if (this.mode === "3d") order.sort((a, b) => this.screen[a][2] - this.screen[b][2]);
      const nset = new Set(this.neigh);
      const dim = this.sel != null;
      for (const i of order) {
        const [x, y, d] = this.screen[i];
        const isSel = i === this.sel, isN = nset.has(i), isH = i === this.hover;
        const r = (isSel ? 7 : isN ? 5.5 : 4) * (this.mode === "3d" ? 0.85 + 0.25 * (d + 1) / 2 : 1);
        ctx.globalAlpha = dim && !isSel && !isN ? 0.35 : this.mode === "3d" ? 0.6 + 0.4 * (d + 1) / 2 : 0.9;
        ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = isSel ? "#3498db" : isN ? "#e67e22" : this.pts[i].color;
        ctx.fill();
        if (isSel || isH) {
          ctx.globalAlpha = 1; ctx.lineWidth = 2; ctx.strokeStyle = isSel ? "#1f5f8b" : "#2c3e50"; ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      if (this.uploadScreen) {
        const [x, y] = this.uploadScreen, s = this.sel === "upload" ? 10 : 8;
        ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath();
        ctx.fillStyle = "#9b59b6"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#5b2c6f"; ctx.stroke();
      }
    }

    _axes() {
      const ctx = this.ctx, o = this.center;
      const len = this.radius * 0.35;
      const base = this._project(o);
      ["PC1", "PC2", "PC3"].forEach((lab, k) => {
        const p = o.slice(); p[k] += len;
        const t = this._project(p);
        ctx.strokeStyle = "#c9d1d6"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(base[0], base[1]); ctx.lineTo(t[0], t[1]); ctx.stroke();
        ctx.fillStyle = "#95a5a6"; ctx.font = "11px sans-serif"; ctx.fillText(lab, t[0] + 3, t[1] - 3);
      });
    }

    _grid() {
      const ctx = this.ctx, w = this.c.clientWidth, h = this.c.clientHeight;
      const o = this._project(this.center);
      ctx.strokeStyle = "#eef1f3"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(0, o[1]); ctx.lineTo(w, o[1]); ctx.moveTo(o[0], 0); ctx.lineTo(o[0], h); ctx.stroke();
      ctx.fillStyle = "#95a5a6"; ctx.font = "11px sans-serif";
      ctx.fillText("PC1 →", w - 44, o[1] - 5); ctx.fillText("PC2 ↑", o[0] + 5, 14);
    }

    _pick(mx, my) {
      let best = null, bd = 9 * 9;
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
        this.zoom = Math.max(0.3, Math.min(12, this.zoom * Math.exp(-e.deltaY * 0.0015)));
        this.draw();
      }, { passive: false });
    }
  }
  root.ScatterPlot = ScatterPlot;
})(typeof self !== "undefined" ? self : this);
