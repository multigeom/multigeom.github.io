/* Interactive parts of the landing page: eigenvector teaser, one-neuron widget, eigenvector explorer, BibTeX copy.
   Data: static/data/{teaser,tuning,explorer}.json, made by scripts/make_*.py. No dependencies. */
(function () {
  "use strict";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

  /* ------------------------------------------------------------------ teaser */
  async function teaser() {
    const T = await (await fetch("static/data/teaser.json")).json();
    const KEYS = ["G_V1", "G_V4", "J_V1", "J_V4"];
    const vec = (o) => { const q = new Int8Array(b64(o.b64).buffer); return Array.from(q, (v) => (v / 127) * o.scale); };
    const decode = (D) => {
      const out = { x0: Array.from(b64(D.image), (u) => u / 127.5 - 1), V: {} };
      for (const key of KEYS) out.V[key] = D[key].map(vec);
      return out;
    };
    // image 0 is the frog of teaser.json; the others are static/data/teaser_img<idx>.json, loaded on demand
    const files = [null, ...(T.cycle || []).map((i) => `static/data/teaser_img${String(i).padStart(4, "0")}.json`)];
    const cache = [decode(T)];
    let cur = 0, x0 = cache[0].x0, V = cache[0].V;
    const amp = document.getElementById("t-amp"), play = document.getElementById("t-play");
    let k = 0, playing = true, phase = 0;
    const ctx = {};
    for (const key of KEYS) ctx[key] = document.getElementById("t-" + key.replace("_", "-")).getContext("2d");
    const nav = document.getElementById("t-img-nav"), count = document.getElementById("t-img-count");
    const note = document.getElementById("t-note"), frogNote = note.textContent;
    const otherNote = "The leading eigenvectors of the multi-scale metric change the image in smooth, spatially coherent ways. " +
      "Those of the Fisher information add high-frequency noise, which natural images rarely contain.";
    async function show(i) {
      cur = (i + files.length) % files.length;
      if (!cache[cur]) cache[cur] = decode(await (await fetch(files[cur])).json());
      ({ x0, V } = cache[cur]);
      count.textContent = `Image ${cur + 1} of ${files.length}`;
      note.textContent = cur === 0 ? frogNote : otherNote;
      draw();
    }
    if (files.length > 1) {
      nav.hidden = false; count.textContent = `Image 1 of ${files.length}`;
      document.getElementById("t-prev").addEventListener("click", () => show(cur - 1).catch(console.error));
      document.getElementById("t-next").addEventListener("click", () => show(cur + 1).catch(console.error));
    }
    function draw() {
      const s = amp.value / 100;
      for (const key in V) {
        const v = V[key][k];
        let m = 0; for (const a of v) m = Math.max(m, Math.abs(a));
        const a = (s * 1.6) / (m || 1);              // largest pixel change: 1.6 (image range is 2)
        const img = ctx[key].createImageData(64, 64);
        for (let i = 0; i < 4096; i++) {
          const g = Math.max(-1, Math.min(1, x0[i] + a * v[i]));
          const u = Math.round((g + 1) * 127.5);
          img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = u; img.data[4 * i + 3] = 255;
        }
        ctx[key].putImageData(img, 0, 0);
      }
    }
    function tick() {
      if (playing) { phase += 0.035; amp.value = Math.round(100 * Math.sin(phase)); draw(); }
      requestAnimationFrame(tick);
    }
    amp.addEventListener("input", () => { playing = false; play.textContent = "▶"; draw(); });
    play.addEventListener("click", () => { playing = !playing; play.textContent = playing ? "❚❚" : "▶"; });
    document.querySelectorAll("#t-k button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#t-k button").forEach((x) => x.classList.toggle("on", x === b));
      k = +b.dataset.k; draw();
    }));
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) { playing = false; play.textContent = "▶"; }
    draw(); tick();
  }

  /* ------------------------------------------------------------------ tiny line plot */
  function plot(canvas, series, o) {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const g = canvas.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const L = 34, R = 8, Tp = 8, B = o.bottom || 26, pw = W - L - R, ph = H - Tp - B;
    const [x0, x1] = o.xlim, [y0, y1] = o.ylim;
    const X = (x) => L + ((x - x0) / (x1 - x0)) * pw, Y = (y) => Tp + ph - ((y - y0) / (y1 - y0)) * ph;
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.beginPath();
    g.moveTo(L, Tp); g.lineTo(L, Tp + ph); g.lineTo(L + pw, Tp + ph); g.stroke();
    g.fillStyle = css("--muted"); g.font = "11px Inter, sans-serif"; g.textAlign = "center";
    for (const t of o.xticks || []) g.fillText(String(t), X(t), Tp + ph + 15);
    if (o.xlabel) g.fillText(o.xlabel, L + pw / 2, H - 1);
    g.textAlign = "right";
    for (const t of o.yticks || []) g.fillText(o.yfmt ? o.yfmt(t) : String(t), L - 5, Y(t) + 4);
    if (o.vline != null) {
      g.strokeStyle = css("--muted"); g.setLineDash([3, 3]); g.beginPath();
      g.moveTo(X(o.vline), Tp); g.lineTo(X(o.vline), Tp + ph); g.stroke(); g.setLineDash([]);
    }
    for (const s of series) {
      if (s.fill) {
        g.fillStyle = s.fill; g.beginPath(); g.moveTo(X(s.x[0]), Y(0));
        s.x.forEach((x, i) => g.lineTo(X(x), Y(s.y[i]))); g.lineTo(X(s.x[s.x.length - 1]), Y(0)); g.fill(); continue;
      }
      g.globalAlpha = s.alpha || 1; g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []); g.beginPath();
      s.x.forEach((x, i) => { const y = Math.max(y0, Math.min(y1, s.y[i])); i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); });
      g.stroke(); g.setLineDash([]); g.globalAlpha = 1;
    }
    return { X, Y, g };
  }

  /* ------------------------------------------------------------------ one-neuron widget */
  async function widget() {
    const D = await (await fetch("static/data/tuning.json")).json();
    const names = ["monotonic", "low-gain", "single-peaked"];
    let col = {};
    const dash = { "monotonic": [], "low-gain": [], "single-peaked": [6, 4] };
    const slider = document.getElementById("w-t"), tval = document.getElementById("w-t-val"), note = document.getElementById("w-note");
    let metric = "G";

    const xt = [-4, -2, 0, 2, 4];
    // tick labels of an autoscaled axis, and its rounded-up maximum (a multiple of 0.5 x a power of ten)
    const fmt = (v) => (v === 0 ? "0" : Math.abs(v) >= 1e3 || Math.abs(v) < 1e-2 ? v.toExponential(0).replace("e+", "e") : String(+v.toPrecision(2)));
    const nice = (v) => { const e = Math.pow(10, Math.floor(Math.log10(v))); return (Math.ceil((v / e) * 2) / 2) * e; };
    function draw() {
      col = { "monotonic": css("--c-mono"), "low-gain": css("--c-low"), "single-peaked": css("--c-peak") };
      document.getElementById("w-legend").innerHTML = names.map((n) =>
        `<span><i style="background:${col[n]}"></i>${n}</span>`).join("") +
        `<span><i style="background:${css("--line")}"></i>uniform stimulus distribution</span>`;
      const ti = +slider.value, t = D.t[ti];
      tval.textContent = "t = " + (t < 0.01 ? t.toExponential(1) : t.toPrecision(2));
      const pmax = Math.max(...D.prior);
      // tuning curves
      plot(document.getElementById("w-tuning"),
        [{ x: D.x, y: D.prior.map((p) => (p / pmax) * 0.3), fill: css("--line") }]
          .concat(names.map((n) => ({ x: D.x, y: D.curves[n].f, color: col[n], dash: dash[n] }))),
        { xlim: [-4, 4], ylim: [-0.05, 1.1], xticks: xt, yticks: [0, 0.5, 1], xlabel: "stimulus x" });
      // panel 2: G(x) integrated up to the scale t of the slider, divided by t (it equals J(x) as t -> 0), or J(x) itself
      const cap = document.getElementById("w-cap-contrib");
      let ys;
      if (metric === "G") {
        cap.textContent = "G(x) up to scale t, divided by t";
        ys = names.map((n) => D.curves[n].cum[ti].map((v) => v / t));
      } else {
        cap.textContent = "Fisher information J(x)";
        ys = names.map((n) => D.curves[n].J);
      }
      const ymax2 = nice(Math.max(...ys.flat()) * 1.05);
      plot(document.getElementById("w-contrib"),
        names.map((n, i) => ({ x: D.x, y: ys[i], color: col[n], dash: dash[n] })),
        { xlim: [-4, 4], ylim: [0, ymax2], xticks: xt, yticks: [0, ymax2 / 2, ymax2], yfmt: fmt, xlabel: "stimulus x" });
      // panel 3: distance from x0 = -4, the integral of sqrt(metric) from x0: under G integrated up to t, or under J
      const dcap = document.getElementById("w-cap-dist");
      let dist;
      if (metric === "G") {
        dcap.textContent = "Distance from x₀ = −4, G up to t";
        dist = names.map((n) => {
          const g = D.curves[n].cum[ti], out = [0];
          for (let i = 1; i < g.length; i++) out.push(out[i - 1] + 0.5 * (Math.sqrt(g[i]) + Math.sqrt(g[i - 1])) * (D.x[i] - D.x[i - 1]));
          return out;
        });
      } else {
        dcap.textContent = "Distance from x₀ = −4, Fisher";
        dist = names.map((n) => D.curves[n].dFR);
      }
      const dmax = nice(Math.max(...dist.flat()) * 1.05);
      plot(document.getElementById("w-dist"),
        names.map((n, i) => ({ x: D.x, y: dist[i], color: col[n], dash: dash[n] })),
        { xlim: [-4, 4], ylim: [0, dmax], xticks: xt, yticks: [0, dmax / 2, dmax], yfmt: fmt, xlabel: "stimulus x", vline: D.x0 });
      // panel 4: mutual information (bar) and 1/2 <G> averaged over the uniform stimulus, G integrated up to t (tick)
      document.getElementById("w-cap-mi").textContent = metric === "G" ? "Mutual info (bits) and ½⟨G⟩ up to t" : "Mutual information (bits) and ½⟨G⟩";
      const meanX = (g) => { let a = 0; for (let i = 1; i < g.length; i++) a += 0.5 * (g[i] + g[i - 1]) * (D.x[i] - D.x[i - 1]); return a / (D.x[D.x.length - 1] - D.x[0]); };
      const c = document.getElementById("w-mi");
      const mmax = Math.ceil(Math.max(...names.map((n) => D.curves[n].mi_bits)) * 1.15 * 2) / 2;
      const P = plot(c, [], { xlim: [0, 3], ylim: [0, mmax], xticks: [], yticks: [0, 1, 2, 3].filter((v) => v <= mmax), bottom: 68 });
      names.forEach((n, i) => {
        const mi = D.curves[n].mi_bits;
        const mg = metric === "G" ? (0.5 * meanX(D.curves[n].cum[ti])) / Math.LN2 : D.curves[n].mi_from_G_bits;
        P.g.fillStyle = col[n]; P.g.fillRect(P.X(i + 0.25), P.Y(mi), P.X(i + 0.75) - P.X(i + 0.25), P.Y(0) - P.Y(mi));
        P.g.strokeStyle = css("--ink"); P.g.lineWidth = 2; P.g.beginPath();
        P.g.moveTo(P.X(i + 0.18), P.Y(mg)); P.g.lineTo(P.X(i + 0.82), P.Y(mg)); P.g.stroke();
        P.g.fillStyle = css("--muted"); P.g.textAlign = "center"; P.g.font = "10px Inter, sans-serif";
        // hyphenated names on two lines, so that neighbouring labels do not overlap in a narrow panel
        const parts = n.includes("-") ? [n.split("-")[0] + "-", n.split("-")[1]] : [n];
        parts.forEach((line, k) => P.g.fillText(line, P.X(i + 0.5), P.Y(0) + 13 + 11 * k));
        // the neuron's tuning curve, drawn small under its label
        const x0t = P.X(i + 0.2), wt = P.X(i + 0.8) - x0t, y0t = P.Y(0) + 38, ht = 22, f = D.curves[n].f;
        P.g.strokeStyle = col[n]; P.g.lineWidth = 1.5; P.g.beginPath();
        f.forEach((v, k) => { const px = x0t + (k / (f.length - 1)) * wt, py = y0t + ht - v * ht; k ? P.g.lineTo(px, py) : P.g.moveTo(px, py); });
        P.g.stroke();
      });
      const ref = (id, txt) => `<b class="pref" data-panel="${id}" tabindex="0">${txt}</b>`;
      note.innerHTML = metric === "G"
        ? (t < 0.002
            ? `At the finest scales the ${ref("w-contrib", "G(x)")} panel equals the Fisher information J(x), so the monotonic and single-peaked neurons coincide.`
            : `At coarser scales the ${ref("w-contrib", "G(x)")} panel falls below the Fisher information and the single-peaked neuron gains less than the monotonic one: once the stimulus is coarse-grained, stimuli on either side of its peak give similar response distributions. In the ${ref("w-mi", "Mutual information")} panel, the bars show the mutual information and the black ticks ½⟨G⟩ integrated up to t, which reach the bars as t grows.`)
        : `Fisher geometry: the monotonic and single-peaked neurons have identical Fisher information (${ref("w-contrib", "J(x)")} panel) and identical distances (${ref("w-dist", "Distance")} panel), and the low-gain neuron has the smallest. Their mutual information (${ref("w-mi", "Mutual information")} panel) is ordered differently: monotonic, then low-gain, then single-peaked.`;
    }
    slider.addEventListener("input", draw);
    document.querySelectorAll("#w-metric button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#w-metric button").forEach((x) => x.classList.toggle("on", x === b));
      metric = b.dataset.m;
      // the Fisher information has no scale t: hide the slider and its label (the toggle stays on the right)
      for (const el of [slider, tval, document.querySelector('label[for="w-t"]')]) el.hidden = metric === "J";
      draw();
    }));
    slider.value = 30; draw();
    window.addEventListener("resize", draw);
  }

  /* ------------------------------------------------------------------ explorer */
  async function explorer() {
    const E = await (await fetch("static/data/explorer.json")).json();
    const thumbs = document.getElementById("x-thumbs"), sprite = document.getElementById("x-sprite"), rows = document.getElementById("x-rows");
    const enc = { inception: "Inception", resnet50: "ResNet-50" }, met = { G: "multi-scale", J: "Fisher" };
    const n = E.rows.length, tot = n * (E.tile + E.pad) - E.pad;
    rows.style.gridTemplateRows = E.rows.map((_, i) => `${E.tile}fr` + (i < n - 1 ? ` ${E.pad}fr` : "")).join(" ");
    rows.innerHTML = E.rows.map((r, i) => `<div class="${i === n / 2 ? "sep" : ""}">${met[r.metric]} · ${r.area} · ${enc[r.encoder]}</div>` +
      (i < n - 1 ? "<span></span>" : "")).join("");
    function select(idx, el) {
      sprite.src = `static/img/explorer/img${String(idx).padStart(4, "0")}.png`;

      thumbs.querySelectorAll("img").forEach((x) => x.classList.toggle("on", x === el));
    }
    E.images.forEach((idx, i) => {
      const im = document.createElement("img");
      im.src = `static/img/explorer/thumb${String(idx).padStart(4, "0")}.png`; im.alt = "test image " + idx; im.loading = "lazy";
      im.addEventListener("click", () => select(idx, im));
      thumbs.appendChild(im);
      if (i === 0) select(idx, im);
    });
    sprite.addEventListener("load", () => { rows.style.height = sprite.clientHeight + "px"; });
    window.addEventListener("resize", () => { rows.style.height = sprite.clientHeight + "px"; });
    void tot;
  }

  /* ------------------------------------------------------------------ text -> panel links */
  function panelLinks() {
    const fig = (el) => { const c = document.getElementById(el.dataset.panel); return c && c.closest(".wfig"); };
    const on = (el) => { const f = fig(el); if (f) f.classList.add("hl"); };
    const off = (el) => { const f = fig(el); if (f) f.classList.remove("hl"); };
    const target = (e) => e.target.closest && e.target.closest(".pref");
    document.addEventListener("mouseover", (e) => { const el = target(e); if (el) on(el); });
    document.addEventListener("mouseout", (e) => { const el = target(e); if (el) off(el); });
    document.addEventListener("focusin", (e) => { const el = target(e); if (el) on(el); });
    document.addEventListener("focusout", (e) => { const el = target(e); if (el) off(el); });
    document.addEventListener("click", (e) => {       // touch screens: show the panel for a moment
      const el = target(e); if (!el) return;
      const f = fig(el); if (!f) return;
      f.scrollIntoView({ block: "nearest", behavior: "smooth" }); on(el); setTimeout(() => off(el), 2500);
    });
  }

  /* ------------------------------------------------------------------ light / dark switch */
  function theme() {
    const b = document.getElementById("theme"), root = document.documentElement;
    const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const label = () => (b.textContent = isDark() ? "◐ Light" : "◐ Dark");
    b.addEventListener("click", () => {
      root.dataset.theme = isDark() ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
      label(); window.dispatchEvent(new Event("resize"));          // canvases redraw with the new colours
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { label(); window.dispatchEvent(new Event("resize")); });
    label();
  }

  /* ------------------------------------------------------------------ copy BibTeX */
  function bib() {
    const b = document.getElementById("copy-bib");
    b.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(document.getElementById("bibtex").innerText); b.textContent = "Copied"; }
      catch (e) { b.textContent = "Select and copy"; }
      setTimeout(() => (b.textContent = "Copy"), 1500);
    });
  }

  document.addEventListener("DOMContentLoaded", () => {
    panelLinks(); teaser().catch(console.error); widget().catch(console.error); explorer().catch(console.error); bib(); theme();
  });
})();
