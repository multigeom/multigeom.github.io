/* Interactive parts of the landing page: eigenvector teaser, one-neuron widget, eigenvector explorer, BibTeX copy.
   Data: static/data/{teaser,tuning,explorer}.json, made by scripts/make_*.py. No dependencies. */
(function () {
  "use strict";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const b64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));

  /* ------------------------------------------------------------------ teaser */
  async function teaser() {
    const T = await (await fetch("static/data/teaser.json")).json();
    const x0 = Array.from(b64(T.image), (u) => u / 127.5 - 1);
    const vec = (o) => { const q = new Int8Array(b64(o.b64).buffer); return Array.from(q, (v) => (v / 127) * o.scale); };
    const V = {};
    for (const key of ["G_V1", "G_V4", "J_V1", "J_V4"]) V[key] = T[key].map(vec);
    const amp = document.getElementById("t-amp"), play = document.getElementById("t-play");
    let k = 0, playing = true, phase = 0;
    const ctx = {};
    for (const key in V) {
      const c = document.getElementById("t-" + key.replace("_", "-"));
      ctx[key] = c.getContext("2d");
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
    const L = 34, R = 8, Tp = 8, B = 26, pw = W - L - R, ph = H - Tp - B;
    const [x0, x1] = o.xlim, [y0, y1] = o.ylim;
    const X = (x) => L + ((x - x0) / (x1 - x0)) * pw, Y = (y) => Tp + ph - ((y - y0) / (y1 - y0)) * ph;
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.beginPath();
    g.moveTo(L, Tp); g.lineTo(L, Tp + ph); g.lineTo(L + pw, Tp + ph); g.stroke();
    g.fillStyle = css("--muted"); g.font = "11px Inter, sans-serif"; g.textAlign = "center";
    for (const t of o.xticks || []) g.fillText(String(t), X(t), Tp + ph + 15);
    if (o.xlabel) g.fillText(o.xlabel, L + pw / 2, H - 1);
    g.textAlign = "right";
    for (const t of o.yticks || []) g.fillText(String(t), L - 5, Y(t) + 4);
    if (o.vline != null) {
      g.strokeStyle = css("--muted"); g.setLineDash([3, 3]); g.beginPath();
      g.moveTo(X(o.vline), Tp); g.lineTo(X(o.vline), Tp + ph); g.stroke(); g.setLineDash([]);
    }
    for (const s of series) {
      if (s.fill) {
        g.fillStyle = s.fill; g.beginPath(); g.moveTo(X(s.x[0]), Y(0));
        s.x.forEach((x, i) => g.lineTo(X(x), Y(s.y[i]))); g.lineTo(X(s.x[s.x.length - 1]), Y(0)); g.fill(); continue;
      }
      g.strokeStyle = s.color; g.lineWidth = s.width || 2; g.setLineDash(s.dash || []); g.beginPath();
      s.x.forEach((x, i) => { const y = Math.max(y0, Math.min(y1, s.y[i])); i ? g.lineTo(X(x), Y(y)) : g.moveTo(X(x), Y(y)); });
      g.stroke(); g.setLineDash([]);
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
    function draw() {
      col = { "monotonic": css("--c-mono"), "low-gain": css("--c-low"), "single-peaked": css("--c-peak") };
      document.getElementById("w-legend").innerHTML = names.map((n) =>
        `<span><i style="background:${col[n]}"></i>${n}</span>`).join("") +
        `<span><i style="background:${css("--line")}"></i>stimulus distribution</span>`;
      const ti = +slider.value, t = D.t[ti];
      tval.textContent = "t = " + (t < 0.01 ? t.toExponential(1) : t.toPrecision(2));
      const pmax = Math.max(...D.prior);
      // tuning curves
      plot(document.getElementById("w-tuning"),
        [{ x: D.x, y: D.prior.map((p) => (p / pmax) * 0.35 - 0.05), fill: css("--line") }]
          .concat(names.map((n) => ({ x: D.x, y: D.curves[n].f, color: col[n], dash: dash[n] }))),
        { xlim: [-4, 4], ylim: [-1.1, 1.1], xticks: xt, yticks: [-1, 0, 1], xlabel: "stimulus x" });
      // information gained at scale t (multi-scale) or Fisher information (t -> 0)
      const ys = names.map((n) => (metric === "G" ? D.curves[n].contrib[ti] : D.curves[n].J));
      const ymax = metric === "G" ? Math.max(...names.flatMap((n) => D.curves[n].contrib.flat())) * 1.05
                                  : Math.max(...ys.flat()) * 1.05;
      plot(document.getElementById("w-contrib"),
        names.map((n, i) => ({ x: D.x, y: ys[i], color: col[n], dash: dash[n] })),
        { xlim: [-4, 4], ylim: [0, ymax], xticks: xt, yticks: [], xlabel: "stimulus x" });
      // distance from x0
      const dk = metric === "G" ? "dG" : "dFR";
      const dmax = Math.max(...names.flatMap((n) => D.curves[n][dk])) * 1.05;
      plot(document.getElementById("w-dist"),
        names.map((n) => ({ x: D.x, y: D.curves[n][dk], color: col[n], dash: dash[n] })),
        { xlim: [-4, 4], ylim: [0, dmax], xticks: xt, yticks: [], xlabel: "stimulus x", vline: D.x0 });
      // MI bars: direct (bar) and 1/2 <G> (tick)
      const c = document.getElementById("w-mi"), mmax = 2.2;
      const P = plot(c, [], { xlim: [0, 3], ylim: [0, mmax], xticks: [], yticks: [0, 1, 2] });
      names.forEach((n, i) => {
        const mi = D.curves[n].mi_bits, mg = D.curves[n].mi_from_G_bits;
        P.g.fillStyle = col[n]; P.g.fillRect(P.X(i + 0.25), P.Y(mi), P.X(i + 0.75) - P.X(i + 0.25), P.Y(0) - P.Y(mi));
        P.g.strokeStyle = css("--ink"); P.g.lineWidth = 2; P.g.beginPath();
        P.g.moveTo(P.X(i + 0.18), P.Y(mg)); P.g.lineTo(P.X(i + 0.82), P.Y(mg)); P.g.stroke();
        P.g.fillStyle = css("--muted"); P.g.textAlign = "center"; P.g.font = "10px Inter, sans-serif";
        P.g.fillText(n, P.X(i + 0.5), P.Y(0) + 14);
      });
      note.textContent = metric === "G"
        ? (t < 0.05
            ? "At fine scales the information gained is the Fisher information: the monotonic and single-peaked curves coincide."
            : "As the stimulus is blurred, the single-peaked neuron loses information fastest: stimuli on either side of its peak become confusable. Bars: mutual information; black ticks: ½⟨G⟩, the same number.")
        : "Fisher geometry: the monotonic and single-peaked neurons have identical Fisher information and identical distances, yet different mutual information.";
    }
    slider.addEventListener("input", draw);
    document.querySelectorAll("#w-metric button").forEach((b) => b.addEventListener("click", () => {
      document.querySelectorAll("#w-metric button").forEach((x) => x.classList.toggle("on", x === b));
      metric = b.dataset.m; slider.disabled = metric === "J"; draw();
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
    teaser().catch(console.error); widget().catch(console.error); explorer().catch(console.error); bib(); theme();
  });
})();
