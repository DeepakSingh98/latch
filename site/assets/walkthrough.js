// Latch walkthrough interactions. Progressive enhancement only: without this
// script every step, mode and panel is rendered in order and remains readable.
(function () {
  "use strict";

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function reduceMotion() {
    return !!(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  // ── 1. Tiny training update stepper ────────────────────────────────────
  function fmt(v) {
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (a >= 0.01) return String(+v.toFixed(4));
    return String(+v.toPrecision(2));
  }

  function initStepper() {
    var fig = $("#fig-update");
    if (!fig) return;
    var items = $$("#steps li");
    var count = $("#step-count");
    var lrSel = $("#lr");
    var prev = $('[data-step="prev"]', fig), next = $('[data-step="next"]', fig), reset = $('[data-step="reset"]', fig);
    var N = items.length;
    var i = 0, first = true;

    // computed: nodes visible from this step on; active: highlighted; read: inputs used
    var S = [
      { add: ["x", "w1", "w2", "t"], active: ["x", "w1", "w2", "t"], read: [], edges: [] },
      { add: ["h"], active: ["h"], read: ["x", "w1"], edges: ["e1", "e2"] },
      { add: ["y"], active: ["y"], read: ["h", "w2"], edges: ["e3", "e4"] },
      { add: ["L"], active: ["L"], read: ["y", "t"], edges: ["e5", "e6"] },
      { add: ["gy"], active: ["gy"], read: ["y", "t"], edges: ["b1"] },
      { add: ["gw2"], active: ["gw2"], read: ["gy", "h"], edges: ["b2"] },
      { add: ["gh"], active: ["gh"], read: ["gy", "w2"], edges: ["b3"] },
      { add: ["gw1"], active: ["gw1"], read: ["gh", "x"], edges: ["b4"] },
      { add: [], active: ["w1", "w2"], read: ["gw1", "gw2"], edges: ["u1", "u2"] },
      { add: [], active: ["h", "y", "L"], read: ["x", "w1", "w2"], edges: ["e1", "e2", "e3", "e4", "e5", "e6"] }
    ];
    var edgeSeen = ["e1", "e2", "e3", "e4", "e5", "e6", "b1", "b2", "b3", "b4", "u1", "u2"];

    function values() {
      var lr = parseFloat(lrSel.value);
      var w1 = 0.5 - lr * 1.5, w2 = 1.5 - lr * 0.5;
      var h = w1 * 2, y = w2 * h, L = 0.5 * (y - 1) * (y - 1);
      return { lr: lr, w1: w1, w2: w2, h: h, y: y, L: L };
    }

    function setText(key, txt) {
      $$('[data-v="' + key + '"]', fig).forEach(function (el) { el.textContent = txt; });
    }

    function render() {
      var shown = {}, k, s;
      for (k = 0; k <= i; k++) S[k].add.forEach(function (n) { shown[n] = true; });
      s = S[i];
      $$(".node", fig).forEach(function (g) {
        var id = g.id.slice(2);
        g.classList.toggle("pending", !shown[id]);
        g.classList.toggle("active", s.active.indexOf(id) >= 0);
        g.classList.toggle("read", s.read.indexOf(id) >= 0);
      });
      var seen = {};
      for (k = 0; k <= i; k++) S[k].edges.forEach(function (e) { seen[e] = true; });
      edgeSeen.forEach(function (e) {
        var g = document.getElementById(e);
        if (!g) return;
        g.classList.toggle("pending", !seen[e]);
        g.classList.toggle("active", s.edges.indexOf(e) >= 0);
      });
      var v = values();
      setText("lr", fmt(v.lr));
      setText("w1n", fmt(v.w1));
      setText("w2n", fmt(v.w2));
      setText("hn", fmt(v.h));
      setText("yn", fmt(v.y));
      setText("Ln", fmt(v.L));
      setText("dir", v.L < 0.125 ? "down" : "up");
      setText("verdict", v.L < 0.125 ? "One update made the prediction better." : "The step was too large. The loss rose.");
      // weights show their updated values from step 9; forward values refresh at step 10
      setText("w1", i >= 8 ? fmt(v.w1) : "0.5");
      setText("w2", i >= 8 ? fmt(v.w2) : "1.5");
      setText("h", i >= 9 ? fmt(v.h) : "1");
      setText("y", i >= 9 ? fmt(v.y) : "1.5");
      setText("L", i >= 9 ? fmt(v.L) : "0.125");
      items.forEach(function (li, j) { li.classList.toggle("current", j === i); });
      if (!first) keepInView(s.active);
      first = false;
      prev.disabled = i === 0;
      next.disabled = i === N - 1;
      if (count) count.textContent = "Step " + (i + 1) + " of " + N + " · ← → keys";
    }

    // On narrow screens the diagram scrolls sideways; keep the active nodes visible.
    var box = fig.querySelector(".scroll-x");
    function keepInView(ids) {
      if (!box || box.scrollWidth <= box.clientWidth + 1 || !ids.length) return;
      var b = box.getBoundingClientRect(), lo = Infinity, hi = -Infinity;
      ids.forEach(function (id) {
        var g = document.getElementById("n-" + id);
        if (!g) return;
        var r = g.getBoundingClientRect();
        lo = Math.min(lo, r.left - b.left + box.scrollLeft);
        hi = Math.max(hi, r.right - b.left + box.scrollLeft);
      });
      var target = Math.max(0, Math.min((lo + hi) / 2 - box.clientWidth / 2, box.scrollWidth - box.clientWidth));
      box.scrollTo({ left: target, behavior: reduceMotion() ? "auto" : "smooth" });
    }

    function go(d) { i = Math.max(0, Math.min(N - 1, i + d)); render(); }
    prev.addEventListener("click", function () { go(-1); });
    next.addEventListener("click", function () { go(1); });
    reset.addEventListener("click", function () { i = 0; render(); next.focus(); });
    lrSel.addEventListener("change", render);
    fig.addEventListener("keydown", function (e) {
      if (e.target.tagName === "SELECT") return;
      if (e.key === "ArrowRight") { go(1); e.preventDefault(); }
      else if (e.key === "ArrowLeft") { go(-1); e.preventDefault(); }
    });
    render();
  }

  // ── Segmented toggles showing one of several figures ───────────────────
  function initToggles() {
    $$("[data-toggle]").forEach(function (group) {
      var name = group.getAttribute("data-toggle");
      var buttons = $$("button[data-show]", group);
      function show(id) {
        buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b.getAttribute("data-show") === id)); });
        $$('[data-group="' + name + '"]').forEach(function (p) { p.hidden = p.id !== id; });
        group.dispatchEvent(new CustomEvent("shown", { detail: id }));
      }
      buttons.forEach(function (b) { b.addEventListener("click", function () { show(b.getAttribute("data-show")); }); });
      show(buttons[0].getAttribute("data-show"));
    });
  }

  // ── Fusion highlights ──────────────────────────────────────────────────
  function initFusion() {
    var fig = $("#fig-fuse");
    if (!fig) return;
    var out = $("#fuse-explain");
    var view = "fuse-kernels", hl = "none";
    var text = {
      "fuse-kernels": {
        none: "Seven kernels run in sequence. Each starts after the previous kernel has finished on every SM.",
        sync: "There are six kernel boundaries. Each kernel waits for its slowest SM to finish before the next kernel starts.",
        mem: "Every intermediate goes out to GPU memory and comes back for the next kernel.",
        arith: "Both versions perform the same multiplications and additions."
      },
      "fuse-mega": {
        none: "One kernel stays resident for the whole update. Its work is divided into four phases.",
        sync: "Three grid-wide barriers remain. Every SM waits for the slowest at each barrier. Shading shows this waiting time.",
        mem: "Fused steps keep values on chip. Values passed across a barrier use the arena in GPU memory. Weights are also read and written there.",
        arith: "Fusion leaves the arithmetic unchanged."
      }
    };
    function upd() {
      fig.setAttribute("data-hl", hl);
      if (out) out.textContent = text[view][hl];
    }
    var tg = $('[data-toggle="fuse"]', fig);
    tg.addEventListener("shown", function (e) { view = e.detail; upd(); });
    var hb = $$("[data-hl-group] button", fig);
    hb.forEach(function (b) {
      b.addEventListener("click", function () {
        hl = b.getAttribute("data-hl");
        hb.forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); });
        upd();
      });
    });
    upd();
  }

  // ── Tabs (pipeline stages and workload switcher) ───────────────────────
  function initTabs() {
    $$('[role="tablist"]').forEach(function (list) {
      var tabs = $$('[role="tab"]', list);
      function select(tab, focus) {
        tabs.forEach(function (t) {
          var on = t === tab;
          t.setAttribute("aria-selected", String(on));
          t.tabIndex = on ? 0 : -1;
          var p = document.getElementById(t.getAttribute("aria-controls"));
          if (p) p.hidden = !on;
        });
        if (focus) tab.focus();
      }
      tabs.forEach(function (t, k) {
        t.addEventListener("click", function () { select(t, false); });
        t.addEventListener("keydown", function (e) {
          var n = null;
          if (e.key === "ArrowRight" || e.key === "ArrowDown") n = tabs[(k + 1) % tabs.length];
          else if (e.key === "ArrowLeft" || e.key === "ArrowUp") n = tabs[(k - 1 + tabs.length) % tabs.length];
          else if (e.key === "Home") n = tabs[0];
          else if (e.key === "End") n = tabs[tabs.length - 1];
          if (n) { e.preventDefault(); select(n, true); }
        });
      });
      var initial = tabs.filter(function (t) { return t.getAttribute("aria-selected") === "true"; })[0] || tabs[0];
      select(initial, false);
    });
  }

  function init() { initStepper(); initToggles(); initFusion(); initTabs(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
