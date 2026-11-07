// Light/dark toggle, same behaviour and storage key as deepaksingh98.github.io. Loaded in <head> so the theme
// applies before first paint; dark unless the visitor has picked light.
(function () {
  var root = document.documentElement;
  var saved = null;
  try { saved = localStorage.getItem("theme"); } catch (e) {}
  root.dataset.theme = saved === "light" ? "light" : "dark";

  function current() {
    if (root.dataset.theme) return root.dataset.theme;
    return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var button = document.querySelector(".theme-toggle");
    if (!button) return;
    function label() { button.setAttribute("aria-label", current() === "dark" ? "Switch to light mode" : "Switch to dark mode"); }
    label();
    button.addEventListener("click", function () {
      var next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) {}
      label();
    });
  });
})();
