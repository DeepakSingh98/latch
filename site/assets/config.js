// Public repository for Latch.
window.LATCH_SITE = {
  repoUrl: "https://github.com/DeepakSingh98/latch"
};

(function () {
  var url = (window.LATCH_SITE && window.LATCH_SITE.repoUrl) || "";
  function apply() {
    var links = document.querySelectorAll("[data-repo-link]");
    for (var i = 0; i < links.length; i++) {
      if (url) {
        links[i].setAttribute("href", url);
        links[i].removeAttribute("hidden");
        links[i].removeAttribute("aria-disabled");
      }
    }
    var pending = document.querySelectorAll("[data-repo-pending]");
    for (var j = 0; j < pending.length; j++) {
      if (url) pending[j].setAttribute("hidden", "");
    }
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", apply);
  else apply();
})();
