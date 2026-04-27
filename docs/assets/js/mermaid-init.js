(function () {
  function renderMermaidBlocks() {
    if (!window.mermaid) {
      return;
    }

    document.querySelectorAll("pre > code.language-mermaid").forEach(function (code) {
      var wrapper = document.createElement("div");
      wrapper.className = "mermaid";
      wrapper.textContent = code.textContent;
      code.parentElement.replaceWith(wrapper);
    });

    window.mermaid.initialize({
      startOnLoad: false,
      theme: "default",
      securityLevel: "loose"
    });
    window.mermaid.run({ querySelector: ".mermaid" });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderMermaidBlocks);
  } else {
    renderMermaidBlocks();
  }
})();
