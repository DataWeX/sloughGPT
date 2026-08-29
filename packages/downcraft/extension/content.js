/**
 * Downcraft Capture — Content Script
 * Intercepts clicks on download links and sends them to the local server.
 */

(function () {
  "use strict";

  const EXTENSIONS = /\.(zip|tar\.gz|tar\.bz2|tar\.xz|rar|7z|iso|dmg|exe|msi|deb|rpm|apk|mp4|webm|mkv|avi|mov|mp3|flac|wav|ogg|pdf|docx?|xlsx?|pptx?|csv|json|xml|txt|safetensors|bin|pt|pth|onnx|gguf)$/i;
  const SERVER = "http://localhost:6400";

  function capture(url, title) {
    fetch(`${SERVER}/capture`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, title: title || document.title }),
    }).catch(() => {});
  }

  function getUrl(el) {
    const a = el.closest("a[href]");
    if (a && EXTENSIONS.test(a.href)) return a.href;
    return null;
  }

  window.addEventListener("mousedown", function (e) {
    const url = getUrl(e.target);
    if (url && e.button === 0 && !e.ctrlKey && !e.metaKey && !e.shiftKey) {
      capture(url, e.target.textContent?.trim());
    }
  }, true);

  window.addEventListener("click", function (e) {
    const url = getUrl(e.target);
    if (url) capture(url, e.target.textContent?.trim());
  }, true);
})();
