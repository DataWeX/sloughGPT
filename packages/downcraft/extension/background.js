/**
 * Downcraft Capture — Background Service Worker
 * Catches downloads that the content script missed.
 */

const SERVER = "http://localhost:6400";
const EXTENSIONS = /\.(zip|tar\.gz|tar\.bz2|tar\.xz|rar|7z|iso|dmg|exe|msi|deb|rpm|apk|mp4|webm|mkv|avi|mov|mp3|flac|wav|ogg|pdf|docx?|xlsx?|pptx?|csv|json|xml|txt|safetensors|bin|pt|pth|onnx|gguf)$/i;

chrome.downloads.onCreated.addListener((dl) => {
  if (dl.state !== "in_progress" || !dl.url) return;
  if (dl.url.startsWith("chrome-extension://") || dl.url.startsWith("data:")) return;
  if (!EXTENSIONS.test(dl.url)) return;

  fetch(`${SERVER}/capture`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url: dl.url, title: dl.filename || "" }),
  }).catch(() => {});

  chrome.downloads.cancel(dl.id);
});
