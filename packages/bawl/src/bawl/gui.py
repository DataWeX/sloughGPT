"""
gui — tkinter GUI for bawl. Launch with `bawl gui`.

Tabs: Page, Crawl, Sitemap, Settings.
"""

import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from typing import Optional

from .parse import parse, Page
from .crawl import crawl as _crawl
from .sitemap import parse as parse_sitemap
from .store import save as _save


class App:
    """Main bawl GUI window."""

    RATE = 0.5
    TIMEOUT = 15

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("bawl")
        self.root.geometry("850x650")
        self.root.minsize(600, 400)
        self._latest_pages: list[Page] = []
        self._build_ui()
        self.root.bind("<Return>", self._on_enter)

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self._page_tab(nb)
        self._crawl_tab(nb)
        self._sitemap_tab(nb)
        self._settings_tab(nb)
        self._status_var = tk.StringVar(value="Ready")
        bar = ttk.Label(self.root, textvariable=self._status_var,
                        relief="sunken", anchor="w", padding=(6, 2))
        bar.pack(fill="x", padx=6, pady=(0, 6))

    # ---- helpers ----

    def _status(self, msg: str) -> None:
        self._status_var.set(msg)
        self.root.update_idletasks()

    def _url_row(self, parent, btn_text, cmd):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 4))
        v = tk.StringVar()
        e = ttk.Entry(frame, textvariable=v)
        e.pack(side="left", fill="x", expand=True, padx=(0, 4))
        b = ttk.Button(frame, text=btn_text, command=cmd)
        b.pack(side="right")
        return v, e, b

    def _save_button(self, parent, cmd):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=(0, 4))
        ttk.Button(f, text="Save to file", command=cmd).pack(side="right")

    def _text_area(self, parent):
        t = scrolledtext.ScrolledText(parent, wrap="word",
                                       font=("Menlo", 11), relief="flat")
        t.pack(fill="both", expand=True, pady=(0, 0))
        return t

    def _progress(self, parent):
        pb = ttk.Progressbar(parent, mode="indeterminate")
        pb.pack(fill="x", pady=(2, 4))
        pb.pack_forget()
        return pb

    def _on_enter(self, event=None):
        nb = self.root.children.get("!notebook")
        if not nb:
            return
        sel = nb.index("current")
        if sel == 0:
            self._page_go()
        elif sel == 1:
            self._crawl_go()
        elif sel == 2:
            self._sitemap_go()
        elif sel == 3:
            pass

    # ---- Settings tab ----

    def _settings_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Settings  ")
        f = ttk.Frame(tab)
        f.pack(fill="x", padx=8, pady=12)

        ttk.Label(f, text="Rate limit (seconds):").grid(row=0, column=0, sticky="w", pady=4)
        rv = tk.DoubleVar(value=self.RATE)
        rs = ttk.Spinbox(f, from_=0, to=5, increment=0.1, textvariable=rv, width=6)
        rs.grid(row=0, column=1, sticky="w", padx=8)
        self._rate_var = rv

        ttk.Label(f, text="Request timeout (seconds):").grid(row=1, column=0, sticky="w", pady=4)
        tv = tk.IntVar(value=self.TIMEOUT)
        ts = ttk.Spinbox(f, from_=1, to=60, textvariable=tv, width=6)
        ts.grid(row=1, column=1, sticky="w", padx=8)
        self._timeout_var = tv

        ttk.Separator(tab, orient="horizontal").pack(fill="x", padx=8, pady=8)
        ttk.Label(tab, text="About", font=("", 11, "bold")).pack(anchor="w", padx=8)
        ttk.Label(tab, text="bawl v0.1 — zero-dependency web crawler. "
                  "Uses stdlib only (urllib, html.parser, tkinter).",
                  wraplength=600).pack(anchor="w", padx=8, pady=4)

    def _rate(self):
        try:
            return float(self._rate_var.get())
        except Exception:
            return self.RATE

    def _timeout(self):
        try:
            return int(self._timeout_var.get())
        except Exception:
            return self.TIMEOUT

    # ---- Page tab ----

    def _page_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Page  ")
        self._page_url_v, _, self._page_btn = self._url_row(tab, "Fetch", self._page_go)
        self._page_save_btn = ttk.Button(tab, text="Save to file",
                                          command=self._page_save, state="disabled")
        self._page_save_btn.pack(anchor="e", pady=(0, 2))
        self._page_txt = self._text_area(tab)
        self._page_pb = self._progress(tab)

    def _page_save(self):
        if not self._latest_pages:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jsonl",
                                             filetypes=[("JSONL", "*.jsonl"), ("All", "*")])
        if not path:
            return
        for p in self._latest_pages:
            _save(p, path=path)
        self._status(f"Saved {len(self._latest_pages)} page(s) to {path}")

    def _page_go(self):
        url = self._page_url_v.get().strip()
        if not url:
            return
        self._page_txt.delete("1.0", "end")
        self._page_btn.config(state="disabled")
        self._page_save_btn.config(state="disabled")
        self._page_pb.pack(fill="x", pady=(2, 4))
        self._page_pb.start(10)
        self._status(f"Fetching {url} ...")
        t = threading.Thread(target=self._page_worker, args=(url,), daemon=True)
        t.start()

    def _page_worker(self, url: str):
        page = parse(url, timeout=self._timeout(), rate=self._rate())
        self.root.after(0, self._page_done, page, url)

    def _page_done(self, page: Optional[Page], url: str):
        self._page_pb.stop()
        self._page_pb.pack_forget()
        self._page_btn.config(state="normal")
        t = self._page_txt
        t.delete("1.0", "end")
        if page is None:
            t.insert("end", f"Failed to fetch: {url}\n")
            self._status("Failed")
            return
        self._latest_pages = [page]
        self._page_save_btn.config(state="normal")
        t.insert("end", f"# {page.title}\n\n", "title")
        t.insert("end", f"URL: {page.url}\n\n")
        t.insert("end", f"{page.text}\n\n")
        if page.links:
            t.insert("end", f"── Links ({len(page.links)}) ──\n")
            for l in page.links:
                t.insert("end", f"  • {l['text'] or l['href']}\n    {l['href']}\n")
        if page.tables:
            t.insert("end", f"\n── Tables ({len(page.tables)}) ──\n")
        if page.lists:
            t.insert("end", f"\n── Lists ({len(page.lists)}) ──\n")
        if page.code:
            t.insert("end", f"\n── Code ({len(page.code)}) ──\n")
        t.tag_config("title", font=("Menlo", 11, "bold"))
        self._status(f"{page.url} — {len(page.text)} chars, {len(page.links)} links")

    # ---- Crawl tab ----

    def _crawl_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Crawl  ")
        self._crawl_url_v, _, self._crawl_btn = self._url_row(tab, "Crawl", self._crawl_go)

        ctrl = ttk.Frame(tab)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Label(ctrl, text="Depth:").pack(side="left")
        dv = tk.IntVar(value=1)
        ttk.Spinbox(ctrl, from_=0, to=10, textvariable=dv, width=4).pack(side="left", padx=2)
        self._crawl_depth_v = dv
        ttk.Label(ctrl, text="Max:").pack(side="left", padx=(12, 0))
        mv = tk.IntVar(value=50)
        ttk.Spinbox(ctrl, from_=1, to=1000, textvariable=mv, width=5).pack(side="left", padx=2)
        self._crawl_max_v = mv

        self._crawl_save_btn = ttk.Button(tab, text="Save to file",
                                           command=self._crawl_save, state="disabled")
        self._crawl_save_btn.pack(anchor="e", pady=(0, 2))
        self._crawl_txt = self._text_area(tab)
        self._crawl_pb = self._progress(tab)

    def _crawl_save(self):
        if not self._latest_pages:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jsonl",
                                             filetypes=[("JSONL", "*.jsonl"), ("All", "*")])
        if not path:
            return
        for p in self._latest_pages:
            _save(p, path=path)
        self._status(f"Saved {len(self._latest_pages)} page(s) to {path}")

    def _crawl_go(self):
        url = self._crawl_url_v.get().strip()
        if not url:
            return
        depth = self._crawl_depth_v.get()
        mp = self._crawl_max_v.get()
        self._crawl_txt.delete("1.0", "end")
        self._crawl_btn.config(state="disabled")
        self._crawl_save_btn.config(state="disabled")
        self._crawl_pb.pack(fill="x", pady=(2, 4))
        self._crawl_pb.start(10)
        self._status(f"Crawling {url} depth={depth} max={mp} ...")
        t = threading.Thread(target=self._crawl_worker,
                             args=(url, depth, mp), daemon=True)
        t.start()

    def _crawl_worker(self, url, depth, mp):
        pages = _crawl(url, depth=depth, max_pages=mp,
                       rate=self._rate(), timeout=self._timeout(),
                       same_domain=True)
        self.root.after(0, self._crawl_done, pages)

    def _crawl_done(self, pages):
        self._crawl_pb.stop()
        self._crawl_pb.pack_forget()
        self._crawl_btn.config(state="normal")
        t = self._crawl_txt
        t.delete("1.0", "end")
        if not pages:
            t.insert("end", "No pages found.\n")
            self._status("No pages crawled")
            return
        self._latest_pages = pages
        self._crawl_save_btn.config(state="normal")
        chars = sum(len(p.text) for p in pages)
        links = sum(len(p.links) for p in pages)
        t.insert("end", f"Crawled {len(pages)} pages ({chars} chars, {links} links)\n\n", "h")
        for i, p in enumerate(pages, 1):
            t.insert("end", f"{i:3d}. {p.title or '(no title)'}\n")
            t.insert("end", f"     {p.url}\n")
            lc = len(p.links)
            t.insert("end", f"     {len(p.text)} chars, {lc} link{'s' if lc != 1 else ''}\n\n")
        t.tag_config("h", font=("", 11, "bold"))
        self._status(f"Crawl done — {len(pages)} pages, {chars} chars")

    # ---- Sitemap tab ----

    def _sitemap_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text="  Sitemap  ")
        self._sitemap_url_v, _, self._sitemap_btn = self._url_row(tab, "Fetch", self._sitemap_go)
        ctrl = ttk.Frame(tab)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Label(ctrl, text="Crawl first").pack(side="left")
        nv = tk.IntVar(value=10)
        ttk.Spinbox(ctrl, from_=0, to=500, textvariable=nv, width=5).pack(side="left", padx=2)
        ttk.Label(ctrl, text="URLs").pack(side="left", padx=(2, 8))
        self._sitemap_n_v = nv
        self._sitemap_crawl_btn = ttk.Button(ctrl, text="Crawl from sitemap",
                                              command=self._sitemap_crawl, state="disabled")
        self._sitemap_crawl_btn.pack(side="left")
        self._sitemap_save_btn = ttk.Button(tab, text="Save to file",
                                             command=self._sitemap_save, state="disabled")
        self._sitemap_save_btn.pack(anchor="e", pady=(0, 2))
        self._sitemap_txt = self._text_area(tab)
        self._sitemap_pb = self._progress(tab)

    def _sitemap_save(self):
        if not self._latest_pages:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jsonl",
                                             filetypes=[("JSONL", "*.jsonl"), ("All", "*")])
        if not path:
            return
        for p in self._latest_pages:
            _save(p, path=path)
        self._status(f"Saved {len(self._latest_pages)} page(s) to {path}")

    def _sitemap_go(self):
        url = self._sitemap_url_v.get().strip()
        if not url:
            return
        self._sitemap_txt.delete("1.0", "end")
        self._sitemap_btn.config(state="disabled")
        self._sitemap_crawl_btn.config(state="disabled")
        self._sitemap_save_btn.config(state="disabled")
        self._sitemap_pb.pack(fill="x", pady=(2, 4))
        self._sitemap_pb.start(10)
        self._status(f"Fetching sitemap {url} ...")
        t = threading.Thread(target=self._sitemap_worker, args=(url,), daemon=True)
        t.start()

    def _sitemap_worker(self, url):
        urls = parse_sitemap(url, timeout=self._timeout())
        self._sitemap_urls = urls
        self.root.after(0, self._sitemap_done, url)

    def _sitemap_done(self, url):
        self._sitemap_pb.stop()
        self._sitemap_pb.pack_forget()
        self._sitemap_btn.config(state="normal")
        t = self._sitemap_txt
        t.delete("1.0", "end")
        urls = getattr(self, "_sitemap_urls", [])
        if not urls:
            t.insert("end", "No URLs found in sitemap.\n")
            self._status("Sitemap empty or unreachable")
            return
        t.insert("end", f"Sitemap has {len(urls)} URL{'s' if len(urls) != 1 else ''}\n\n", "h")
        for u in urls[:50]:
            t.insert("end", f"  {u}\n")
        if len(urls) > 50:
            t.insert("end", f"\n... and {len(urls) - 50} more\n")
        self._sitemap_crawl_btn.config(state="normal" if urls else "disabled")
        self._status(f"{len(urls)} URLs found")
        t.tag_config("h", font=("", 11, "bold"))

    def _sitemap_crawl(self):
        urls = getattr(self, "_sitemap_urls", [])
        n = self._sitemap_n_v.get()
        if not urls or n <= 0:
            return
        targets = urls[:n]
        self._sitemap_txt.delete("1.0", "end")
        self._sitemap_btn.config(state="disabled")
        self._sitemap_crawl_btn.config(state="disabled")
        self._sitemap_save_btn.config(state="disabled")
        self._sitemap_pb.pack(fill="x", pady=(2, 4))
        self._sitemap_pb.start(10)
        self._status(f"Crawling {len(targets)} URLs from sitemap ...")
        t = threading.Thread(target=self._sitemap_crawl_worker,
                             args=(targets,), daemon=True)
        t.start()

    def _sitemap_crawl_worker(self, targets):
        pages = []
        for url in targets:
            page = parse(url, timeout=self._timeout(), rate=self._rate())
            if page:
                pages.append(page)
        self.root.after(0, self._crawl_done, pages)  # reuse crawl display
