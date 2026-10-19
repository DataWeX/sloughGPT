"""challenge — the yield: automation stops where a human must take over.

A bot-challenge page is the one place an agent must NOT keep driving:
clicking through a CAPTCHA or a Cloudflare interstitial trains the
site's bot score against the profile and burns it. The policy is one
rule — detected challenge ⇒ halt the run, keep the browser exactly
where it is (profile intact), report the evidence, hand the wheel to
the human.

Detection is pure and text-first: callers pass what the page already
says (title, URL, visible text). No backend, no network, stdlib only.
The optional OCR path in ``avion.vision`` may later refine a
detection — it never gates one: a missing OCR dependency must not turn
a yield into a blind drive.

False positives are this module's real cost, so markers are anchored
on phrases a *challenge* page shows a human ("verify you are human",
"i'm not a robot"), never on vendor names that appear in the scripts
of healthy pages — "recaptcha" alone matches nothing here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Challenge:
    """A challenge page, with the evidence that named it."""

    kind: str  # cloudflare | captcha | slider | login_wall
    evidence: str  # the phrase/shape that fired, quoted
    url: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "evidence": self.evidence, "url": self.url}


# (kind, phrase) in priority order — first match wins, so the most
# specific family (cloudflare's interstitial before generic captcha
# wording) is what callers see.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("cloudflare", "just a moment"),
    ("cloudflare", "checking your browser"),
    ("cloudflare", "performing security verification"),
    ("cloudflare", "one moment please"),
    ("captcha", "verify you are human"),
    ("captcha", "confirm you are not a robot"),
    ("captcha", "i'm not a robot"),
    ("captcha", "solve the captcha"),
    ("captcha", "select all squares"),
    ("slider", "slide to verify"),
    ("slider", "drag the slider"),
)


def detect_challenge(
    *, title: str = "", url: str = "", text: str = ""
) -> Challenge | None:
    """First challenge signal in the page's words, or None on a healthy page.

    ``text`` is whatever the caller already holds — body innerText, an
    accessibility-tree dump, OCR output. Matching is case-insensitive
    substring. The URL adds one structural signal: X's login-flow page
    (``/i/flow/login``) means the profile's cookies died mid-flow — the
    flow cannot mean anything until a human signs in again.
    """
    haystack = f"{title}\n{text}".lower()
    for kind, phrase in _MARKERS:
        if phrase in haystack:
            return Challenge(kind=kind, evidence=f"{phrase!r} shown on page", url=url)
    if "/i/flow/login" in (url or "").lower():
        return Challenge(
            kind="login_wall",
            evidence="landed on X's login flow (profile cookies are dead)",
            url=url,
        )
    return None


async def page_challenge(backend, *, a11y_text: str = "") -> Challenge | None:
    """Challenge signals on the live page — best-effort, never fatal.

    Reads title/URL/body text through the backend. Any read failure
    skips the check (returns None): a checker that cannot see must
    never fail or fake the run it protects. ``a11y_text`` — a page-text
    dump the caller already holds — substitutes for the body read.
    """
    if backend is None:
        return None
    try:
        url = str(await backend.get_url() or "")
        title = str(await backend.evaluate("document.title") or "")
        text = a11y_text or str(
            await backend.evaluate("(document.body && document.body.innerText) || ''")
            or ""
        )
    except Exception:
        return None
    return detect_challenge(title=title, url=url, text=text)
