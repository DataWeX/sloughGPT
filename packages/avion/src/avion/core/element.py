"""Element identification and interaction.

Provides a unified Element wrapper and ElementFinder that works across
backends. Supports multiple identification strategies: CSS selectors,
text content, role/ARIA, test IDs, and accessibility tree queries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class SelectorStrategy(Enum):
    """How to locate an element."""

    CSS = "css"
    TEXT = "text"
    ROLE = "role"
    TEST_ID = "test_id"
    LABEL = "label"
    PLACEHOLDER = "placeholder"
    ARIA = "aria"
    XPATH = "xpath"
    NTH = "nth"  # index-based within a set


@dataclass(frozen=True)
class Selector:
    """A single identification strategy."""

    strategy: SelectorStrategy
    value: str
    exact: bool = False

    def describe(self) -> str:
        if self.strategy == SelectorStrategy.CSS:
            return f"css={self.value}"
        if self.strategy == SelectorStrategy.TEXT:
            mode = "exact" if self.exact else "contains"
            return f"text({mode})={self.value!r}"
        if self.strategy == SelectorStrategy.ROLE:
            return f"role={self.value}"
        if self.strategy == SelectorStrategy.TEST_ID:
            return f"testId={self.value!r}"
        if self.strategy == SelectorStrategy.LABEL:
            return f"label={self.value!r}"
        if self.strategy == SelectorStrategy.PLACEHOLDER:
            return f"placeholder={self.value!r}"
        if self.strategy == SelectorStrategy.ARIA:
            return f"aria={self.value!r}"
        if self.strategy == SelectorStrategy.XPATH:
            return f"xpath={self.value}"
        if self.strategy == SelectorStrategy.NTH:
            return f"nth={self.value}"
        return f"{self.strategy.value}={self.value!r}"


@dataclass(frozen=True)
class ElementLocator:
    """Chain of selectors to locate an element (first match wins)."""

    selectors: tuple[Selector, ...]
    description: str = ""

    @classmethod
    def css(cls, selector: str, description: str = "") -> ElementLocator:
        return cls(selectors=(Selector(SelectorStrategy.CSS, selector),), description=description)

    @classmethod
    def text(cls, text: str, exact: bool = False, description: str = "") -> ElementLocator:
        return cls(
            selectors=(Selector(SelectorStrategy.TEXT, text, exact=exact),),
            description=description,
        )

    @classmethod
    def role(cls, role: str, name: str = "", description: str = "") -> ElementLocator:
        value = f"{role}[{name}]" if name else role
        return cls(selectors=(Selector(SelectorStrategy.ROLE, value),), description=description)

    @classmethod
    def test_id(cls, test_id: str, description: str = "") -> ElementLocator:
        return cls(
            selectors=(Selector(SelectorStrategy.TEST_ID, test_id),),
            description=description,
        )

    @classmethod
    def label(cls, label: str, description: str = "") -> ElementLocator:
        return cls(
            selectors=(Selector(SelectorStrategy.LABEL, label),),
            description=description,
        )

    @classmethod
    def chain(cls, *locators: ElementLocator, description: str = "") -> ElementLocator:
        """Chain multiple locators — tries each in order."""
        all_selectors = []
        for loc in locators:
            all_selectors.extend(loc.selectors)
        return cls(selectors=tuple(all_selectors), description=description)

    def describe(self) -> str:
        if self.description:
            return self.description
        return " | ".join(s.describe() for s in self.selectors)


@dataclass
class Element:
    """Wrapper around a raw backend element with interaction methods."""

    raw: Any
    locator: ElementLocator
    backend: str = "unknown"
    _cache: dict[str, Any] = field(default_factory=dict)

    @property
    def tag(self) -> str:
        return self._cache.get("tag", "")

    @property
    def text(self) -> str:
        return self._cache.get("text", "")

    @property
    def inner_html(self) -> str:
        return self._cache.get("inner_html", "")

    @property
    def is_visible(self) -> bool:
        return self._cache.get("is_visible", True)

    @property
    def is_enabled(self) -> bool:
        return self._cache.get("is_enabled", True)

    @property
    def bounding_box(self) -> dict[str, float] | None:
        return self._cache.get("bounding_box")

    @property
    def attributes(self) -> dict[str, str]:
        return self._cache.get("attributes", {})

    @property
    def value(self) -> Any:
        return self._cache.get("value")

    def get_attribute(self, name: str) -> str | None:
        return self.attributes.get(name)

    def describe(self) -> str:
        loc = self.locator.describe()
        return f"Element({loc}, tag={self.tag!r}, text={self.text!r})"

    def __repr__(self) -> str:
        return self.describe()


@runtime_checkable
class Backend(Protocol):
    """Abstract interface for browser/interaction backends."""

    @property
    def name(self) -> str: ...

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def navigate(self, url: str) -> None: ...

    async def find_element(self, locator: ElementLocator) -> Element | None: ...

    async def find_elements(self, locator: ElementLocator) -> list[Element]: ...

    async def click(self, element: Element) -> None: ...

    async def fill(self, element: Element, value: str) -> None: ...

    async def select_option(self, element: Element, value: str) -> None: ...

    async def check(self, element: Element) -> None: ...

    async def uncheck(self, element: Element) -> None: ...

    async def hover(self, element: Element) -> None: ...

    async def screenshot(self, path: str | None = None) -> bytes: ...

    async def get_url(self) -> str: ...

    async def get_title(self) -> str: ...

    async def wait_for(self, locator: ElementLocator, timeout: float = 10.0) -> Element: ...

    async def wait_for_timeout(self, ms: float) -> None: ...

    async def evaluate(self, expression: str) -> Any: ...

    async def get_accessibility_tree(self) -> dict[str, Any]: ...


class ElementFinder:
    """High-level element finder with retry and fallback logic.

    Usage::

        finder = ElementFinder(backend)
        el = await finder.find(ElementLocator.css(".my-button"))
        el = await finder.find(ElementLocator.text("Submit"))
    """

    def __init__(self, backend: Backend, default_timeout: float = 10.0):
        self._backend = backend
        self._default_timeout = default_timeout

    async def find(self, locator: ElementLocator, timeout: float | None = None) -> Element:
        """Find a single element. Raises if not found."""
        timeout = timeout or self._default_timeout
        el = await self._backend.wait_for(locator, timeout=timeout)
        if el is None:
            raise ElementNotFoundError(f"Element not found: {locator.describe()}")
        return el

    async def find_all(self, locator: ElementLocator) -> list[Element]:
        """Find all matching elements (may be empty)."""
        return await self._backend.find_elements(locator)

    async def find_optional(self, locator: ElementLocator, timeout: float = 3.0) -> Element | None:
        """Find element, return None if not found."""
        try:
            return await self._backend.wait_for(locator, timeout=timeout)
        except Exception:
            return None

    async def find_within(
        self, parent: Element, locator: ElementLocator, timeout: float | None = None
    ) -> Element:
        """Find element within a parent element's subtree.

        Default implementation delegates to find_all and checks ancestry.
        Backends can override for more efficient scoped queries.
        """
        results = await self.find_all(locator)
        if not results:
            raise ElementNotFoundError(
                f"Element not found within {parent.describe()}: {locator.describe()}"
            )
        return results[0]

    async def count(self, locator: ElementLocator) -> int:
        """Count matching elements."""
        elements = await self.find_all(locator)
        return len(elements)

    async def is_visible(self, locator: ElementLocator, timeout: float = 3.0) -> bool:
        """Check if element is visible."""
        el = await self.find_optional(locator, timeout=timeout)
        return el is not None and el.is_visible

    async def get_text(self, locator: ElementLocator, timeout: float | None = None) -> str:
        """Get text content of element."""
        el = await self.find(locator, timeout=timeout)
        return el.text

    async def get_attribute(
        self, locator: ElementLocator, attr: str, timeout: float | None = None
    ) -> str | None:
        """Get attribute value of element."""
        el = await self.find(locator, timeout=timeout)
        return el.get_attribute(attr)


class ElementNotFoundError(Exception):
    """Raised when an element cannot be found."""
