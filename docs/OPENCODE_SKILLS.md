# OpenCode Skills: SloughGPT Type Safety & Linting Guide

<!--
OPENCODE SYSTEM PROMPT REFERENCE:
================================
This file is the canonical knowledge base for SloughGPT development.
Before working on the codebase, OpenCode agents MUST:
1. Read relevant sections of this file
2. Reference error patterns and fixes documented here
3. Follow the patterns and conventions documented below
4. Use the quick reference commands for linting and type checking

KEY SECTIONS FOR NEW WORK:
- Common Type Errors & Fixes (errors you WILL encounter)
- Interface Definition Patterns (how to add new components)
- Ruff Linting Rules (what ruff will flag)
- Frontend Design Skills (when creating UIs)

QUICK COMMANDS:
- Type check: python3 -m pyright domains/
- Lint fix: ruff check domains/ --fix && ruff format domains/
- Test: python3 tests/test_domain_async_init.py

Location: /Users/mac/sloughGPT/docs/OPENCODE_SKILLS.md
-->

This document serves as a knowledge base for OpenCode agents working on the SloughGPT codebase. It documents common errors, type checking patterns, linting rules, and frontend design principles to prevent repeating mistakes.

## Table of Contents

1. [Quick Reference Commands](#quick-reference-commands)
2. [Common Type Errors & Fixes](#common-type-errors--fixes)
3. [Interface Definition Patterns](#interface-definition-patterns)
4. [Ruff Linting Rules](#ruff-linting-rules)
5. [Pyright Type Checking](#pyright-type-checking)
6. [Domain Architecture Patterns](#domain-architecture-patterns)
7. [Import Patterns](#import-patterns)
8. [Dataclass Patterns](#dataclass-patterns)

---

## Quick Reference Commands

```bash
# Type checking
cd /Users/mac/sloughGPT
python3 -m pyright domains/

# Linting
python3 -m ruff check domains/
python3 -m ruff format domains/

# Run tests
python3 tests/test_domain_async_init.py

# Clear pyright cache if stale
rm -rf ~/.cache/pyright
```

---

## Common Type Errors & Fixes

### Error: "No parameter named 'X'"

**Cause:** Dataclass field mismatch between interface and implementation.

**Fix:** Add the missing field to the dataclass definition.

**Example:**
```python
# Wrong - missing format and metadata
@dataclass
class UIResponse:
    request_id: str
    status: str
    data: Any
    timestamp: float

# Correct - with optional fields
@dataclass
class UIResponse:
    request_id: str
    status: str
    data: Any
    timestamp: float
    format: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
```

---

### Error: "Return type mismatch in override"

**Cause:** Method signature doesn't match parent interface.

**Fix:** Match the return type exactly.

**Example:**
```python
# Wrong
class IWebInterface(ABC):
    @abstractmethod
    async def start_server(self) -> bool:
        pass

class WebInterface(IWebInterface):
    async def start_server(self) -> None:  # Mismatch!
        pass

# Correct
class IWebInterface(ABC):
    @abstractmethod
    async def start_server(self) -> None:
        pass

class WebInterface(IWebInterface):
    async def start_server(self) -> None:
        pass
```

---

### Error: "Method 'X' overrides class 'Y' in an incompatible manner"

**Cause:** Parameter count, names, or types don't match.

**Fix:** Ensure implementation matches interface signature exactly.

**Example:**
```python
# Wrong - parameter name mismatch
class IAuthenticationService(ABC):
    @abstractmethod
    async def authenticate(self, username: str, password: str) -> Optional[str]:
        pass

class AuthenticationService(IAuthenticationService):
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Any]:  # Mismatch
        pass

# Correct
class IAuthenticationService(ABC):
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Any]:
        pass

class AuthenticationService(IAuthenticationService):
    async def authenticate(self, credentials: Dict[str, str]) -> Optional[Any]:
        pass
```

---

### Error: "Attribute 'X' is unknown"

**Cause:** Missing field in dataclass or missing method in interface.

**Fix:** Add the field/method to the appropriate class.

**Example:**
```python
# Wrong - missing consolidate_memories in interface
class IMemoryManager(ABC):
    @abstractmethod
    async def store_memory(self, memory: Any) -> str:
        pass

    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        pass

# Correct - with consolidate_memories
class IMemoryManager(ABC):
    @abstractmethod
    async def store_memory(self, memory: Any) -> str:
        pass

    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        pass

    @abstractmethod
    async def consolidate_memories(self) -> Dict[str, Any]:
        pass
```

---

### Error: "Cannot instantiate abstract class 'X'"

**Cause:** Not all abstract methods implemented.

**Fix:** Implement all `@abstractmethod` decorated methods.

**Example:**
```python
# Wrong - Missing scale method
class IDeploymentManager(ABC):
    @abstractmethod
    async def deploy(self, config: Dict[str, Any], environment: str) -> str:
        pass

    @abstractmethod
    async def scale(self, service_id: str, replicas: int) -> bool:
        pass

class DeploymentManager(IDeploymentManager):
    async def deploy(self, config, environment) -> str:  # scale() missing!
        pass

# Correct - All methods implemented
class DeploymentManager(IDeploymentManager):
    async def deploy(self, config, environment) -> str:
        pass

    async def scale(self, service_id: str, replicas: int) -> bool:
        pass
```

---

### Error: "Argument missing for parameter"

**Cause:** Wrong number of arguments passed to method.

**Fix:** Match the interface signature.

**Example:**
```python
# Wrong
user = await auth.authenticate(username, password)  # But interface expects Dict

# Correct
user = await auth.authenticate({"username": username, "password": password})
```

---

### Error: "Import '...__init__' could not be resolved"

**Cause:** Wrong relative import path or missing export.

**Fix:** Use correct relative import level.

```python
# In domains/cognitive/base.py (one level deep)
from ..__init__ import BaseDomain  # Correct - two levels up from cognitive

# In domains/cognitive/memory/__init__.py (two levels deep)
from ...__init__ import IMemoryManager  # Correct - three levels up
```

---

## Interface Definition Patterns

### Standard Interface Template

```python
class IInterfaceName(ABC):
    """Interface description"""

    @abstractmethod
    async def method_one(self, param1: Type1, param2: Type2) -> ReturnType:
        """Method description"""
        pass

    @abstractmethod
    async def method_two(self, param3: Type3) -> ReturnType2:
        """Method description"""
        pass
```

### Domain Base Class Template

```python
class BaseDomain:
    """Base class for all domains"""

    def __init__(self, domain_name: str):
        self.domain_name = domain_name
        self.is_initialized = False
        self.components: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the domain"""
        self.is_initialized = True

    async def shutdown(self) -> None:
        """Shutdown the domain"""
        self.is_initialized = False
```

---

## Ruff Linting Rules

### Common Ruff Errors

| Code | Issue | Fix |
|------|-------|-----|
| F401 | Unused import | Remove import or add to `__all__` |
| E501 | Line too long (>100) | Split line or add `# noqa: E501` |
| I001 | Import not sorted | Run `ruff format` |
| F841 | Unused variable | Remove assignment or prefix with `_` |

### Auto-fix Ruff Issues

```bash
# Auto-format and fix import sorting
ruff format domains/
ruff check domains/ --fix
```

### Suppress Line

```python
# noqa: E501 - For intentionally long lines
very_long_line = "this_is_a_very_long_string_that_exceeds_the_100_character_limit_and_is_acceptable_in_this_case"
```

---

## Pyright Type Checking

### Configuration

Pyright is configured to check the `domains/` directory. Key settings:
- Python version: 3.9+
- Type checking mode: basic
- Report missing imports: true

### Common Pyright Errors

1. **`reportCallIssue`** - Wrong argument types
2. **`reportAttributeAccessIssue`** - Missing attribute
3. **`reportIncompatibleMethodOverride`** - Interface mismatch
4. **`reportOptionalMemberAccess`** - Optional member accessed without check

### Type Suppression

```python
# For intentional dynamic typing
stats["connections_by_type"] = connections_by_type  # type: ignore

# For known safe operations
value = some_dict.get("key", default)  # pyright may complain
```

---

## Domain Architecture Patterns

### Domain Structure

```
domains/
├── __init__.py          # Exports interfaces, base classes, types
├── ui/
│   ├── __init__.py     # Exports UI components
│   ├── web/
│   ├── chat/
│   ├── api/
│   └── cli/
├── cognitive/
│   ├── __init__.py
│   ├── base.py         # CognitiveDomain
│   ├── memory/
│   ├── reasoning/
│   ├── metacognition/
│   └── processor.py
├── infrastructure/
│   ├── __init__.py
│   ├── base.py
│   ├── database/
│   ├── cache/
│   └── deployment/
└── enterprise/
    ├── __init__.py
    ├── base.py
    ├── auth/
    ├── users/
    ├── monitoring/
    └── cost/
```

### Component Pattern

```python
from ...__init__ import IInterface, BaseComponent, ComponentException

class ComponentName(BaseComponent, IInterface):
    """Component description"""

    def __init__(self) -> None:
        super().__init__("component_name")
        self.logger = logging.getLogger(f"sloughgpt.{self.component_name}")
        # Initialize state
        self.is_initialized = False

    async def initialize(self) -> None:
        """Initialize component"""
        try:
            # Setup logic
            self.is_initialized = True
        except Exception as e:
            raise ComponentException(f"Initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown component"""
        self.is_initialized = False
```

---

## Import Patterns

### Correct Import Patterns

```python
# For types from the same level
from ..__init__ import BaseDomain, DomainException

# For types from the same domain
from .component import ComponentClass

# For standard library
from typing import Dict, Any, Optional
import asyncio
import logging

# For third party
import jwt
from dataclasses import dataclass
```

### Import Organization

```python
# Standard library imports (alphabetical)
import asyncio
import logging
import time
from typing import Dict, Any, List, Optional

# Third party imports
from dataclasses import dataclass
from enum import Enum

# Local imports
from ...__init__ import BaseComponent
from .helper import HelperClass
```

---

## Dataclass Patterns

### Simple Dataclass

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ConfigClass:
    field1: str
    field2: int
    field3: bool = False  # Default value
    field4: Optional[str] = None  # Optional field
```

### With Field Order Requirements

```python
@dataclass
class UIResponse:
    request_id: str
    status: str
    data: Any
    timestamp: float
    format: Optional[Any] = None  # Optional after required
    metadata: Optional[Dict[str, Any]] = None
```

---

## Enum Patterns

### Standard Enum

```python
from enum import Enum

class DatabaseType(Enum):
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    REDIS = "redis"
```

### String Enum (for easier parsing)

```python
class SecurityLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"  # Note: "secret" is not valid
```

---

## Error Handling Pattern

```python
async def method_name(self) -> ReturnType:
    """Description"""
    try:
        # Operation that might fail
        result = await self._do_operation()
        return result
    except Exception as e:
        self.logger.error(f"Operation failed: {e}")
        raise ComponentException(f"Operation failed: {e}")
```

---

## Testing Pattern

### Async Initialization Test

```python
import asyncio
import pytest

@pytest.mark.asyncio
async def test_component_init():
    from domains.domain import Component
    
    component = Component()
    assert not component.is_initialized
    
    await component.initialize()
    assert component.is_initialized
    
    await component.shutdown()
    assert not component.is_initialized
```

---

## Checklist for New Components

When adding a new component:

- [ ] Define interface in `domains/__init__.py`
- [ ] Implement component class with proper inheritance
- [ ] Add all required interface methods
- [ ] Initialize `is_initialized` to `False`
- [ ] Implement `initialize()` and `shutdown()` methods
- [ ] Add to domain's `__init__.py` exports
- [ ] Run `ruff format` to fix imports
- [ ] Run `pyright domains/` to verify types
- [ ] Add test in `tests/test_domain_async_init.py`

---

## Notes

- **SecurityLevel**: Use `RESTRICTED`, not `SECRET` (not defined)
- **UserRole**: Use `ADMIN`, `USER`, `GUEST` (no `SERVICE`)
- **DatabaseConfig**: Use `database` field, not `connection_string`
- **ICacheManager.set()**: `ttl` parameter is `Optional[int]` with default 3600
- **IAuthenticationService**: Uses `credentials: Dict[str, str]`, not separate username/password

---

## File: docs/OPENCODE_SKILLS.md

This file should be referenced when:
- Adding new interfaces or components
- Fixing type errors
- Configuring linting
- Onboarding new developers

---

# Frontend Design Skills

This section documents frontend design principles, typography guidelines, and aesthetic patterns for creating distinctive, high-quality user interfaces.

## Table of Contents

1. [Typography Principles](#typography-principles)
2. [RPG Theme Guidelines](#rpg-theme-guidelines)
3. [Avoiding Generic AI Aesthetics](#avoiding-generic-ai-aesthetics)
4. [Font Selection Guide](#font-selection-guide)
5. [Motion & Animation](#motion--animation)
6. [Color & Theme Patterns](#color--theme-patterns)
7. [Frontend Examples](#frontend-examples)

---

## Typography Principles

### Key Rule: Typography Instantly Signals Quality

Avoid boring, generic fonts at all costs. Typography is the single most impactful design decision.

### Fonts to NEVER Use

- Inter
- Roboto
- Open Sans
- Lato
- Default system fonts
- Arial, Helvetica (unless deliberately styled)

### Recommended Font Categories

#### Code Aesthetic
- **JetBrains Mono** - Clean, programming-focused
- **Fira Code** - Excellent ligatures
- **Space Grotesk** - Geometric with character

#### Editorial/Elegant
- **Playfair Display** - High-contrast serif
- **Crimson Pro** - Classical, readable
- **Fraunces** - Vintage personality

#### Technical/Documentation
- **IBM Plex family** - Professional, precise
- **Source Sans 3** - Editorial yet technical
- **DM Sans** - Geometric and modern

#### Distinctive/Display
- **Bricolage Grotesque** - Unique, expressive
- **Newsreader** - Editorial serif
- **Syne** - Avant-garde character

### Pairing Principle

Create contrast through pairing:
- **Display + Monospace** - Titles in display font, code in mono
- **Serif + Geometric Sans** - Classic meets modern
- **Variable font across weights** - Use extremes, not middling

### Weight Extremes Rule

Use extremes, not middling values:
- **Bad:** 400 weight vs 600 weight (too similar)
- **Good:** 100/200 weight vs 800/900 weight (dramatic contrast)
- **Size jumps:** 3x+ difference, not 1.5x

### Font Loading

Always load from Google Fonts or proper font sources:

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">
```

```css
:root {
  --font-display: 'Playfair Display', serif;
  --font-mono: 'JetBrains Mono', monospace;
  --font-body: 'IBM Plex Sans', sans-serif;
}
```

---

## RPG Theme Guidelines

### Always Design with RPG Aesthetic

When appropriate for the context (games, dashboards, admin panels):

- **Fantasy-inspired color palettes** with rich, dramatic tones
- **Ornate borders** and decorative frame elements
- **Parchment textures**, leather-bound styling, weathered materials
- **Epic, adventurous atmosphere** with dramatic lighting
- **Medieval-inspired serif typography** with embellished headers

### RPG Theme CSS Variables

```css
:root {
  /* RPG Color Palette - Deep, Dramatic */
  --rpg-bg-dark: #0d0a0f;
  --rpg-bg-panel: #1a141c;
  --rpg-border: #3d2d42;
  --rpg-accent-gold: #c9a227;
  --rpg-accent-crimson: #8a1c1c;
  --rpg-text-primary: #e8e1d5;
  --rpg-text-muted: #8a7f75;
  --rpg-glow: rgba(201, 162, 39, 0.3);
  
  /* Typography */
  --rpg-font-header: 'Crimson Pro', serif;
  --rpg-font-body: 'Source Sans 3', sans-serif;
  --rpg-font-mono: 'JetBrains Mono', monospace;
  
  /* Effects */
  --rpg-border-ornate: 2px solid var(--rpg-border);
  --rpg-shadow-glow: 0 0 20px var(--rpg-glow);
}
```

### RPG Panel Component

```css
.rpg-panel {
  background: linear-gradient(145deg, var(--rpg-bg-panel), var(--rpg-bg-dark));
  border: var(--rpg-border-ornate);
  border-radius: 4px;
  padding: 1.5rem;
  box-shadow: 
    inset 0 0 40px rgba(0, 0, 0, 0.5),
    var(--rpg-shadow-glow);
  position: relative;
}

.rpg-panel::before {
  content: '';
  position: absolute;
  top: 4px;
  left: 4px;
  right: 4px;
  bottom: 4px;
  border: 1px solid var(--rpg-border);
  pointer-events: none;
}
```

---

## Avoiding Generic AI Aesthetics

### The "AI Slop" Problem

You tend to converge toward generic, "on distribution" outputs. This creates predictable, boring designs.

### What to Avoid

- **Overused font families:** Inter, Roboto, Arial, system fonts
- **Clichéd color schemes:** Purple gradients on white backgrounds
- **Predictable layouts:** Same grid, same spacing, same patterns
- **Cookie-cutter design:** Lacks context-specific character

### How to Stand Out

1. **Commit to a cohesive aesthetic** - Don't be timid
2. **Use dominant colors with sharp accents** - Not evenly distributed palettes
3. **Draw from IDE themes** - Many excellent color systems exist
4. **Draw from cultural aesthetics** - Cyberpunk, fantasy, brutalism, etc.

### Contrast Principle

```css
/* Timid (AVOID) */
.button {
  background: #6366f1;
  color: white;
  padding: 0.75rem 1.5rem;
}

/* Bold (GOOD) */
.button {
  background: linear-gradient(135deg, var(--accent-crimson), var(--accent-crimson-dark));
  color: var(--text-gold);
  padding: 1rem 2.5rem;
  font-family: var(--font-display);
  text-transform: uppercase;
  letter-spacing: 0.15em;
  border: 2px solid var(--accent-gold);
  box-shadow: var(--shadow-glow);
}
```

---

## Font Selection Guide

### Decision Tree

```
Is this a code/technical display?
├── YES → JetBrains Mono, Fira Code, IBM Plex Mono
└── NO → Is this an elegant/editorial context?
    ├── YES → Playfair Display, Crimson Pro, Newsreader
    └── NO → Is this a game/RPG context?
        ├── YES → MedievalSharp, Uncial Antiqua, Cinzel
        └── NO → Is this a modern/tech context?
            ├── YES → Space Grotesk, IBM Plex Sans, Bricolage Grotesque
            └── NO → Source Sans 3, DM Sans
```

### Google Fonts Quick Reference

```html
<!-- Elegant Editorial -->
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=Crimson+Pro:wght@400;600&display=swap" rel="stylesheet">

<!-- Technical Modern -->
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=JetBrains+Mono:wght@400&display=swap" rel="stylesheet">

<!-- RPG/Fantasy -->
<link href="https://fonts.googleapis.com/css2?family=Cinzel:wght@400;700&family=Crimson+Pro:wght@400;600&display=swap" rel="stylesheet">

<!-- Bold Tech -->
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesk:opsz@12..96&family=IBM+Plex+Mono:wght@400&display=swap" rel="stylesheet">
```

### Font Stack Recommendations

```css
/* Display - Make it special */
--font-display: 'Playfair Display', 'Crimson Pro', Georgia, serif;

/* Body - Readable but distinctive */
--font-body: 'Source Sans 3', 'IBM Plex Sans', system-ui, sans-serif;

/* Monospace - Code and data */
--font-mono: 'JetBrains Mono', 'Fira Code', 'IBM Plex Mono', monospace;
```

---

## Motion & Animation

### Prioritize CSS-Only for HTML

```css
/* Page load reveal */
@keyframes reveal {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.page-content {
  animation: reveal 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

/* Staggered children */
.card:nth-child(1) { animation-delay: 0.1s; }
.card:nth-child(2) { animation-delay: 0.2s; }
.card:nth-child(3) { animation-delay: 0.3s; }
.card:nth-child(4) { animation-delay: 0.4s; }
```

### Motion Principles

1. **One well-orchestrated page load** > scattered micro-interactions
2. **Staggered reveals** (animation-delay) create delight
3. **Cubic-bezier easing** (0.16, 1, 0.3, 1) feels natural
4. **Avoid linear animations** - they feel robotic

### High-Impact Moments

```css
/* Hero section dramatic reveal */
.hero-title {
  font-family: var(--font-display);
  font-size: clamp(2.5rem, 8vw, 6rem);
  font-weight: 900;
  line-height: 0.95;
  letter-spacing: -0.03em;
  animation: heroReveal 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
  opacity: 0;
  transform: scale(0.95);
}

@keyframes heroReveal {
  to {
    opacity: 1;
    transform: scale(1);
  }
}

/* Interactive hover states */
.button:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 30px var(--glow-color);
  transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
}
```

---

## Color & Theme Patterns

### CSS Variables for Consistency

```css
:root {
  /* Primary brand colors - commit boldly */
  --color-bg-primary: #0a0a0f;
  --color-bg-secondary: #14141f;
  --color-text-primary: #f0f0f5;
  --color-text-muted: #8888aa;
  
  /* Accent - sharp and distinctive */
  --color-accent: #00d4aa;
  --color-accent-glow: rgba(0, 212, 170, 0.4);
  
  /* Status colors - meaningful contrast */
  --color-success: #00ff88;
  --color-warning: #ffaa00;
  --color-error: #ff4466;
  --color-info: #66aaff;
  
  /* Semantic */
  --color-border: #2a2a3a;
  --color-surface: #1a1a2a;
}
```

### Dark Theme Best Practices

```css
/* Never pure black backgrounds */
.dark-theme {
  background: linear-gradient(
    180deg,
    var(--color-bg-primary) 0%,
    var(--color-bg-secondary) 100%
  );
}

/* Subtle depth through layering */
.card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  box-shadow: 
    0 4px 20px rgba(0, 0, 0, 0.3),
    inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
```

---

## Frontend Examples

### Bold Dashboard Header

```html
<style>
.dashboard-header {
  font-family: 'Space Grotesk', sans-serif;
  background: linear-gradient(
    135deg,
    #1a1a2e 0%,
    #16213e 50%,
    #0f3460 100%
  );
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  padding: 2rem 3rem;
}

.dashboard-title {
  font-size: clamp(1.8rem, 5vw, 3.5rem);
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(
    135deg,
    #00d4aa 0%,
    #00ff88 50%,
    #aaffff 100%
  );
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
</style>

<header class="dashboard-header">
  <h1 class="dashboard-title">SloughGPT Admin</h1>
</header>
```

### RPG Stats Panel

```html
<style>
.stat-panel {
  font-family: 'Cinzel', serif;
  background: linear-gradient(
    145deg,
    #1a141c 0%,
    #0d0a0f 100%
  );
  border: 2px solid #3d2d42;
  border-radius: 4px;
  padding: 1.5rem;
  position: relative;
  overflow: hidden;
}

.stat-panel::before {
  content: '⚔';
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  font-size: 1.5rem;
  opacity: 0.3;
}

.stat-value {
  font-size: 2.5rem;
  font-weight: 700;
  color: #c9a227;
  text-shadow: 0 0 20px rgba(201, 162, 39, 0.4);
}
</style>

<div class="stat-panel">
  <div class="stat-label">Experience</div>
  <div class="stat-value">47,892</div>
</div>
```

### Elegant Documentation

```html
<style>
.docs-page {
  font-family: 'IBM Plex Serif', serif;
  background: #faf9f7;
  color: #1a1a1a;
}

.docs-title {
  font-family: 'Playfair Display', serif;
  font-size: 3rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #1a1a1a;
  margin-bottom: 1rem;
}

.docs-prose {
  font-family: 'IBM Plex Serif', serif;
  font-size: 1.125rem;
  line-height: 1.8;
  color: #4a4a4a;
}

.docs-prose p {
  margin-bottom: 1.5rem;
}
</style>
```

---

## Quick Reference: Font Choices by Context

| Context | Display Font | Body Font | Mono Font |
|---------|-------------|-----------|-----------|
| **Admin Dashboard** | Space Grotesk | Source Sans 3 | JetBrains Mono |
| **RPG/Game UI** | Cinzel | Crimson Pro | JetBrains Mono |
| **Documentation** | Playfair Display | IBM Plex Serif | IBM Plex Mono |
| **Technical Blog** | Bricolage Grotesque | DM Sans | Fira Code |
| **E-commerce** | Fraunces | Source Sans 3 | JetBrains Mono |
| **Portfolio** | Syne | Inter (styled) | Space Mono |

---

## Reminder: Think Outside the Box

You tend to converge on common choices (Space Grotesk, for example) across generations. **Avoid this.**

Each project deserves:
- **Unique font pairing** that fits the context
- **Cohesive color palette** with dramatic accents
- **Purposeful motion** that enhances, not distracts
- **Distinctive character** that feels designed, not generated

When in doubt, reference:
- Excellent IDE themes (Dracula, Nord, Catppuccin, Tokyo Night)
- Game UIs (Elden Ring, Baldur's Gate 3, Disco Elysium)
- Editorial design (Monocle, New York Times Magazine)
- Brutalist web design (awwwards.com winners)

---

<!--
================================================================================
OPENCODE ALWAYS REFERENCE THIS FILE
================================================================================

Before any coding task on SloughGPT:
1. Read OPENCODE SYSTEM PROMPT REFERENCE at the top of this file
2. Check QUICK COMMANDS for lint/typecheck
3. Reference COMMON TYPE ERRORS when encountering pyright/ruff errors
4. Follow INTERFACE DEFINITION PATTERNS when adding new components
5. Use FRONTEND DESIGN SKILLS when creating any UI

File Location: /Users/mac/sloughGPT/docs/OPENCODE_SKILLS.md

This file is the single source of truth for:
- Type error patterns and fixes
- Interface design conventions
- Linting rules
- Frontend aesthetics and typography
- Component patterns

Last Updated: 2026-02-12
================================================================================
-->
