"""Abstract backend interface.

Defines the Backend protocol that all browser automation adapters must implement.
"""

from avion.core.element import Backend  # re-export

__all__ = ["Backend"]
