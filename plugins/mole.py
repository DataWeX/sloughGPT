"""
Mole plugin — the read-only site checker, managed through the plugin framework.

Formerly a hardwired page (``/mole`` + a permanent sidebar nav entry). The
checks themselves still live in ``domain.core`` (``run_mole``) and the report
is still served by ``routers/mole.py``; what moved is the REGISTRATION: this
plugin is the lifecycle handle that makes Mole an optional capability instead
of core surface area. It appears in ``GET /plugins`` (loaded from the repo
``plugins/`` dir by ``POST /plugins/reload``), can be enabled/disabled there,
and the web UI only renders the Mole panel while it is enabled.

Drop-in contract with the framework: ``load_from_directory`` expects a
module-level ``plugin_class`` symbol (see PluginManager.load_from_directory).
"""

from __future__ import annotations

from domain.plugins import Plugin


class MolePlugin(Plugin):
    """Read-only checks across the API, streams and journeys."""

    __version__ = "1.0.0"
    __author__ = "sloughGPT"

    def setup(self, context):
        # No hooks to register: the mole's probe surface is domain.core's
        # run_mole(); this plugin exists to own enable/disable + discovery.
        pass


plugin_class = MolePlugin
