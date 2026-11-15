---
id: 20261114_150751_slonet-as-first-class-modelprovider-honest-capabilities-regi
status: todo
board: 20261114_150751_slonet-as-first-class-modelprovider-honest-capabilities-regi
title: "SloNet as first-class ModelProvider: honest capabilities + registry default + protocol parity test"
tags: ["models", "inference", "architecture"]
sprint: ""
gh: null
---

# SloNet as first-class ModelProvider: honest capabilities + registry default + protocol parity test

User ask: expose SloNet through the model wrapper provider seam like every other model. The seam already exists — ModelProvider protocol (domain/models/_internal/provider/protocols.py), ProviderRouter, register_provider/get_provider; SloNetChatProvider is already registered as 'slonet-native' in provider/setup.py. Do NOT put provider logic inside SloNet itself: SloNet is the dumb tensor/autograd engine (serves training too); the adapter (SloNetChatProvider, domain/inference/_internal/slonet_provider.py) is the wrapper, router is the consumer. Scope: (1) honest ModelCapabilities — ProviderRouter hardcodes vision=True today; router must reflect the underlying provider's declared capabilities (our model is vision=False until multimodal encoder lands — flips true only after that card). (2) 'slonet-native' as default registered text provider when a local checkpoint exists (single selection path, no serving path bypassing the router — endpoint-rule discipline). (3) parity test asserting SloNetChatProvider satisfies ModelProvider exactly like other providers (runtime_checkable protocol gate). Out of scope: SloNet tensor API/training path changes, hosted ApiProvider transport semantics, multimodal encoder itself. Gate: affected tests + ruff.
