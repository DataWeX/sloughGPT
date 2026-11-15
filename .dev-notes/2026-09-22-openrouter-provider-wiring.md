# OpenRouter / OpenAI-compatible provider wiring

**Status:** done
**Sprint:** —
**gh:** —

## Progress
- Backend `60b64c1b` (`feat/openrouter-provider`): `configure_api_provider()`,
  `ProviderSettings`, GET+PATCH `/settings/providers/api`, startup apply, tests.
  Merged into main via `feat/openrouter-settings-ui`.
- Frontend `5d6e1034`: `settingsController.getProviderApi/updateProviderApi` +
  `SettingsExternalProviderCard` on the settings page (endpoint URL, API key,
  model id, enable toggle) — separate from service API URL + HF token.
- Key is server-stored only (GET returns `api_key_set`, never raw key);
  PATCH includes `api_key` only when user types one (or explicit clear).
- Verified: 112 FE tests green, tsc 0, 15 wiring/api_provider + 7 router pytest green.

## Task
Wire `ApiProvider` (domain/inference/_internal/api_provider.py) into the
runtime so sloughGPT chat can use OpenRouter (or any OpenAI-compatible
`/v1/chat/completions` endpoint) as a model source alongside the local
SloNet model.

Currently `ApiProvider` is fully built + tested (streaming, retries,
vision, functions) but never instantiated in production — only referenced
by its own docstring example and `test_api_provider.py`.

## Body

1. Routing: a provider registry/switch so chat selects `ApiProvider` vs
   local SloNet per session (or settings toggle).
2. Settings UI: provider section in the Connection card (API URL + API key
   + model id) — kept separate from the existing HF-token field, which is
   for downloading private HF models, not inference.
3. Key storage: reuse the local-browser secret pattern already used for the
   HF token.

## Constraints
- Do NOT confuse with the Connection card's API URL (backend service
  address, health-checked) — this is a new provider endpoint setting.
- Follow the Core Infrastructure Sync Rule: frontend calls go through
  http-client.ts.