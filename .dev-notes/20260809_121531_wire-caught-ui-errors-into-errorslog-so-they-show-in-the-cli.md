---
id: 20260809_121531_wire-caught-ui-errors-into-errorslog-so-they-show-in-the-cli
title: Wire caught UI errors into /errors/log so they show in the CLI
status: done
tags: ui,errors,observability
created: 2026-08-09T12:15:31.138666+00:00
---

Wire caught UI errors into /errors/log so they show in the CLI

DatasetImportModal: handleImport + handleSearch catch blocks now call reportError(message, 'dataset-import', { metadata: { source, action } }) after setError. Handled UI errors (e.g. 'Enter a search term or ISBN') previously stayed local to the red alert box and never reached the backend, so they never appeared in the API server CLI. Now they batch through /errors/log and log as CLIENT ERROR at error level on the server console. Test: added mock of @/lib/error-reporter + 'shows error and reports it for empty ISBN search'. Verified tsc exit 0, modal 23/23, training page 16/16.