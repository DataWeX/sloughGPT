---
id: 20260731_142315_infra-coverage-round-2b-personality-companion-loggers-vector
title: Infra coverage round 2b: personality, companion, loggers, vector stores
status: done
tags: infrastructure,tests
created: 2026-07-31T14:23:15.326853+00:00
---

Infra coverage round 2b: personality, companion, loggers, vector stores

Added 137 tests across 7 previously-untested modules: ai_personality (23), companion (29), console_logger (28), shell_logger (13), web_logger (21), chromadb_store (11), pinecone_store (12). All green. ConsoleLogger/ShellLogger ANSI tested via monkeypatched _Ansi constants (module-level _c() evaluated at import); time-format assertions compute expected via datetime.fromtimestamp (TZ-safe). ChromaDB/Pinecone connect() exercised via sys.modules import mocking; discovered ValueError in pinecone connect is swallowed by generic except Exception -> returns False. Companion clean_response replacement is case-sensitive. Regression gate (prior 5 data-pipeline files + logging/vector_store neighbors) green, exit 0. pycache cleaned.