---
id: 20260805_095324_dedupe-fine-tuned-models-in-chat-dropdown
title: Dedupe fine-tuned models in chat dropdown
status: done
tags: frontend,chat
created: 2026-08-05T09:53:24.377939+00:00
---

Dedupe fine-tuned models in chat dropdown

Loading a fine-tuned model reports its dir name as the loaded identity, so /models returns it as the loaded entry and the chat base-model group would show it as a duplicate of the dedicated Fine-tuned section. useChatModelSettings.fetchInitialData now awaits listFineTuned() in the same batch and filters fine-tuned dir names out of availableModels. Also pass reported identity (not base id) to the process-guard worker name. Tests: 1 new hook test (filtering); 53 affected tests, 812 chat+hooks, tsc clean.