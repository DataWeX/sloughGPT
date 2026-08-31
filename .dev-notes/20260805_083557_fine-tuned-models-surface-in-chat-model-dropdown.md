---
id: 20260805_083557_fine-tuned-models-surface-in-chat-model-dropdown
title: Fine-tuned models surface in chat model dropdown
status: done
tags: frontend,chat,feature
created: 2026-08-05T08:35:57.473422+00:00
---

Fine-tuned models surface in chat model dropdown

Added fine-tuned model group to chat ModelDropdown: lists models/hf-finetuned dirs, Load action calls POST /training/finetuned-models/{name}/load, sets model + refreshes health. Wired through ChatToolbarContext.model.fineTuned (optional group), useChatModelSettings (fineTuned/fineTunedLoading/fetchFineTuned/handleLoadFineTuned), useChatToolbarValue. Tests: 4 new ModelDropdown tests, 1 toolbar wiring test, 3 hook tests. 42 affected tests pass, 809 chat/hooks tests pass, tsc clean.