---
id: 20260815_050100_standardize-route-loading-fallbacks-labeled-skeletons
title: Standardize route loading fallbacks — labeled skeletons
status: done
tags: frontend,web,ui,loading
created: 2026-08-15T05:01:00.814063+00:00
---

Standardize route loading fallbacks — labeled skeletons

13 loading.tsx fallbacks (agents, auth, benchmark, feedback, files, images, learn, monitoring, registry, security, souls, tokenizer, voice) passed title="" to PageContainer, which renders an anonymous PageSkeleton with no page label. Coded properly: each now passes the real page title/subtitle plus loadingContent={<PageSkeleton cards={3} header={false} />}, so the loading state renders the actual AppRouteHeader (labeled skeleton) with content skeletons below. Dropped the centering className hacks on agents/monitoring that compensated for the anonymous skeleton. route-fallbacks.test.tsx: 66/66 pass. tsc: 0 errors in these files; one pre-existing error in hooks/useTrainingSession.ts (concurrent actor's in-progress work, not touched).