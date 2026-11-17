'use client'

/**
 * useChatStoreBridge — the store's CONFIG seam (state never flows through here).
 *
 * Since card 20261116_124020 the ChatStore is the single state authority, and
 * since stage 3 it also owns the send/regenerate pipeline — which reads its
 * inputs from `_config`. An init-once config would silently freeze those
 * inputs at first render (stale model, stubbed feedback/vision/knowledge
 * callbacks, useLocalEngine pinned false), so this bridge keeps `_config` in
 * sync with every live controller input: model/soul/agent settings, engine
 * refs, prompts, toasts, and the hook-owned session-list callbacks
 * (saveSessionToStorage / renameSession) the store calls through the seam.
 *
 * `init` still runs once for its session/userId bootstrap; the sync effect
 * only ever replaces `_config`.
 */

import { useCallback, useEffect, useRef } from 'react'
import { useChatStore, type ChatConfig } from '@/lib/chat-store'
import { useFeedbackStore } from '@/lib/feedback-store'
import type { ChatPageController } from '@/features/chat/hooks/useChatPageController'

export function useChatStoreBridge(controller: ChatPageController) {
  const { model, showToast, vision, agents, engine, customSystemPrompt, chat } = controller
  const { recordFeedback } = useFeedbackStore()
  const initRef = useRef(false)

  const buildConfig = useCallback(
    (): ChatConfig => ({
      model: model.model,
      temperature: model.temperature,
      maxTokens: model.maxTokens,
      currentSoul: model.currentSoul,
      currentAgent: agents.currentAgent ?? null,
      useLocalEngine: engine.useLocalEngine,
      engineRef: engine.engineRef,
      engineLoadingRef: engine.engineLoadingRef,
      initLocalEngine: engine.initLocalEngine,
      customSystemPrompt: customSystemPrompt || '',
      showToast,
      recordFeedback,
      onVisionUpdate: (caps, history, vocab) => {
        vision.setVisionCaps(caps)
        vision.setVisionCaptionHistory(history)
        vision.setVisionVocabSize(vocab)
      },
      onKnowledgeUpdate: (ctx) => {
        agents.setKnowledgeCtx((prev) => ({ ...prev, ...ctx }))
      },
      // Session-list ops stay hook-owned (useChatSessions): the store calls
      // them through this seam instead of touching the list itself.
      saveSessionToStorage: (msgs, sessionId) => chat.saveSessionToStorage(msgs, sessionId),
      renameSession: (id, name) => chat.renameSession(id, name),
    }),
    [model, showToast, vision, agents, engine, customSystemPrompt, recordFeedback, chat],
  )

  // Bootstrap once: sessionId/userId hydration lives in store.init.
  useEffect(() => {
    if (initRef.current) return
    initRef.current = true
    useChatStore.getState().init(buildConfig())
  }, [buildConfig])

  // Config sync — whenever any input changes, the store's _config follows.
  useEffect(() => {
    useChatStore.setState({ _config: buildConfig() })
  }, [buildConfig])
}
