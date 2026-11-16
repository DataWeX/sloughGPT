'use client'

/**
 * useChatStoreBridge — store config initialization only.
 *
 * Since card 20261116_124020 the ChatStore is the single state authority:
 * useChatMessages declares its state as store selectors and writes through
 * store actions, so the old one-way controller → store message mirror that
 * lived here is gone (it could only ever lag behind — readers on the store,
 * writers on the hook, no reverse path). What remains is `init`, which hands
 * the store its ChatConfig (model, prompts, showToast, local-engine refs) —
 * the store's send/stop/regenerate actions need it via getState().
 */

import { useEffect, useRef } from 'react'
import { useChatStore, type ChatConfig } from '@/lib/chat-store'
import type { ChatPageController } from '@/features/chat/hooks/useChatPageController'

export function useChatStoreBridge(controller: ChatPageController) {
  const { model, showToast } = controller
  const initRef = useRef(false)

  // Initialize store config once
  useEffect(() => {
    if (initRef.current) return
    initRef.current = true

    const config: ChatConfig = {
      model: model.model,
      temperature: model.temperature,
      maxTokens: model.maxTokens,
      currentSoul: model.currentSoul,
      currentAgent: controller.agents?.currentAgent ?? null,
      useLocalEngine: false,
      engineRef: { current: null } as any,
      engineLoadingRef: { current: false } as any,
      initLocalEngine: async () => false,
      customSystemPrompt: controller.customSystemPrompt || '',
      showToast,
      recordFeedback: async () => true,
      onVisionUpdate: () => {},
      onKnowledgeUpdate: () => {},
    }
    useChatStore.getState().init(config)
  }, [controller, model, showToast])
}
