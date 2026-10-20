'use client'

/**
 * useChatStoreBridge — syncs existing controller state into the Zustand store.
 *
 * This is a migration bridge: it reads from the existing useChatPageController
 * and pushes state into useChatStore so that components can gradually switch
 * from controller props to Zustand selectors.
 *
 * Call this once in ChatPage. It runs an effect that keeps the store in sync.
 */

import { useEffect, useRef } from 'react'
import { useChatStore, type ChatConfig } from '@/lib/chat-store'
import type { ChatSession } from '@/lib/chat-utils'
import type { ChatPageController } from '@/features/chat/hooks/useChatPageController'

export function useChatStoreBridge(controller: ChatPageController) {
  const { chat, model, health, ui, showToast } = controller
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
      engineRef: { current: null },
      engineLoadingRef: { current: false },
      initLocalEngine: async () => false,
      customSystemPrompt: controller.customSystemPrompt || '',
      showToast,
      recordFeedback: async () => true,
      onVisionUpdate: () => {},
      onKnowledgeUpdate: () => {},
    }
    useChatStore.getState().init(config)
  }, [controller, model, showToast])

  // Sync messages from controller → store (one-way bridge)
  useEffect(() => {
    const unsubscribe = useChatStore.subscribe((state, prevState) => {
      // If store messages diverge from controller, the store is the source of truth
      // (during migration, controller is still primary — sync controller → store)
    })

    // Push controller state into store on every change
    useChatStore.setState({
      messages: chat.messages,
      loading: chat.loading,
      sessionLoading: chat.sessionLoading,
      input: chat.input,
      images: chat.images,
      sessionId: chat.sessionIdRef.current,
      toolEvents: chat.toolEvents,
      ragVerification: chat.ragVerification,
      contextLayers: chat.contextLayers,
      pendingToolApproval: chat.pendingToolApproval,
      currentError: chat.currentError,
      // The store slot is ChatSession[]; the controller exposes the
      // session-controller Conversation shape — map (not cast) across the seam.
      sidebarConversations: chat.sidebarConversations.map((c): ChatSession => ({
        id: c.id,
        name: c.name,
        messages: (c.messages ?? []).map((m) => ({
          id: m.id,
          role: m.role === 'assistant' ? 'assistant' : 'user',
          content: m.content,
          timestamp: new Date(m.timestamp ?? Date.now()),
        })),
        createdAt: c.created_at,
        updatedAt: c.updated_at,
        synced: Boolean(c.synced),
        starred: c.starred,
        pinned: c.pinned,
        archived: c.archived,
      })),
      selectedMessageIds: chat.selectedMessageIds,
      selectionMode: chat.selectionMode,
    })

    return unsubscribe
  }, [
    chat.messages,
    chat.loading,
    chat.sessionLoading,
    chat.input,
    chat.images,
    chat.toolEvents,
    chat.ragVerification,
    chat.contextLayers,
    chat.pendingToolApproval,
    chat.currentError,
    chat.sidebarConversations,
    chat.selectedMessageIds,
    chat.selectionMode,
    chat.sessionIdRef,
  ])
}
