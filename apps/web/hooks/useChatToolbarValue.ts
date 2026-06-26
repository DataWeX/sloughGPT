'use client'

import { useMemo, type MutableRefObject } from 'react'
import type { ChatToolbarContextValue } from '@/contexts/ChatToolbarContext'
import type { useChatUI } from './useChatUI'
import type { useChatVision } from './useChatVision'
import type { useChatAgents } from './useChatAgents'
import type { useChatLocalEngine } from './useChatLocalEngine'
import type { useChatModelSettings } from './useChatModelSettings'
import type { useChatMessages } from './useChatMessages'
import type { ApiHealthSnapshot } from './useApiHealth'
import type { Conversation } from '@/lib/session-controller'
import { datasetController } from '@/lib/dataset-controller'

interface UseChatToolbarValueConfig {
  ui: ReturnType<typeof useChatUI>
  vision: ReturnType<typeof useChatVision>
  agents: ReturnType<typeof useChatAgents>
  engine: ReturnType<typeof useChatLocalEngine>
  model: ReturnType<typeof useChatModelSettings>
  chat: ReturnType<typeof useChatMessages>
  health: ApiHealthSnapshot
  matchCount: number
  matchIds: string[]
  handlePrevMatch: () => void
  handleNextMatch: () => void
  handleSelectAgentWithToast: (agent: any) => void
  modelDescriptions: Record<string, string>
  showToast: (message: string, type?: string) => void
  onSystemPrompt: () => void
}

export function useChatToolbarValue(config: UseChatToolbarValueConfig): ChatToolbarContextValue {
  const {
    ui, agents, engine, model, chat, health,
    matchCount, matchIds, handlePrevMatch, handleNextMatch,
    handleSelectAgentWithToast, modelDescriptions, showToast,
  } = config

  return useMemo<ChatToolbarContextValue>(() => ({
    conversations: {
      conversations: chat.sidebarConversations,
      sessionIdRef: chat.sessionIdRef,
      onLoad: chat.loadSession,
      onStar: chat.starSession,
      onPin: chat.pinSession,
      onNewChat: chat.newChat,
    },
    search: {
      query: ui.searchQuery,
      onChange: ui.handleSearchChange,
      onClear: ui.handleSearchClear,
      matchIndex: ui.matchIndex,
      matchCount: config.matchCount,
      matchIds: config.matchIds,
      onPrevMatch: config.handlePrevMatch,
      onNextMatch: config.handleNextMatch,
      showMobile: ui.showMobileSearch,
      setShowMobile: ui.setShowMobileSearch,
      searchInputRef: ui.searchInputRef,
    },
    model: {
      availableModels: model.availableModels,
      current: model.model,
      loading: model.loadingModel,
      generating: chat.loading,
      infoMap: model.modelInfoMap,
      descriptions: modelDescriptions,
      downloadProgress: model.downloadProgress,
      onSelect: model.handleSelectModel,
      onUnload: model.handleUnloadModel,
    },
    soul: {
      souls: model.souls,
      current: model.currentSoul,
      onSelect: model.handleSelectSoul,
    },
    knowledge: {
      showing: agents.knowledgeCtx.showing,
      count: agents.knowledgeCtx.count,
      context: agents.knowledgeCtx.context,
      onToggle: agents.handleToggleKnowledge,
    },
    agent: {
      agents: agents.agents,
      current: agents.currentAgent,
      onSelect: config.handleSelectAgentWithToast,
    },
    localEngine: {
      modelUrl: engine.localModelUrl,
      useLocal: engine.useLocalEngine,
      loading: engine.localEngineLoading,
      archInfo: engine.localArchInfo,
      onToggle: engine.handleToggleLocalEngine,
    },
    actions: {
      onVoiceMode: () => ui.setVoiceMode(true),
      onToggleTools: () => ui.setToolPanelOpen(prev => !prev),
      onExportMarkdown: chat.handleExportMarkdown,
      onCopyMarkdown: chat.handleCopyMarkdown,
      onSystemPrompt: config.onSystemPrompt,
      onSaveAsDataset: async () => {
        try {
          const msgs = chat.messages
            .filter(m => m.role === 'user' || m.role === 'assistant')
            .map(m => ({ role: m.role, content: m.content }))
          if (msgs.length === 0) { showToast('No messages to save', 'error'); return }
          const res = await datasetController.createFromChat({
            messages: msgs,
            name: `chat-${new Date().toISOString().slice(0, 10)}`,
          })
          showToast(`Saved ${res.messages_exported} messages as dataset: ${res.name}`, 'success')
        } catch { showToast('Failed to save dataset', 'error') }
      },
      hasMessages: chat.messages.length > 0,
      messageCount: chat.messages.length,
    },
    health: {
      status: health === null ? 'loading' : health === 'offline' ? 'offline' : health.model_loaded ? 'ok' : 'degraded',
      summary: health === null ? 'Connecting...' : health === 'offline' ? 'Server offline' : health.summary || '',
      modelLoaded: health !== null && health !== 'offline' && health.model_loaded,
      modelType: health !== null && health !== 'offline' ? health.model_type : '',
    },
    sidebar: {
      open: ui.sidebarOpen,
      onToggle: () => ui.setSidebarOpen(prev => !prev),
      onClose: () => ui.setSidebarOpen(false),
    },
  }), [chat, ui, model, agents, engine, health, matchCount, matchIds, handlePrevMatch, handleNextMatch, handleSelectAgentWithToast, modelDescriptions, showToast])
}
