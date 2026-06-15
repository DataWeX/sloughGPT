'use client'

import { useEffect, useCallback, useMemo, useState } from 'react'
import dynamic from 'next/dynamic'
import { useApiHealth } from '@/hooks/useApiHealth'
import { useChatUI } from '@/hooks/useChatUI'
import { useChatVision } from '@/hooks/useChatVision'
import { useChatAgents } from '@/hooks/useChatAgents'
import { useChatLocalEngine } from '@/hooks/useChatLocalEngine'
import { useChatModelSettings } from '@/hooks/useChatModelSettings'
import { useChatKeyboard } from '@/hooks/useChatKeyboard'
import { useChatMessages } from '@/hooks/useChatMessages'
import { computeSearchMatches } from '@/lib/chat-utils'
import { modelController } from '@/lib/model-controller'
import { generationConfigController } from '@/lib/generation-config-controller'
import { AGENTS } from '@/lib/agents'
import { useFeedbackStore } from '@/lib/feedback-store'
import { useToastStore } from '@/lib/toast-store'
import { addGlobalError } from '@/lib/error-store'
import {
  ChatSettings, ChatArea, ErrorBanner,
} from '@/components/chat'
import { ChatToolbar } from '@/components/chat/ChatToolbar'
import { ChatToolPanel } from '@/components/chat/ChatToolPanel'
import { DownloadDialog } from '@/components/chat/DownloadDialog'
import { ChatProvider } from '@/contexts/ChatContext'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import { useChatToolbarValue } from '@/hooks/useChatToolbarValue'
import { useChatContextValue } from '@/hooks/useChatContextValue'

const VoiceChatMode = dynamic(() => import('@/components/chat/VoiceChatMode').then(m => m.VoiceChatMode), { ssr: false })
const ConversationViewer = dynamic(() => import('@/components/chat/ConversationViewer').then(m => m.ConversationViewer), { ssr: false })
const ConversationSearch = dynamic(() => import('@/components/chat/ConversationSearch').then(m => m.ConversationSearch), { ssr: false })

export default function ChatPage() {
  const showToast = useCallback((message: string, type: string = 'success') => {
    const store = useToastStore.getState()
    const exists = store.toasts.some(t => t.message === message && t.type === type)
    if (!exists) store.addToast(message, type as 'success' | 'error' | 'info')
  }, [])

  const { state: health, refresh: refreshHealth } = useApiHealth()
  const { recordFeedback, fetchStats, fetchAdapterStats } = useFeedbackStore()

  const ui = useChatUI()
  const vision = useChatVision()
  const agents = useChatAgents()
  const engine = useChatLocalEngine(showToast)
  const model = useChatModelSettings(showToast, refreshHealth)
  const [modelDescriptions, setModelDescriptions] = useState<Record<string, string>>({})

  const chat = useChatMessages({
    model: model.model,
    temperature: model.temperature,
    maxTokens: model.maxTokens,
    currentSoul: model.currentSoul,
    currentAgent: agents.currentAgent,
    useLocalEngine: engine.useLocalEngine,
    engineRef: engine.engineRef,
    engineLoadingRef: engine.engineLoadingRef,
    initLocalEngine: engine.initLocalEngine,
    showToast,
    recordFeedback,
    fetchStats,
    fetchAdapterStats,
    onVisionUpdate: (caps, history, vocab) => {
      vision.setVisionCaps(caps)
      vision.setVisionCaptionHistory(history)
      vision.setVisionVocabSize(vocab)
    },
    onKnowledgeUpdate: (ctx) => {
      agents.setKnowledgeCtx(prev => ({ ...prev, ...ctx }))
    },
  })

  useChatKeyboard({
    loading: chat.loading,
    currentError: chat.currentError,
    showSettings: ui.showSettings,
    setToolPanelOpen: ui.setToolPanelOpen,
    setShowSettings: ui.setShowSettings,
    setLoading: chat.setLoading,
    setCurrentError: chat.setCurrentError,
    loadingRef: chat.loadingRef,
    newChatRef: chat.newChatRef,
    handleRegenerateRef: chat.handleRegenerateRef,
  })

  // ── Computed (cross-hook) ──────────────────────────────────────────────────

  const { matchIds, matchCount } = useMemo(
    () => computeSearchMatches(chat.messages, ui.searchQuery),
    [chat.messages, ui.searchQuery],
  )

  const handlePrevMatch = useCallback(() => {
    const newIdx = ui.matchIndex > 0 ? ui.matchIndex - 1 : matchCount - 1
    ui.setMatchIndex(newIdx)
    const el = document.getElementById(`msg-${matchIds[newIdx]}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [ui.matchIndex, matchCount, matchIds, ui.setMatchIndex])

  const handleNextMatch = useCallback(() => {
    const newIdx = ui.matchIndex < matchCount - 1 ? ui.matchIndex + 1 : 0
    ui.setMatchIndex(newIdx)
    const el = document.getElementById(`msg-${matchIds[newIdx]}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [ui.matchIndex, matchCount, matchIds, ui.setMatchIndex])

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const newChatHandler = () => chat.newChatRef.current?.()
    window.addEventListener('new-chat', newChatHandler)
    return () => window.removeEventListener('new-chat', newChatHandler)
  }, [])

  useEffect(() => {
    const handler = () => ui.setShowConversationSearch(true)
    window.addEventListener('search-conversations', handler)
    return () => window.removeEventListener('search-conversations', handler)
  }, [ui.setShowConversationSearch])

  useEffect(() => {
    modelController.list().then(models => {
      const desc: Record<string, string> = {}
      models.forEach(m => { if (m.description) desc[m.id] = m.description })
      setModelDescriptions(desc)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    const handler = () => {
      const lastAssistant = [...chat.messages].reverse().find(m => m.role === 'assistant')
      if (lastAssistant?.content) {
        navigator.clipboard.writeText(lastAssistant.content).then(() => {
          showToast('Last response copied', 'info')
        }).catch(() => {})
      }
    }
    window.addEventListener('copy-last-response', handler)
    return () => window.removeEventListener('copy-last-response', handler)
  }, [chat.messages, showToast])

  useEffect(() => {
    fetchStats()
    fetchAdapterStats()
    const healthModel = health && health !== 'offline' && health.model_loaded ? health.model_type : undefined
    model.fetchInitialData(healthModel)
    agents.fetchInitialData()
  }, [fetchStats, fetchAdapterStats, health, model.fetchInitialData, agents.fetchInitialData])

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    fetch(`${API_URL}/vector/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: 'chromadb', dimension: 384 }),
    }).catch((err) => addGlobalError(err, 'Chat:VectorInit'))
  }, [])

  // ── Flyweights: clearChat / selectAgent with toast ────────────────────────

  const clearChat = useCallback(() => chat.newChat(), [chat.newChat])

  const handleSelectAgentWithToast = useCallback((agent: any) => {
    agents.setCurrentAgent(agent)
    showToast(`Switched to ${agent?.name || 'no agent'}`)
  }, [agents.setCurrentAgent, showToast])

  const toolbarValue = useChatToolbarValue({
    ui,
    vision,
    agents,
    engine,
    model,
    chat,
    health,
    matchCount,
    matchIds,
    handlePrevMatch,
    handleNextMatch,
    handleSelectAgentWithToast,
    modelDescriptions,
    showToast,
  })

  const chatContextValue = useChatContextValue({
    health,
    refreshHealth,
    model,
    agents,
    vision,
    ui,
    chat,
    showToast,
  })

  return (
    <ChatProvider value={chatContextValue}>
    <a href="#chat-messages" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-background focus:border focus:rounded-lg focus:shadow-lg">
      Skip to messages
    </a>
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <main className="flex flex-1 min-h-0 overflow-hidden rounded-none lg:rounded-lg border border-border/30 bg-background shadow-sm" aria-label="Chat">
        <div className="flex flex-col flex-1 min-h-0 min-w-0 max-w-full overflow-hidden">
          <ChatToolbarProvider value={toolbarValue}>
            <ChatToolbar />
          </ChatToolbarProvider>

          <ChatSettings
            isOpen={ui.showSettings}
            model={model.model}
            temperature={model.temperature}
            maxTokens={model.maxTokens}
            onModelChange={model.setModel}
            availableModels={model.availableModels}
            onTemperatureChange={(temp) => {
              model.setTemperature(temp)
              generationConfigController.update({ temperature: temp }).catch(() => {})
            }}
            onMaxTokensChange={(tokens) => {
              model.setMaxTokens(tokens)
              generationConfigController.update({ max_new_tokens: tokens }).catch(() => {})
            }}
            onClear={clearChat}
            hasMessages={chat.messages.length > 0}
          />

          {chat.currentError && (
            <ErrorBanner
              error={chat.currentError}
              onRetry={chat.handleRetry}
              onDismiss={() => chat.setCurrentError(null)}
            />
          )}

          <ChatArea
            messages={chat.messages}
            loading={chat.loading}
            sessionLoading={chat.sessionLoading}
            model={model.model}
            health={health}
            onRefreshHealth={refreshHealth}
            onCopy={chat.handleCopy}
            onRegenerate={chat.handleRegenerate}
            onThumbsUp={chat.handleThumbsUp}
            onThumbsDown={chat.handleThumbsDown}
            onEdit={chat.handleEditMessage}
            searchQuery={ui.searchQuery}
            onSuggestionClick={chat.handleSuggestionClick}
            value={chat.input}
            onChange={chat.setInput}
            onSend={chat.sendMessage}
            onStop={() => {
              if (chat.loadingRef.current) {
                chat.loadingRef.current.abort()
              }
              chat.setLoading(false)
            }}
            images={chat.images}
            onAddImage={chat.handleAddImage}
            onRemoveImage={chat.handleRemoveImage}
            onAudioTranscript={(text) => {
              chat.setInput(prev => prev ? `${prev} ${text}` : text)
            }}
            onGeneratedImage={(dataUrl, prompt) => {
              chat.setMessages(prev => [...prev, {
                id: `img-${Date.now()}`, role: 'user',
                content: `[Generate image: ${prompt}]`, timestamp: new Date(),
                images: [{ id: `gen-${Date.now()}`, dataUrl, name: 'generated.png' }],
              }])
              showToast('Image generated — see message above', 'info')
            }}
            onPDFError={(error) => {
              showToast(`PDF analysis failed: ${error}`, 'error')
            }}
            onPDFAnalysis={(analysis, filename) => {
              chat.setMessages(prev => [...prev, {
                id: `pdf-user-${Date.now()}`,
                role: 'user',
                content: `📎 Uploaded PDF: ${filename}`,
                timestamp: new Date(),
              }, {
                id: `pdf-${Date.now()}`,
                role: 'assistant',
                content: analysis,
                timestamp: new Date(),
              }])
              showToast('PDF analyzed — see response below', 'info')
            }}
          />

          <ConversationViewer
            isOpen={ui.showConversationViewer}
            onClose={() => ui.setShowConversationViewer(false)}
            messages={chat.messages.map(m => ({
              id: m.id, role: m.role, content: m.content,
              timestamp: typeof m.timestamp === 'number' ? m.timestamp : m.timestamp?.getTime() || Date.now(),
            }))}
            title="Current Conversation"
          />

          <ConversationSearch
            open={ui.showConversationSearch}
            onClose={() => ui.setShowConversationSearch(false)}
            onNavigate={(sessionId) => chat.loadSession(sessionId)}
          />

        </div>
      </main>

      {ui.toolPanelOpen && (
        <ChatToolPanel
          open={true}
          onClose={() => ui.setToolPanelOpen(false)}
          sessionId={chat.sessionIdRef.current}
        />
      )}

      <DownloadDialog
        open={model.pendingDownload !== null}
        pendingDownload={model.pendingDownload}
        modelInfoMap={model.modelInfoMap}
        onCancel={() => model.setPendingDownload(null)}
        onConfirm={(modelId) => {
          const info = model.modelInfoMap[modelId]
          model.startDownloadFlowRef.current(modelId, info?.size_gb)
        }}
      />

      {ui.voiceMode && (
        <VoiceChatMode
          onMessage={async (text) => {
            chat.setInput(text)
            await chat.sendMessage(text)
          }}
          onClose={() => ui.setVoiceMode(false)}
        />
      )}
    </div>
    </ChatProvider>
  )
}
