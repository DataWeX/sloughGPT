'use client'

import { useEffect, useCallback, useMemo } from 'react'
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
import { PUBLIC_API_URL } from '@/lib/config'
import { modelController } from '@/lib/model-controller'
import { generationConfigController } from '@/lib/generation-config-controller'
import { soulsController, type Checkpoint } from '@/lib/souls-controller'
import { agentsController } from '@/lib/agents-controller'
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

const VoiceChatMode = dynamic(() => import('@/components/chat/VoiceChatMode').then(m => m.VoiceChatMode), { ssr: false })
const ConversationViewer = dynamic(() => import('@/components/chat/ConversationViewer').then(m => m.ConversationViewer), { ssr: false })
const ConversationSearch = dynamic(() => import('@/components/chat/ConversationSearch').then(m => m.ConversationSearch), { ssr: false })

export default function ChatPage() {
  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info' = 'success') => {
    const store = useToastStore.getState()
    const exists = store.toasts.some(t => t.message === message && t.type === type)
    if (!exists) store.addToast(message, type)
  }, [])

  const { state: health, refresh: refreshHealth } = useApiHealth()
  const { recordFeedback, fetchStats, fetchAdapterStats } = useFeedbackStore()

  const ui = useChatUI()
  const vision = useChatVision()
  const agents = useChatAgents()
  const engine = useChatLocalEngine(showToast)
  const model = useChatModelSettings(showToast, refreshHealth)

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
    fetchStats()
    fetchAdapterStats()
    if (health && health !== 'offline' && health.model_loaded && health.model_type) {
      model.setModel(health.model_type)
    }
    modelController.list().then((models) => {
      model.setAvailableModels(models.map(m => m.id))
      const infoMap: Record<string, { cached?: boolean; size_gb?: number }> = {}
      models.forEach(m => { infoMap[m.id] = { cached: m.cached, size_gb: m.size_gb } })
      model.setModelInfoMap(infoMap)
    }).catch((err) => addGlobalError(err, 'Chat:ModelsList'))
    generationConfigController.get().then((config) => {
      model.setTemperature(config.temperature)
      model.setMaxTokens(config.max_new_tokens)
    }).catch((err) => addGlobalError(err, 'Chat:GenConfig'))
    soulsController.list().then((data) => {
      model.setSouls(data.souls || [])
      if (data.current_soul) {
        const found = (data.souls || []).find(s => s.name === data.current_soul)
        if (found) model.setCurrentSoul(found)
      }
    }).catch((err) => addGlobalError(err, 'Chat:Souls'))
    agentsController.list().then((data) => {
      const localAgents = Object.values(AGENTS)
      const merged = data && data.length > 0 ? data : localAgents
      agents.setAgents(merged)
      const savedAgentId = localStorage.getItem('man_current_agent') || 'general'
      const found = merged.find(a => a.id === savedAgentId)
      if (found) agents.setCurrentAgent(found)
    }).catch((err) => {
      addGlobalError(err, 'Chat:Agents')
      const localAgents = Object.values(AGENTS)
      agents.setAgents(localAgents)
      const savedAgentId = localStorage.getItem('man_current_agent') || 'general'
      const found = localAgents.find(a => a.id === savedAgentId)
      if (found) agents.setCurrentAgent(found)
    })
    soulsController.listCheckpoints().then(({ checkpoints: ckpts }) => {
      model.setCheckpoints((ckpts || []).map((c: Checkpoint) => ({
        name: c.name || 'unknown',
        loss: c.loss,
        traits: c.traits ? Object.keys(c.traits) : undefined,
        is_loaded: (c as any).is_loaded || false,
        eval_verdict: c.verdict,
      })))
    }).catch((err) => addGlobalError(err, 'Chat:Checkpoints'))
    fetch(`${PUBLIC_API_URL}/learn/status`).then(r => r.json()).then(data => {
      if (data && data.total_tokens_ingested !== undefined) model.setLearnerInfo(data)
    }).catch((err) => addGlobalError(err, 'Chat:LearnerStatus'))
  }, [fetchStats, fetchAdapterStats, health,
      model.setModel, model.setAvailableModels, model.setModelInfoMap,
      model.setTemperature, model.setMaxTokens,
      model.setSouls, model.setCurrentSoul,
      agents.setAgents, agents.setCurrentAgent,
      model.setCheckpoints, model.setLearnerInfo,
  ])

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
  const handleSelectAgentWithToast = useCallback((a: any) => {
    agents.setCurrentAgent(a)
    localStorage.setItem('man_current_agent', a.id)
    showToast(`Switched to ${a.name}`, 'info')
  }, [agents.setCurrentAgent, showToast])

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <div className="flex flex-1 min-h-0 overflow-hidden rounded-none lg:rounded-lg border border-border/30 bg-background shadow-sm">
        <div className="flex flex-col flex-1 min-h-0 min-w-0 max-w-full overflow-hidden">
          <ChatToolbar
            sidebarConversations={chat.sidebarConversations}
            sessionIdRef={chat.sessionIdRef}
            onLoadSession={chat.loadSession}
            onStarSession={chat.starSession}
            onPinSession={chat.pinSession}
            onNewChat={chat.newChat}
            searchQuery={ui.searchQuery}
            onSearchChange={ui.handleSearchChange}
            onSearchClear={ui.handleSearchClear}
            matchIndex={ui.matchIndex}
            matchCount={matchCount}
            matchIds={matchIds}
            onPrevMatch={handlePrevMatch}
            onNextMatch={handleNextMatch}
            showMobileSearch={ui.showMobileSearch}
            setShowMobileSearch={ui.setShowMobileSearch}
            availableModels={model.availableModels}
            model={model.model}
            loadingModel={model.loadingModel}
            generating={chat.loading}
            modelInfoMap={model.modelInfoMap}
            downloadProgress={model.downloadProgress}
            onSelectModel={model.handleSelectModel}
            souls={model.souls}
            currentSoul={model.currentSoul}
            onSelectSoul={model.handleSelectSoul}
            knowledgeCtx={agents.knowledgeCtx}
            onToggleKnowledge={agents.handleToggleKnowledge}
            agents={agents.agents}
            currentAgent={agents.currentAgent}
            onSelectAgent={handleSelectAgentWithToast}
            localModelUrl={engine.localModelUrl}
            useLocalEngine={engine.useLocalEngine}
            localEngineLoading={engine.localEngineLoading}
            localArchInfo={engine.localArchInfo}
            onToggleLocalEngine={engine.handleToggleLocalEngine}
            onVoiceMode={() => ui.setVoiceMode(true)}
            onToggleTools={() => ui.setToolPanelOpen(prev => !prev)}
            onExportMarkdown={chat.handleExportMarkdown}
            hasMessages={chat.messages.length > 0}
          />

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
      </div>

      {ui.toolPanelOpen && (
        <ChatToolPanel
          open={true}
          onClose={() => ui.setToolPanelOpen(false)}
          sessionId={chat.sessionIdRef.current}
          learnerInfo={model.learnerInfo}
          learnerTraining={model.learnerTraining}
          onTrainStep={async () => {
            model.setLearnerTraining(true)
            try {
              const resp = await fetch(`${PUBLIC_API_URL}/learn/train`, { method: 'POST' })
              if (resp.ok) {
                const data = await resp.json()
                if (data.current_loss !== undefined) showToast(`Train step: loss ${data.current_loss.toFixed(4)}`)
                else showToast('Train step complete')
                model.setLearnerInfo(prev => {
                  if (!prev) return prev
                  return {
                    ...prev,
                    train_steps_completed: data.train_steps_completed ?? prev.train_steps_completed,
                    current_loss: data.current_loss ?? prev.current_loss,
                    loss_history: data.loss_history ?? prev.loss_history,
                  }
                })
              } else showToast('Train step failed', 'error')
            } catch { showToast('Train step failed', 'error') }
            finally { model.setLearnerTraining(false) }
          }}
          checkpoints={model.checkpoints}
          currentCheckpoint={model.currentCheckpoint}
          onLoadCheckpoint={async (name) => {
            try {
              await soulsController.loadCheckpoint(name)
              model.setCurrentCheckpoint(name)
              showToast(`Checkpoint loaded: ${name}`)
            } catch { showToast('Failed to load checkpoint', 'error') }
          }}
          agents={agents.agents}
          currentAgent={agents.currentAgent}
          onSelectAgent={(a) => agents.setCurrentAgent(a)}
          availableModels={model.availableModels}
          currentModel={model.model}
          onSelectModel={async (m) => {
            if (m === model.model) return
            model.setModel(m)
            try {
              await modelController.load(m)
              showToast(`Model loaded: ${m}`)
            } catch { showToast(`Failed to load model: ${m}`, 'error') }
          }}
          souls={model.souls}
          currentSoulName={model.currentSoul?.name}
          onSwitchSoul={async (name) => {
            try {
              await soulsController.switch(name)
              const s = model.souls.find(s => s.name === name)
              if (s) model.setCurrentSoul(s)
            } catch (e) { console.error('Failed to switch soul:', e) }
          }}
          onOpenSettings={ui.toggleSettings}
          onOpenShortcuts={() => window.dispatchEvent(new CustomEvent('toggle-shortcuts'))}
          onOpenConversationViewer={() => ui.setShowConversationViewer(true)}
          visionImagesLearned={vision.visionCaps?.images_learned}
          visionTrained={vision.visionCaps?.trained}
          visionStatus={vision.visionCaps?.status}
          visionCaptionHistory={vision.visionCaptionHistory}
          visionVocabSize={vision.visionVocabSize}
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
  )
}
