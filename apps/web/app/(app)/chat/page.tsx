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
import type { ChatMessage } from '@/lib/chat-utils'
import { modelController } from '@/lib/model-controller'
import { chatController } from '@/lib/chat-controller'
import { generationConfigController } from '@/lib/generation-config-controller'
import { AGENTS } from '@/lib/agents'
import { useFeedbackStore } from '@/lib/feedback-store'
import { useToastStore } from '@/lib/toast-store'
import { addGlobalError } from '@/lib/error-store'
import { imagesController } from '@/lib/images-controller'
import { filesController } from '@/lib/files-controller'
import {
  ChatSettings, ChatArea, ErrorBanner,
} from '@/components/chat'
import { ModeBar } from '@/components/chat/ModeBar'
import { ChatToolbar } from '@/components/chat/ChatToolbar'
import { ChatToolPanel } from '@/components/chat/ChatToolPanel'
import { ConversationSidebar } from '@/components/chat/ConversationSidebar'
import { DownloadDialog } from '@/components/chat/DownloadDialog'
import { SystemPromptDialog } from '@/components/chat/SystemPromptDialog'
import { ChatProvider } from '@/contexts/ChatContext'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import { useChatToolbarValue } from '@/hooks/useChatToolbarValue'
import { useChatHealthValue, useChatModelValue, useChatUIValue } from '@/hooks/useChatContextValue'

import { Button } from '@/components/ui/button'

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
  const [chatMode, setChatMode] = useState<'chat' | 'write' | 'decide' | 'explain' | 'translate' | 'brainstorm' | 'wellness' | 'create' | 'read' | 'talk'>('chat')
  const [writeTone, setWriteTone] = useState('Friendly')
  const [writeType, setWriteType] = useState('Email')
  const [decideStructure, setDecideStructure] = useState('Pros & Cons')
  const [explainDifficulty, setExplainDifficulty] = useState('Simple')
  const [translateLangPair, setTranslateLangPair] = useState('EN→ES')
  const [brainstormTopic, setBrainstormTopic] = useState('Name Ideas')
  const [wellnessType, setWellnessType] = useState('Sleep Story')
  const [createStyle, setCreateStyle] = useState('Realistic')
  const [readFileData, setReadFileData] = useState<{ text: string; filename: string; pages: number } | null>(null)
  const [readLoading, setReadLoading] = useState(false)
  const [customSystemPrompt, setCustomSystemPrompt] = useState(() => {
    try { return localStorage.getItem('chat:customSystemPrompt') || '' } catch { return '' }
  })
  const [systemPromptOpen, setSystemPromptOpen] = useState(false)

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
    customSystemPrompt,
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
    searchInputRef: ui.searchInputRef,
    handleSearchChange: ui.handleSearchChange,
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
  }, [ui, matchCount, matchIds])

  const handleNextMatch = useCallback(() => {
    const newIdx = ui.matchIndex < matchCount - 1 ? ui.matchIndex + 1 : 0
    ui.setMatchIndex(newIdx)
    const el = document.getElementById(`msg-${matchIds[newIdx]}`)
    el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [ui, matchCount, matchIds])

  // ── Effects ───────────────────────────────────────────────────────────────

  useEffect(() => {
    const newChatHandler = () => chat.newChatRef.current?.()
    window.addEventListener('new-chat', newChatHandler)
    return () => window.removeEventListener('new-chat', newChatHandler)
  }, [chat.newChatRef])

  useEffect(() => {
    const handler = () => ui.setShowConversationSearch(true)
    window.addEventListener('search-conversations', handler)
    return () => window.removeEventListener('search-conversations', handler)
  }, [ui])

  useEffect(() => {
    modelController.list().then(models => {
      const desc: Record<string, string> = {}
      models.forEach(m => { if (m.description) desc[m.id] = m.description })
      setModelDescriptions(desc)
    }).catch(() => {})
  }, [])

  const [suggestions, setSuggestions] = useState<{ text: string; icon: string }[]>([])

  useEffect(() => {
    if (health && health !== 'offline' && health.model_loaded) {
      chatController.getSuggestions().then(setSuggestions)
    } else {
      setSuggestions([])
    }
  }, [health])

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
  }, [fetchStats, fetchAdapterStats, health, model, agents])

  useEffect(() => {
    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    fetch(`${API_URL}/vector/init`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: 'chromadb', dimension: 384 }),
    }).catch((err) => addGlobalError(err, 'Chat:VectorInit'))
  }, [])

  // ── Flyweights: clearChat / selectAgent with toast ────────────────────────

  const clearChat = useCallback(() => chat.newChat(), [chat])

  const handleSelectAgentWithToast = useCallback((agent: any) => {
    agents.setCurrentAgent(agent)
    showToast(`Switched to ${agent?.name || 'no agent'}`)
  }, [agents, showToast])

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
    onSystemPrompt: () => setSystemPromptOpen(true),
  })

  const healthValue = useChatHealthValue({ health, refreshHealth })
  const modelValue = useChatModelValue({ model, agents, vision, chat, showToast })
  const uiValue = useChatUIValue({ ui, showToast })

  const handleSaveSystemPrompt = useCallback((value: string) => {
    setCustomSystemPrompt(value)
    try { localStorage.setItem('chat:customSystemPrompt', value) } catch {}
  }, [])

  const handleWriteSend = useCallback(async () => {
    if (chatMode === 'write' && chat.input.trim()) {
      const instruction = `Write a ${writeTone.toLowerCase()} ${writeType.toLowerCase()} about: `
      chat.sendMessage(instruction + chat.input.trim())
      chat.setInput('')
    } else if (chatMode === 'decide' && chat.input.trim()) {
      const instruction = `Help me decide using ${decideStructure.toLowerCase()}: `
      chat.sendMessage(instruction + chat.input.trim())
      chat.setInput('')
    } else if (chatMode === 'explain' && chat.input.trim()) {
      const instruction = `Explain this at a ${explainDifficulty.toLowerCase()} level (as if explaining to a ${explainDifficulty.toLowerCase()} learner): `
      chat.sendMessage(instruction + chat.input.trim())
      chat.setInput('')
    } else if (chatMode === 'translate' && chat.input.trim()) {
      const [src, tgt] = translateLangPair.split('→')
      const instruction = `Translate this from ${src} to ${tgt}: `
      chat.sendMessage(instruction + chat.input.trim())
      chat.setInput('')
    } else if (chatMode === 'brainstorm' && chat.input.trim()) {
      const instruction = `Let's brainstorm ${brainstormTopic.toLowerCase()}. Be creative, give me ideas in a friendly list format: `
      chat.sendMessage(instruction + chat.input.trim())
      chat.setInput('')
    } else if (chatMode === 'wellness' && chat.input.trim()) {
      const prompts: Record<string, string> = {
        'Sleep Story': 'Tell me a calming sleep story',
        'Meditation': 'Guide me through a short meditation',
        'Breathing': 'Guide me through a breathing exercise',
        'Affirmation': 'Share a positive affirmation',
      }
      const instruction = `Respond in a gentle, soothing tone. ${prompts[wellnessType] || 'Help me feel calm'}: `
      chat.sendMessage(instruction + chat.input.trim())
      chat.setInput('')
    } else if (chatMode === 'create' && chat.input.trim()) {
      const text = chat.input.trim()
      chat.setInput('')
      // Add user message immediately
      const userMsg: ChatMessage = {
        id: Date.now().toString(), role: 'user', content: text, timestamp: new Date(),
      }
      const pendingId = (Date.now() + 1).toString()
      const pendingMsg: ChatMessage = {
        id: pendingId, role: 'assistant', content: '✨ **Creating your image...**', timestamp: new Date(),
      }
      chat.setMessages(prev => [...prev, userMsg, pendingMsg])
      chat.setLoading(true)
      // Generate image
      try {
        const result = await imagesController.generate(text, createStyle.toLowerCase())
        chat.setMessages(prev => prev.map(m =>
          m.id === pendingId
            ? { ...m, content: `Here's your ${createStyle.toLowerCase()} image:\n\n![${text}](${result.image})` }
            : m
        ))
      } catch (err: any) {
        chat.setMessages(prev => prev.map(m =>
          m.id === pendingId
            ? { ...m, content: `❌ Sorry, I couldn't create that image. ${err?.message || 'Please try again.'}` }
            : m
        ))
      } finally {
        chat.setLoading(false)
      }
    } else if (chatMode === 'read' && chat.input.trim()) {
      if (!readFileData) {
        useToastStore.getState().addToast('Upload a file first, then ask your question', 'info')
        return
      }
      const question = chat.input.trim()
      chat.setInput('')
      const fullPrompt = `[I'm asking about the file "${readFileData.filename}"]\n\nHere is the file content:\n${readFileData.text.slice(0, 12000)}\n\n---\n\nMy question: ${question}`
      chat.sendMessage(fullPrompt)
    } else {
      chat.sendMessage()
    }
  }, [chatMode, writeTone, writeType, decideStructure, explainDifficulty, translateLangPair, brainstormTopic, wellnessType, createStyle, readFileData, chat])

  const handleReadFile = useCallback(async (file: File) => {
    setReadLoading(true)
    try {
      const result = await filesController.extract(file)
      setReadFileData({ text: result.text, filename: result.filename, pages: result.pages })
      const fileName = result.filename
      const pageInfo = result.extension === '.pdf' ? ` (${result.pages} pages)` : ''
      // Add a system message confirming the file was read
      chat.setMessages(prev => [...prev, {
        id: `file-${Date.now()}`, role: 'assistant', content: `📄 **Read: ${fileName}**${pageInfo}\n\nGot it! I've read ${result.chars.toLocaleString()} characters${pageInfo ? ` across ${result.pages} pages` : ''}. What do you want to know?`, timestamp: new Date(),
      }])
    } catch (err: any) {
      useToastStore.getState().addToast(`Couldn't read file: ${err?.message || 'Unknown error'}`, 'error')
    } finally {
      setReadLoading(false)
    }
  }, [chat])

  // Open voice overlay when Talk mode is selected
  useEffect(() => {
    if (chatMode === 'talk') {
      ui.setVoiceMode(true)
    }
  }, [chatMode, ui])

  return (
    <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>
    <a href="#chat-messages" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-background focus:border focus:rounded-lg focus:shadow-lg">
      Skip to messages
    </a>
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <ConversationSidebar
        conversations={chat.sidebarConversations}
        currentConversationId={chat.sessionIdRef.current}
        onLoadConversation={chat.loadSession}
        onNewChat={chat.newChat}
        onDeleteConversation={chat.deleteSession}
        onStarConversation={chat.starSession}
        onPinConversation={chat.pinSession}
        onArchiveConversation={chat.archiveSession}
        archivedCount={chat.archivedCount}
        onRenameConversation={chat.renameSession}
        open={ui.sidebarOpen}
        onClose={() => ui.setSidebarOpen(false)}
      />
      <main className="flex flex-1 min-h-0 overflow-hidden rounded-none lg:rounded-lg border border-border/30 bg-[hsl(var(--chat-bg))] shadow-sm" aria-label="Chat">
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

          <ModeBar
            mode={chatMode}
            tone={writeTone}
            type={writeType}
            decideStructure={decideStructure}
            difficulty={explainDifficulty}
            langPair={translateLangPair}
            brainstormTopic={brainstormTopic}
            wellnessType={wellnessType}
            createStyle={createStyle}
            onModeChange={setChatMode}
            onToneChange={setWriteTone}
            onTypeChange={setWriteType}
            onDecideStructureChange={setDecideStructure}
            onDifficultyChange={setExplainDifficulty}
            onLangPairChange={setTranslateLangPair}
            onBrainstormTopicChange={setBrainstormTopic}
            onWellnessTypeChange={setWellnessType}
            onCreateStyleChange={setCreateStyle}
          />

          {chatMode === 'read' && (
            <div className="px-3 py-2 border-b border-border/10 bg-muted/5">
              {!readFileData ? (
                <div className="flex flex-col items-center gap-2 py-6 border-2 border-dashed border-border/30 rounded-lg text-center cursor-pointer hover:border-primary/40 hover:bg-muted/10 transition-colors" onDragOver={e => e.preventDefault()} onDrop={async e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleReadFile(f) }}>
                  <input
                    type="file"
                    accept=".pdf,.docx,.txt,.md,.csv,.json"
                    className="hidden"
                    id="read-file-input"
                    onChange={e => { const f = e.target.files?.[0]; if (f) handleReadFile(f) }}
                  />
                  <label htmlFor="read-file-input" className="cursor-pointer flex flex-col items-center gap-1">
                    <span className="text-2xl">📄</span>
                    <span className="text-sm font-medium">{readLoading ? 'Reading your file...' : 'Drop a file here or click to upload'}</span>
                    <span className="text-[11px] text-muted-foreground">PDF, Word, TXT, MD, CSV, JSON</span>
                  </label>
                </div>
              ) : (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-base">📄</span>
                  <span className="font-medium truncate max-w-[200px]">{readFileData.filename}</span>
                  {readFileData.pages > 1 && <span className="text-xs text-muted-foreground">({readFileData.pages} pages)</span>}
                  <button
                    onClick={() => { setReadFileData(null); chat.setMessages(prev => prev.filter(m => !m.id.startsWith('file-'))) }}
                    className="ml-auto text-xs text-muted-foreground hover:text-foreground px-2 py-0.5 rounded hover:bg-muted/10"
                  >
                    Remove
                  </button>
                </div>
              )}
            </div>
          )}

          <ChatArea
            messages={chat.messages}
            loading={chat.loading}
            sessionLoading={chat.sessionLoading}
            model={model.model}
            health={health}
            suggestions={suggestions}
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
            onSend={handleWriteSend}
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
          onClose={() => { ui.setVoiceMode(false); setChatMode('chat') }}
        />
      )}

      <SystemPromptDialog
        open={systemPromptOpen}
        onOpenChange={setSystemPromptOpen}
        value={customSystemPrompt}
        onSave={handleSaveSystemPrompt}
      />
    </div>
    </ChatProvider>
  )
}
