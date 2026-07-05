'use client'
export const dynamic = 'force-dynamic'

import { useEffect, useCallback, useMemo, useState } from 'react'
import dynamicNext from 'next/dynamic'
import { useRouter, useSearchParams } from 'next/navigation'
import { useApiHealth } from '@/hooks/useApiHealth'
import { soulsController } from '@/lib/controllers'
import type { ChatCommand } from '@/lib/chat-commands'
import { useChatUI } from '@/hooks/useChatUI'
import { useChatVision } from '@/hooks/useChatVision'
import { useChatAgents } from '@/hooks/useChatAgents'
import { useChatLocalEngine } from '@/hooks/useChatLocalEngine'
import { useChatModelSettings } from '@/hooks/useChatModelSettings'
import { useChatKeyboard } from '@/hooks/useChatKeyboard'
import { useChatBookmarks } from '@/hooks/useChatBookmarks'
import { useChatMessages } from '@/hooks/useChatMessages'
import { computeSearchMatches } from '@/lib/chat-utils'
import type { ChatMessage } from '@/lib/chat-utils'
import { modelController } from '@/lib/model-controller'
import { chatController } from '@/lib/chat-controller'
import { generationConfigController } from '@/lib/generation-config-controller'

import { useFeedbackStore } from '@/lib/feedback-store'
import { useToastStore } from '@/lib/toast-store'
import { addGlobalError } from '@/lib/error-store'
import { useSettings } from '@/lib/store'
import { apiPost } from '@/lib/http-client'
import { imagesController } from '@/lib/images-controller'
import type { ImageStyle } from '@/lib/images-controller'
import { filesController } from '@/lib/files-controller'
import { ChatArea, ErrorBanner } from '@/components/chat'
import { ImageDropZone } from '@/components/chat/ImageDropZone'
import { resizeImage } from '@/components/chat/ImageUpload'
import { ModeBar } from '@/components/chat/ModeBar'
import { ChatToolbar } from '@/components/chat/ChatToolbar'
import { ChatProvider } from '@/contexts/ChatContext'
import { ChatToolbarProvider } from '@/contexts/ChatToolbarContext'
import { useChatToolbarValue } from '@/hooks/useChatToolbarValue'
import { useChatHealthValue, useChatModelValue, useChatUIValue } from '@/hooks/useChatContextValue'


const VoiceChatMode = dynamicNext(() => import('@/components/chat/VoiceChatMode').then(m => m.VoiceChatMode), { ssr: false })
const ConversationViewer = dynamicNext(() => import('@/components/chat/ConversationViewer').then(m => m.ConversationViewer), { ssr: false })
const ConversationSearch = dynamicNext(() => import('@/components/chat/ConversationSearch').then(m => m.ConversationSearch), { ssr: false })
const SearchConversationsDialog = dynamicNext(() => import('@/components/chat/SearchConversationsDialog').then(m => m.SearchConversationsDialog), { ssr: false })
const ChatSettings = dynamicNext(() => import('@/components/chat/ChatSettings').then(m => m.ChatSettings), { ssr: false })
const ConversationSidebar = dynamicNext(() => import('@/components/chat/ConversationSidebar').then(m => m.ConversationSidebar), { ssr: false })
const ChatToolPanel = dynamicNext(() => import('@/components/chat/ChatToolPanel').then(m => m.ChatToolPanel), { ssr: false })
const DownloadDialog = dynamicNext(() => import('@/components/chat/DownloadDialog').then(m => m.DownloadDialog), { ssr: false })
const SystemPromptDialog = dynamicNext(() => import('@/components/chat/SystemPromptDialog').then(m => m.SystemPromptDialog), { ssr: false })
const ReadFileSection = dynamicNext(() => import('@/components/chat/ReadFileSection'), { ssr: false })

export default function ChatPage() {
  const showToast = useCallback((message: string, type: string = 'success') => {
    const store = useToastStore.getState()
    const exists = store.toasts.some(t => t.message === message && t.type === type)
    if (!exists) store.addToast(message, type as 'success' | 'error' | 'info')
  }, [])

  const router = useRouter()
  const searchParams = useSearchParams()
  const { state: health, refresh: refreshHealth } = useApiHealth()
  const { recordFeedback, fetchStats, fetchAdapterStats } = useFeedbackStore()

  const ui = useChatUI()
  const vision = useChatVision()
  const agents = useChatAgents()
  const engine = useChatLocalEngine(showToast)
  const model = useChatModelSettings(showToast, refreshHealth)
  const settings = useSettings()
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
  const [searchConversationsOpen, setSearchConversationsOpen] = useState(false)
  const [searchConversationsQuery, setSearchConversationsQuery] = useState('')

  const { bookmarks, addBookmark, removeBookmark, isBookmarked, clearAll } = useChatBookmarks()

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

  useEffect(() => {
    const sessionId = searchParams.get('session')
    if (sessionId) {
      chat.loadSession(sessionId)
    }
  }, [chat, searchParams])

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
    apiPost('/vector/init', { provider: 'chromadb', dimension: 384 })
      .catch((err) => addGlobalError(err, 'Chat:VectorInit'))
  }, [])

  // ── Flyweights: clearChat / selectAgent with toast ────────────────────────

  const clearChat = useCallback(() => chat.newChat(), [chat])

  const handleExecuteCommand = useCallback(async (cmd: ChatCommand, args: string[]) => {
    const addSystemMessage = (content: string) => {
      chat.setMessages(prev => [...prev, {
        id: `cmd-${Date.now()}`, role: 'assistant' as const, content, timestamp: new Date(),
      }])
    }
    const context: import('@/lib/chat-commands').CommandContext = {
      showToast: (msg, type) => showToast(msg, type || 'info'),
      clearChat,
      setTemperature: model.setTemperature,
      setModel: async (name) => { await model.setModel(name) },
      setSoul: async (name) => { await soulsController.switch(name) },
      exportChat: chat.handleExportMarkdown,
      attachFile: () => {
        const input = document.querySelector<HTMLInputElement>('input[type="file"]')
        input?.click()
      },
      searchKnowledge: async (query) => {
        addSystemMessage(`🔍 Searching knowledge base for "${query}"...`)
      },
      navigateTo: router.push,
      addSystemMessage,
      sendMessage: chat.sendMessage,
      archiveConversation: () => {
        const name = `Chat - ${new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}`
        const sid = chat.sessionIdRef.current
        if (sid) chat.renameSession(sid, name)
        chat.setInput('')
        showToast(`Archived as "${name}"`, 'success')
      },
      renameConversation: (name: string) => {
        const sid = chat.sessionIdRef.current
        if (sid) chat.renameSession(sid, name)
      },
      searchConversations: (query: string) => {
        setSearchConversationsQuery(query)
        setSearchConversationsOpen(true)
      },
    }
    try {
      await cmd.execute(args, context)
    } catch (err: any) {
      showToast(`Command failed: ${err?.message || 'Unknown error'}`, 'error')
    }
  }, [chat, clearChat, model, showToast, router, setSearchConversationsOpen, setSearchConversationsQuery])

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
    onSearchConversations: () => setSearchConversationsOpen(true),
    bookmarkCount: bookmarks.length,
  })

  const healthValue = useChatHealthValue({ health, refreshHealth })
  const modelValue = useChatModelValue({ model, agents, vision, chat, showToast })
  const uiValue = useChatUIValue({ ui, showToast })

  const handleSaveSystemPrompt = useCallback((value: string) => {
    setCustomSystemPrompt(value)
    try { localStorage.setItem('chat:customSystemPrompt', value) } catch {}
  }, [])

  const handleCreateImage = useCallback(async (prompt: string) => {
    const userMsg: ChatMessage = { id: Date.now().toString(), role: 'user', content: prompt, timestamp: new Date() }
    const pendingId = (Date.now() + 1).toString()
    const pendingMsg: ChatMessage = { id: pendingId, role: 'assistant', content: '✨ **Creating your image...**', timestamp: new Date() }
    chat.setMessages(prev => [...prev, userMsg, pendingMsg])
    chat.setLoading(true)
    try {
      const result = await imagesController.generate(prompt, createStyle.toLowerCase() as ImageStyle)
      chat.setMessages(prev => prev.map(m =>
        m.id === pendingId ? { ...m, content: `Here's your ${createStyle.toLowerCase()} image:\n\n![${prompt}](${result.image})` } : m
      ))
    } catch (err: any) {
      chat.setMessages(prev => prev.map(m =>
        m.id === pendingId ? { ...m, content: `❌ Sorry, I couldn't create that image. ${err?.message || 'Please try again.'}` } : m
      ))
    } finally { chat.setLoading(false) }
  }, [chat, createStyle])

  const handleWriteSend = useCallback(async () => {
    const input = chat.input.trim()
    if (!input && chatMode !== 'read') { chat.sendMessage(); return }
    if (chatMode === 'write') {
      chat.sendMessage(`Write a ${writeTone.toLowerCase()} ${writeType.toLowerCase()} about: ${input}`)
      chat.setInput('')
    } else if (chatMode === 'decide') {
      chat.sendMessage(`Help me decide using ${decideStructure.toLowerCase()}: ${input}`)
      chat.setInput('')
    } else if (chatMode === 'explain') {
      chat.sendMessage(`Explain this at a ${explainDifficulty.toLowerCase()} level (as if explaining to a ${explainDifficulty.toLowerCase()} learner): ${input}`)
      chat.setInput('')
    } else if (chatMode === 'translate') {
      const [src, tgt] = translateLangPair.split('→')
      chat.sendMessage(`Translate this from ${src} to ${tgt}: ${input}`)
      chat.setInput('')
    } else if (chatMode === 'brainstorm') {
      chat.sendMessage(`Let's brainstorm ${brainstormTopic.toLowerCase()}. Be creative, give me ideas in a friendly list format: ${input}`)
      chat.setInput('')
    } else if (chatMode === 'wellness') {
      const prompts: Record<string, string> = { 'Sleep Story': 'Tell me a calming sleep story', 'Meditation': 'Guide me through a short meditation', 'Breathing': 'Guide me through a breathing exercise', 'Affirmation': 'Share a positive affirmation' }
      chat.sendMessage(`Respond in a gentle, soothing tone. ${prompts[wellnessType] || 'Help me feel calm'}: ${input}`)
      chat.setInput('')
    } else if (chatMode === 'create') {
      chat.setInput('')
      await handleCreateImage(input)
    } else if (chatMode === 'read') {
      if (!readFileData) { useToastStore.getState().addToast('Upload a file first, then ask your question', 'info'); return }
      chat.setInput('')
      chat.sendMessage(`[I'm asking about the file "${readFileData.filename}"]\n\nHere is the file content:\n${readFileData.text.slice(0, 12000)}\n\n---\n\nMy question: ${input}`)
    } else {
      chat.sendMessage()
    }
  }, [chatMode, writeTone, writeType, decideStructure, explainDifficulty, translateLangPair, brainstormTopic, wellnessType, readFileData, chat, handleCreateImage])

  const handleToggleBookmark = useCallback((messageId: string) => {
    const msg = chat.messages.find(m => m.id === messageId)
    if (!msg) return
    if (isBookmarked(messageId)) {
      removeBookmark(messageId)
    } else {
      addBookmark({
        id: msg.id,
        content: typeof msg.content === 'string' ? msg.content : '',
        role: msg.role,
        timestamp: typeof msg.timestamp === 'number' ? msg.timestamp : msg.timestamp?.getTime() || Date.now(),
      })
    }
  }, [chat.messages, isBookmarked, removeBookmark, addBookmark])

  const handleDeleteMessage = useCallback((messageId: string) => {
    chat.setMessages(prev => prev.filter(m => m.id !== messageId))
    if (isBookmarked(messageId)) {
      removeBookmark(messageId)
    }
    showToast('Message deleted', 'info')
  }, [chat, isBookmarked, removeBookmark, showToast])

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

  const handleImageDropped = useCallback(async (file: File) => {
    try {
      const dataUrl = await resizeImage(file, 512)
      chat.handleAddImage(dataUrl)
      showToast('Image attached — drop more or send message', 'info')
    } catch {
      showToast('Failed to attach image', 'error')
    }
  }, [chat, showToast])

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

          {ui.showSettings && (
          <ChatSettings
            isOpen={true}
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
          )}

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
            <ReadFileSection
              readLoading={readLoading}
              readFileData={readFileData}
              onFileSelected={handleReadFile}
              onRemove={() => { setReadFileData(null); chat.setMessages(prev => prev.filter(m => !m.id.startsWith('file-'))) }}
            />
          )}

          <ImageDropZone onImageDropped={handleImageDropped}>
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
              toolEvents={chat.toolEvents}
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
              onExecuteCommand={handleExecuteCommand}
              isBookmarked={isBookmarked}
              onBookmark={handleToggleBookmark}
              onDelete={handleDeleteMessage}
              collapsibleLength={settings.collapsibleMessageLength}
            />
          </ImageDropZone>

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
          bookmarks={bookmarks}
          onRemoveBookmark={removeBookmark}
          onClearBookmarks={clearAll}
        />
      )}

      {model.pendingDownload !== null && (
        <DownloadDialog
          open={true}
          pendingDownload={model.pendingDownload}
          modelInfoMap={model.modelInfoMap}
          onCancel={() => model.setPendingDownload(null)}
          onConfirm={(modelId) => {
            const info = model.modelInfoMap[modelId]
            model.startDownloadFlowRef.current(modelId, info?.size_gb)
          }}
        />
      )}

      {ui.voiceMode && (
        <VoiceChatMode
          onMessage={async (text) => {
            chat.setInput(text)
            await chat.sendMessage(text)
          }}
          onClose={() => { ui.setVoiceMode(false); setChatMode('chat') }}
        />
      )}

      {searchConversationsOpen && (
        <SearchConversationsDialog
          open={true}
          onOpenChange={setSearchConversationsOpen}
          initialQuery={searchConversationsQuery}
        />
      )}
      {systemPromptOpen && (
        <SystemPromptDialog
          open={true}
          onOpenChange={setSystemPromptOpen}
          value={customSystemPrompt}
          onSave={handleSaveSystemPrompt}
        />
      )}
    </div>
    </ChatProvider>
  )
}
