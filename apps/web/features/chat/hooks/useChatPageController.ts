'use client'

import { useEffect, useCallback, useMemo, useState } from 'react'
import { useRouter, useSearchParams } from 'next/navigation'

import { useLiveStatus } from '@/hooks/useLiveStatus'
import { soulsController, multimodalController, modelController } from '@/lib/controllers'
import type { ChatCommand } from '@/lib/chat-commands'
import { useChatUI } from '@/features/chat/hooks/useChatUI'
import { useChatVision } from '@/features/chat/hooks/useChatVision'
import { useChatAgents } from '@/features/chat/hooks/useChatAgents'
import type { AgentDef } from '@/lib/agents'
import { extractErrorMessage, formatToastError } from '@/lib/error-utils'
import { PDF_ANALYSIS_MAX_TOKENS } from '@/lib/format-bytes'
import { useChatLocalEngine } from '@/features/chat/hooks/useChatLocalEngine'
import { useChatModelSettings } from '@/features/chat/hooks/useChatModelSettings'
import { useChatKeyboard } from '@/features/chat/hooks/useChatKeyboard'
import { useChatBookmarks } from '@/features/chat/hooks/useChatBookmarks'
import { useChatMessages } from '@/features/chat/hooks/useChatMessages'
import { useChatMode } from '@/features/chat/hooks/useChatMode'
import { useMessageNotes } from '@/features/chat/hooks/useMessageNotes'
import { useMessageThreads } from '@/features/chat/hooks/useMessageThreads'
import { computeSearchMatches } from '@/lib/chat-utils'
import type { ChatMessage } from '@/lib/chat-utils'
import { chatController } from '@/lib/chat-controller'
import { useFeedbackStore } from '@/lib/feedback-store'
import { useToastStore } from '@/lib/toast-store'
import { useSettings } from '@/lib/store'
import { imagesController } from '@/lib/images-controller'
import type { ImageStyle } from '@/lib/images-controller'
import { chatDB } from '@/lib/db'

import { knowledgeController } from '@/lib/knowledge-controller'
import { resizeImage } from '@/features/chat/components/input/ImageUpload'
import { useChatToolbarValue } from '@/features/chat/hooks/useChatToolbarValue'
import { useChatHealthValue, useChatModelValue, useChatUIValue } from '@/features/chat/hooks/useChatContextValue'
import { useConvSidebar } from '@/features/chat/contexts/ConvSidebarContext'

const MAX_FILE_CONTENT_CHARS = 12000
const MAX_KNOWLEDGE_CONTENT_CHARS = 500

export function useChatPageController(
  showToast: (message: string, type?: 'success' | 'error' | 'info') => void,
  refreshHealth: () => Promise<void>,
) {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { healthLegacy: health } = useLiveStatus()
  const { recordFeedback, fetchStats, fetchAdapterStats } = useFeedbackStore()

  const ui = useChatUI()
  const vision = useChatVision()
  const agents = useChatAgents()
  const engine = useChatLocalEngine(showToast)
  const model = useChatModelSettings(showToast, refreshHealth)
  const settings = useSettings()
  const [modelDescriptions, setModelDescriptions] = useState<Record<string, string>>({})
  const [readFileData, setReadFileData] = useState<{ text: string; filename: string; pages: number } | null>(null)
  const [readLoading, setReadLoading] = useState(false)
  const [customSystemPrompt, setCustomSystemPrompt] = useState('')
  useEffect(() => {
    chatDB.getKV<string>('chat:customSystemPrompt').then(v => {
      if (v) setCustomSystemPrompt(v)
    })
  }, [])
  const [systemPromptOpen, setSystemPromptOpen] = useState(false)
  const [noteDialogOpen, setNoteDialogOpen] = useState(false)
  const [noteDialogMessageId, setNoteDialogMessageId] = useState<string | null>(null)
  const [noteSearchOpen, setNoteSearchOpen] = useState(false)
  const [shortcutsOpen, setShortcutsOpen] = useState(false)
  const [templatesOpen, setTemplatesOpen] = useState(false)
  const [conversationSearchQuery, setConversationSearchQuery] = useState('')
  const [conversationSearchOpen, setConversationSearchOpen] = useState(false)

  const { bookmarks, addBookmark, removeBookmark, isBookmarked, clearAll } = useChatBookmarks()

  const { setOpen: setConvSidebarOpen, convCollapsed, toggleConv } = useConvSidebar()
  useEffect(() => {
    setConvSidebarOpen(true)
    return () => { setConvSidebarOpen(false) }
  }, [setConvSidebarOpen])

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

  const messageNotes = useMessageNotes({ sessionId: chat.sessionIdRef.current })

  const threads = useMessageThreads({ messages: chat.messages })
  const [activeThreadMessageId, setActiveThreadMessageId] = useState<string | null>(null)

  const {
    chatMode, setChatMode,
    writeTone, setWriteTone,
    writeType, setWriteType,
    rewriteStyle, setRewriteStyle,
    decideStructure, setDecideStructure,
    explainDifficulty, setExplainDifficulty,
    translateLangPair, setTranslateLangPair,
    brainstormTopic, setBrainstormTopic,
    wellnessType, setWellnessType,
    createStyle, setCreateStyle,
    handleSend: handleModeSend,
  } = useChatMode({
    chat: {
      input: chat.input,
      setInput: chat.setInput,
      sendMessage: chat.sendMessage,
      setMessages: chat.setMessages,
      setLoading: chat.setLoading,
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
    onRenameConversation: () => {
      const name = prompt('Rename conversation:')
      if (name && name.trim()) {
        const sid = chat.sessionIdRef.current
        if (sid) chat.renameSession(sid, name.trim())
        showToast(`Renamed to "${name.trim()}"`, 'success')
      }
    },
    onExportMarkdown: () => chat.handleExportMarkdown(),
    onCancelStream: () => chat.cancelStream(),
    onApproveTool: () => chat.handleToolApproval(true),
    onDenyTool: () => chat.handleToolApproval(false),
    onDuplicateConversation: () => {
      const sid = chat.sessionIdRef.current
      if (sid) {
        chat.duplicateSession(sid)
        showToast('Conversation duplicated', 'success')
      }
    },
    onToggleBookmarks: () => ui.setToolPanelOpen(prev => !prev),
    onToggleSidebar: () => ui.setSidebarOpen(prev => !prev),
    onAddNoteToLastMessage: () => {
      const messages = chat.messages
      if (messages.length > 0) {
        const lastMsg = messages[messages.length - 1]
        if (lastMsg.id) {
          setNoteDialogMessageId(lastMsg.id)
          setNoteDialogOpen(true)
        }
      }
    },
    onOpenNoteSearch: () => setNoteSearchOpen(true),
    onOpenShortcuts: () => setShortcutsOpen(true),
    onOpenTemplates: () => setTemplatesOpen(true),
    onOpenConversationSearch: () => setConversationSearchOpen(true),
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
    }).catch(() => /* model descriptions unavailable — UI still works */ {})
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
        }).catch(() => /* clipboard unavailable */ {})
      }
    }
    window.addEventListener('copy-last-response', handler)
    return () => window.removeEventListener('copy-last-response', handler)
  }, [chat.messages, showToast])

  useEffect(() => {
    fetchStats()
    fetchAdapterStats()
    const { fetchInitialData: fetchModelData } = model
    const { fetchInitialData: fetchAgentData } = agents
    const healthModel = health && health !== 'offline' && (health.model_loaded || health.model_type) ? health.model_type : undefined
    fetchModelData(healthModel)
    fetchAgentData()
  }, [fetchStats, fetchAdapterStats, health, model.fetchInitialData, agents.fetchInitialData])

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
        const sid = chat.sessionIdRef.current
        if (sid) {
          chat.archiveSession(sid, true)
          chat.newChat()
        }
        showToast('Conversation archived', 'success')
      },
      renameConversation: (name: string) => {
        const sid = chat.sessionIdRef.current
        if (sid) chat.renameSession(sid, name)
      },
      searchConversations: (query: string) => {
        ui.setShowConversationSearch(true)
      },
    }
    try {
      await cmd.execute(args, context)
    } catch (err: unknown) {
      showToast(formatToastError(err, 'Could not command'), 'error')
    }
  }, [chat, clearChat, model, showToast, router, ui])

  const handleSelectAgentWithToast = useCallback((agent: AgentDef | null) => {
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
    onSearchConversations: () => ui.setShowConversationSearch(true),
    bookmarkCount: bookmarks.length,
  })

  const healthValue = useChatHealthValue({ health, refreshHealth })
  const modelValue = useChatModelValue({ model, agents, vision, chat, showToast })
  const uiValue = useChatUIValue({ ui, showToast })

  const handleSaveSystemPrompt = useCallback((value: string) => {
    setCustomSystemPrompt(value)
    chatDB.setKV('chat:customSystemPrompt', value)
  }, [])

  const handleCreateImage = useCallback(async (prompt: string) => {
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: 'user', content: prompt, timestamp: new Date() }
    const pendingId = crypto.randomUUID()
    const pendingMsg: ChatMessage = { id: pendingId, role: 'assistant', content: '✨ **Creating your image...**', timestamp: new Date() }
    chat.setMessages(prev => [...prev, userMsg, pendingMsg])
    chat.setLoading(true)
    try {
      const result = await imagesController.generate(prompt, createStyle.toLowerCase() as ImageStyle)
      chat.setMessages(prev => prev.map(m =>
        m.id === pendingId ? { ...m, content: `Here's your ${createStyle.toLowerCase()} image:\n\n![${prompt}](${result.image})` } : m
      ))
    } catch (err: unknown) {
      chat.setMessages(prev => prev.map(m =>
        m.id === pendingId ? { ...m, content: `❌ Sorry, I couldn't create that image. ${extractErrorMessage(err, 'Please try again.')}` } : m
      ))
    } finally { chat.setLoading(false) }
  }, [chat, createStyle])

  const handleWriteSend = useCallback(async () => {
    const input = chat.input.trim()
    if (!input && chatMode !== 'read') { chat.sendMessage(); return }
    await handleModeSend(readFileData)
  }, [chatMode, readFileData, chat, handleModeSend])

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

  const handleSaveToKnowledge = useCallback(async (messageId: string, content: string) => {
    try {
      await knowledgeController.add(content.slice(0, MAX_KNOWLEDGE_CONTENT_CHARS), 'chat-saved', true)
      showToast('Saved to knowledge', 'success')
    } catch {
      showToast('Could not save to knowledge', 'error')
    }
  }, [showToast])

  const handleReadFile = useCallback(async (file: File) => {
    setReadLoading(true)
    try {
      const text = await file.text()
      const fileName = file.name
      const ext = fileName.includes('.') ? fileName.slice(fileName.lastIndexOf('.')) : ''
      const pages = ext === '.pdf' ? Math.max(1, Math.ceil(text.length / 3000)) : 0
      setReadFileData({ text, filename: fileName, pages })
      const pageInfo = pages > 0 ? ` (${pages} pages)` : ''
      // Add a system message confirming the file was read
      chat.setMessages(prev => [...prev, {
        id: `file-${Date.now()}`, role: 'assistant', content: `📄 **Read: ${fileName}**${pageInfo}\n\nGot it! I've read ${text.length.toLocaleString()} characters${pageInfo ? ` across ${pages} pages` : ''}. What do you want to know?`, timestamp: new Date(),
      }])
    } catch (err: unknown) {
      useToastStore.getState().addToast(formatToastError(err, "Couldn't read file"), 'error')
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
      showToast('Could not attach image', 'error')
    }
  }, [chat, showToast])

  const handleTextDropped = useCallback((content: string, filename: string) => {
    const prefix = `📄 ${filename}:\n\`\`\`\n`
    const suffix = `\n\`\`\`\n\nWhat would you like me to do with this file?`
    chat.setInput(prev => prev ? `${prev}\n\n${prefix}${content}${suffix}` : `${prefix}${content}${suffix}`)
    showToast(`Text from ${filename} inserted — edit or send`, 'info')
  }, [chat, showToast])

  const handlePDFDropped = useCallback(async (file: File) => {
    showToast(`Analyzing ${file.name}...`, 'info')
    try {
      const result = await multimodalController.uploadPDF(file, 'Analyze this document and summarize its contents.', {
        perPage: false,
        maxNewTokens: PDF_ANALYSIS_MAX_TOKENS,
      })
      const analysis = result.analysis || JSON.stringify(result)
      chat.setMessages(prev => [...prev, {
        id: `pdf-user-${Date.now()}`,
        role: 'user',
        content: `📎 Uploaded PDF: ${file.name}`,
        timestamp: new Date(),
      }, {
        id: `pdf-${Date.now()}`,
        role: 'assistant',
        content: analysis,
        timestamp: new Date(),
      }])
      showToast('PDF analyzed — see response below', 'info')
    } catch (err: unknown) {
      showToast(formatToastError(err, 'Could not pdf analysis'), 'error')
    }
  }, [chat, showToast])

  // Open voice overlay when Talk mode is selected
  useEffect(() => {
    if (chatMode === 'talk') {
      ui.setVoiceMode(true)
    }
  }, [chatMode, ui])

  const onSaveNote = useCallback((note: string) => {
    if (noteDialogMessageId) {
      if (note === '') {
        messageNotes.removeNote(noteDialogMessageId)
      } else {
        messageNotes.setNote(noteDialogMessageId, note)
      }
    }
  }, [noteDialogMessageId, messageNotes])

  const onDeleteNote = useCallback(() => {
    if (noteDialogMessageId) {
      messageNotes.removeNote(noteDialogMessageId)
    }
  }, [noteDialogMessageId, messageNotes])

  const onAddNote = useCallback((messageId: string) => {
    setNoteDialogMessageId(messageId)
    setNoteDialogOpen(true)
  }, [])

  const onStartThread = useCallback((parentMessageId: string) => {
    threads.createThread(parentMessageId)
    setActiveThreadMessageId(parentMessageId)
  }, [threads])

  const onReplyInThread = useCallback((threadId: string, content: string) => {
    threads.addToThread(threadId, {
      id: `thread-msg-${Date.now()}`,
      role: 'user',
      content,
      timestamp: new Date(),
    })
  }, [threads])

  const onCloseThread = useCallback(() => {
    setActiveThreadMessageId(null)
  }, [])

  return {
    health,
    refreshHealth,
    showToast,
    ui,
    chat,
    model,
    agents,
    vision,
    engine,
    chatMode, setChatMode,
    writeTone, setWriteTone,
    writeType, setWriteType,
    rewriteStyle, setRewriteStyle,
    decideStructure, setDecideStructure,
    explainDifficulty, setExplainDifficulty,
    translateLangPair, setTranslateLangPair,
    brainstormTopic, setBrainstormTopic,
    wellnessType, setWellnessType,
    createStyle, setCreateStyle,
    bookmarks,
    removeBookmark,
    clearAll,
    isBookmarked,
    convCollapsed,
    toggleConv,
    modelDescriptions,
    readFileData, setReadFileData,
    readLoading,
    customSystemPrompt,
    systemPromptOpen, setSystemPromptOpen,
    suggestions,
    collapsibleLength: settings.collapsibleMessageLength,
    clearChat,
    toolbarValue,
    healthValue,
    modelValue,
    uiValue,
    handleSaveSystemPrompt,
    handleWriteSend,
    handleExecuteCommand,
    handleSelectAgentWithToast,
    handleToggleBookmark,
    handleDeleteMessage,
    handleSaveToKnowledge,
    handleReadFile,
    handleImageDropped,
    handleTextDropped,
    handlePDFDropped,
    contextLayers: chat.contextLayers,
    handleReact: chat.handleReact,
    handlePin: chat.handlePin,
    selectedMessageIds: chat.selectedMessageIds,
    selectionMode: chat.selectionMode,
    toggleSelectionMode: chat.toggleSelectionMode,
    toggleMessageSelection: chat.toggleMessageSelection,
    selectAllMessages: chat.selectAllMessages,
    clearSelection: chat.clearSelection,
    deleteSelectedMessages: chat.deleteSelectedMessages,
    noteMap: messageNotes.notes,
    noteDialogOpen: noteDialogOpen,
    setNoteDialogOpen: setNoteDialogOpen,
    noteDialogMessageId: noteDialogMessageId,
    noteDialogNote: noteDialogMessageId ? messageNotes.getNote(noteDialogMessageId) || '' : '',
    onSaveNote,
    onDeleteNote,
    onAddNote,
    noteSearchOpen: noteSearchOpen,
    setNoteSearchOpen: setNoteSearchOpen,
    onNavigateToNote: (sessionId: string, messageId: string) => {
      if (sessionId !== chat.sessionIdRef.current) {
        chat.loadSession(sessionId)
      }
      setTimeout(() => {
        const el = document.getElementById(`msg-${messageId}`)
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'center' })
          el.focus()
        }
      }, 100)
    },
    activeThreadMessageId: activeThreadMessageId,
    activeThread: activeThreadMessageId ? threads.getThread(activeThreadMessageId) : undefined,
    activeThreadMessages: activeThreadMessageId && threads.getThread(activeThreadMessageId)
      ? threads.getThreadMessages(threads.getThread(activeThreadMessageId)!.id)
      : [],
    onStartThread,
    onReplyInThread,
    onCloseThread,
    hasThread: threads.hasThread,
    threadCount: threads.threadCount,
    shortcutsOpen: shortcutsOpen,
    setShortcutsOpen: setShortcutsOpen,
    templatesOpen: templatesOpen,
    setTemplatesOpen: setTemplatesOpen,
    conversationSearchQuery: conversationSearchQuery,
    setConversationSearchQuery: setConversationSearchQuery,
    conversationSearchOpen: conversationSearchOpen,
    setConversationSearchOpen: setConversationSearchOpen,
  }
}

export type ChatPageController = ReturnType<typeof useChatPageController>
