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
    if (chatMode === 'write') {
      chat.sendMessage(`Write a ${writeTone.toLowerCase()} ${writeType.toLowerCase()} about: ${input}`)
      chat.setInput('')
    } else if (chatMode === 'rewrite') {
      const rewritePrompts: Record<string, string> = {
        'Fix Grammar': 'Fix all grammar and spelling errors in this text while keeping the meaning',
        'Make Shorter': 'Make this text shorter and more concise while keeping the key points',
        'Make Friendlier': 'Rewrite this text in a warmer, more friendly tone',
        'Make Professional': 'Rewrite this text in a professional, formal tone',
        'Sound Like Me': 'Rewrite this text to sound more natural and conversational, like a real person wrote it',
      }
      chat.sendMessage(`${rewritePrompts[rewriteStyle] || 'Rewrite this text'}:\n\n${input}`)
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
      chat.sendMessage(`[I'm asking about the file "${readFileData.filename}"]\n\nHere is the file content:\n${readFileData.text.slice(0, MAX_FILE_CONTENT_CHARS)}\n\n---\n\nMy question: ${input}`)
    } else {
      chat.sendMessage()
    }
  }, [chatMode, writeTone, writeType, rewriteStyle, decideStructure, explainDifficulty, translateLangPair, brainstormTopic, wellnessType, readFileData, chat, handleCreateImage])

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
  }
}

export type ChatPageController = ReturnType<typeof useChatPageController>
