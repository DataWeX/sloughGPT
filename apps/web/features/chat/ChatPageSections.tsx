'use client'

import dynamicNext from '@/vite/next-compat/dynamic'
import { memo, useCallback, useMemo } from 'react'

import type { ChatPageController } from '@/features/chat/hooks/useChatPageController'
import { ChatArea, ErrorBanner } from '@/features/chat/components'
import { ToolApprovalDialog } from '@/features/chat/components/ToolApprovalDialog'
import { ContextInjectionBar } from '@/features/chat/components/ContextInjectionBar'
import { ImageDropZone } from '@/features/chat/components/layout/ImageDropZone'
import { ChatToolbar } from '@/features/chat/components/toolbar/ChatToolbar'
import { logger } from '@/lib/dev-log'
import { ChatToolbarProvider } from '@/features/chat/contexts/ChatToolbarContext'
import { useAppStore } from '@/lib/store'
import { useConversationScope, saveConversationScope } from '@/lib/conversation-scope-store'
import { useChatStore } from '@/lib/chat-store'

const VoiceChatMode = dynamicNext(
  () => import('@/features/chat/components/input/VoiceChatMode').then((m) => m.VoiceChatMode),
  { ssr: false },
)
const ConversationViewer = dynamicNext(
  () =>
    import('@/features/chat/components/sidebar/ConversationViewer').then(
      (m) => m.ConversationViewer,
    ),
  { ssr: false },
)
const ConversationSearch = dynamicNext(
  () =>
    import('@/features/chat/components/sidebar/ConversationSearch').then(
      (m) => m.ConversationSearch,
    ),
  { ssr: false },
)
const ChatSettings = dynamicNext(
  () => import('@/features/chat/components/dialogs/ChatSettings').then((m) => m.ChatSettings),
  { ssr: false },
)
const ConversationSidebar = dynamicNext(
  () =>
    import('@/features/chat/components/sidebar/ConversationSidebar').then(
      (m) => m.ConversationSidebar,
    ),
  { ssr: false },
)
const ChatToolPanel = dynamicNext(
  () => import('@/features/chat/components/panels/ChatToolPanel').then((m) => m.ChatToolPanel),
  { ssr: false },
)
const DownloadDialog = dynamicNext(
  () => import('@/features/chat/components/dialogs/DownloadDialog').then((m) => m.DownloadDialog),
  { ssr: false },
)
const SystemPromptDialog = dynamicNext(
  () =>
    import('@/features/chat/components/dialogs/SystemPromptDialog').then(
      (m) => m.SystemPromptDialog,
    ),
  { ssr: false },
)
const ReadFileSection = dynamicNext(
  () => import('@/features/chat/components/dialogs/ReadFileSection'),
  { ssr: false },
)
const NoteDialog = dynamicNext(
  () => import('@/features/chat/components/dialogs/NoteDialog').then((m) => m.NoteDialog),
  { ssr: false },
)
const ThreadPanel = dynamicNext(
  () => import('@/features/chat/components/ThreadPanel').then((m) => m.ThreadPanel),
  { ssr: false },
)
const KeyboardShortcutsPanel = dynamicNext(
  () =>
    import('@/features/chat/components/dialogs/KeyboardShortcutsPanel').then(
      (m) => m.KeyboardShortcutsPanel,
    ),
  { ssr: false },
)
const TemplateDialog = dynamicNext(
  () => import('@/features/chat/components/dialogs/TemplateDialog').then((m) => m.TemplateDialog),
  { ssr: false },
)
const ChatStatsPanel = dynamicNext(
  () => import('@/features/chat/components/dialogs/ChatStatsPanel').then((m) => m.ChatStatsPanel),
  { ssr: false },
)

interface ChatPageSectionProps {
  controller: ChatPageController
  consciousnessOpen?: boolean
  onConsciousnessToggle?: () => void
}

export const ChatSidebarSection = memo(function ChatSidebarSection({
  controller,
}: ChatPageSectionProps) {
  const { chat, ui, convCollapsed, toggleConv } = controller
  const handleStar = useCallback(
    (id: string, starred: boolean) => chat.starSession(id, starred),
    [chat],
  )
  const handlePin = useCallback(
    (id: string, pinned: boolean) => chat.pinSession(id, pinned),
    [chat],
  )
  const handleArchive = useCallback(
    (id: string, archived: boolean) => chat.archiveSession(id, archived),
    [chat],
  )
  const handleDuplicate = useCallback((id: string) => chat.duplicateSession(id), [chat])
  const handleCloseSidebar = useCallback(() => ui.setSidebarOpen(false), [ui])
  return (
    <ConversationSidebar
      conversations={chat.sidebarConversations}
      currentConversationId={chat.sessionIdRef.current}
      onLoadConversation={chat.loadSession}
      onNewChat={chat.newChat}
      onDeleteConversation={chat.deleteSession}
      onStarConversation={handleStar}
      onPinConversation={handlePin}
      onArchiveConversation={handleArchive}
      archivedCount={chat.archivedCount}
      onRenameConversation={chat.renameSession}
      onDuplicateConversation={handleDuplicate}
      open={ui.sidebarOpen}
      onClose={handleCloseSidebar}
      collapsed={convCollapsed}
      onToggleCollapse={toggleConv}
    />
  )
})

export const ChatToolbarSection = memo(function ChatToolbarSection({
  controller,
  consciousnessOpen,
  onConsciousnessToggle,
}: ChatPageSectionProps) {
  return (
    <ChatToolbarProvider value={controller.toolbarValue}>
      <ChatToolbar
        consciousnessOpen={consciousnessOpen}
        onConsciousnessToggle={onConsciousnessToggle}
      />
    </ChatToolbarProvider>
  )
})

export const ChatSettingsSection = memo(function ChatSettingsSection({
  controller,
}: ChatPageSectionProps) {
  const { ui, model, chat, clearChat } = controller
  const messages = useChatStore((s) => s.messages)
  const settings = useAppStore((state) => state.settings)
  const updateSettings = useAppStore((state) => state.updateSettings)
  // Effective value for THIS conversation: its AppScope delta over the global
  // default — the same per-field precedence the server applies at the turn
  // seam. The dialog edits the conversation instance, never the globals.
  const scope = useConversationScope((state) => state.scopes[chat.sessionIdRef.current])

  const handleTemperatureChange = useCallback(
    (temp: number) => {
      const sid = chat.sessionIdRef.current
      if (!sid) return
      saveConversationScope(sid, { temperature: temp }).catch((e) => {
        logger.warning('Could not conversation scope temperature save', { exception: String(e) })
      })
    },
    [chat],
  )

  const handleMaxTokensChange = useCallback(
    (tokens: number) => {
      const sid = chat.sessionIdRef.current
      if (!sid) return
      saveConversationScope(sid, { max_tokens: tokens }).catch((e) => {
        logger.warning('Could not conversation scope max_tokens save', { exception: String(e) })
      })
    },
    [chat],
  )

  const handleAutoApproveToolsChange = useCallback(
    (value: boolean) => {
      updateSettings({ autoApproveTools: value })
    },
    [updateSettings],
  )

  if (!ui.showSettings) return null
  return (
    <ChatSettings
      isOpen={true}
      model={model.model}
      temperature={scope?.temperature ?? settings.defaultTemp}
      maxTokens={scope?.max_tokens ?? settings.defaultMaxTokens}
      autoApproveTools={settings.autoApproveTools}
      onModelChange={model.setModel}
      availableModels={model.availableModels}
      onTemperatureChange={handleTemperatureChange}
      onMaxTokensChange={handleMaxTokensChange}
      onAutoApproveToolsChange={handleAutoApproveToolsChange}
      onClear={clearChat}
      hasMessages={messages.length > 0}
    />
  )
})

export const ChatChatSection = memo(function ChatChatSection({ controller }: ChatPageSectionProps) {
  // ── Zustand selectors for hot-path state (only re-renders when these change) ──
  const messages = useChatStore((s) => s.messages)
  const loading = useChatStore((s) => s.loading)
  const sessionLoading = useChatStore((s) => s.sessionLoading)
  const input = useChatStore((s) => s.input)
  const images = useChatStore((s) => s.images)
  const toolEvents = useChatStore((s) => s.toolEvents)
  const ragVerification = useChatStore((s) => s.ragVerification)
  const contextLayers = useChatStore((s) => s.contextLayers)
  const pendingToolApproval = useChatStore((s) => s.pendingToolApproval)
  const currentError = useChatStore((s) => s.currentError)
  const selectedMessageIds = useChatStore((s) => s.selectedMessageIds)
  const selectionMode = useChatStore((s) => s.selectionMode)

  // ── Controller for actions and non-hot-path state ──
  const {
    chat,
    ui,
    model,
    health,
    suggestions,
    refreshHealth,
    showToast,
    readFileData,
    setReadFileData,
    readLoading,
    handleReadFile,
    handleWriteSend,
    handleExecuteCommand,
    handleImageDropped,
    handleTextDropped,
    handlePDFDropped,
    isBookmarked,
    handleToggleBookmark,
    handleDeleteMessage,
    handleSaveToKnowledge,
    collapsibleLength,
  } = controller

  // ── Actions from store (stable references) ──
  const storeStop = useChatStore((s) => s.stop)
  const storeSetInput = useChatStore((s) => s.setInput)
  const storeSetMessages = useChatStore((s) => s.setMessages)
  const storeSetLoading = useChatStore((s) => s.setLoading)
  const storeSetCurrentError = useChatStore((s) => s.setCurrentError)
  const storeHandleToolApproval = useChatStore((s) => s.handleToolApproval)
  const storeHandleReact = useChatStore((s) => s.handleReact)
  const storeHandlePin = useChatStore((s) => s.handlePin)
  const storeToggleMessageSelection = useChatStore((s) => s.toggleMessageSelection)

  const handleStop = useCallback(() => {
    if (chat.loadingRef.current) {
      chat.loadingRef.current.abort()
    }
    storeStop()
  }, [chat, storeStop])

  const handleCancel = useCallback(async () => {
    const sessionId = chat.sessionIdRef.current
    if (sessionId) {
      const { chatController } = await import('@/lib/chat-controller')
      chatController.cancelStream(sessionId).catch(() => {})
    }
    handleStop()
  }, [chat, handleStop])

  const handleAudioRecorded = useCallback(
    async (blob: Blob) => {
      try {
        const { chatController } = await import('@/lib/chat-controller')
        const sessionId = chat.sessionIdRef.current
        if (!sessionId) return
        const result = await chatController.sendVoiceMessage(sessionId, blob)
        if (result.audio_path) {
          const audioUrl = chatController.getVoiceAudioUrl(sessionId, `voice-${Date.now()}`)
          storeSetMessages((prev) => [
            ...prev,
            {
              id: `voice-${Date.now()}`,
              role: 'user',
              content: result.transcript || '(voice message)',
              timestamp: new Date(),
              audio: {
                id: `audio-${Date.now()}`,
                url: audioUrl,
                durationMs: result.audio_duration_ms,
              },
            },
          ])
        }
      } catch (err) {
        showToast('Could not save voice message', 'error')
      }
    },
    [storeSetMessages, showToast],
  )

  const handleAudioTranscript = useCallback(
    (text: string) => {
      storeSetInput((prev) => (prev ? `${prev} ${text}` : text))
    },
    [storeSetInput],
  )

  const handleGeneratedImage = useCallback(
    (dataUrl: string, prompt: string) => {
      storeSetMessages((prev) => [
        ...prev,
        {
          id: `img-${Date.now()}`,
          role: 'user',
          content: `[Generate image: ${prompt}]`,
          timestamp: new Date(),
          images: [{ id: `gen-${Date.now()}`, dataUrl, name: 'generated.png' }],
        },
      ])
      showToast('Image generated — see message above', 'info')
    },
    [storeSetMessages, showToast],
  )

  const handlePDFError = useCallback(
    (error: string) => {
      showToast(`PDF analysis failed: ${error}`, 'error')
    },
    [showToast],
  )

  const handlePDFAnalysis = useCallback(
    (analysis: string, filename: string) => {
      storeSetMessages((prev) => [
        ...prev,
        {
          id: `pdf-user-${Date.now()}`,
          role: 'user',
          content: `📎 Uploaded PDF: ${filename}`,
          timestamp: new Date(),
        },
        {
          id: `pdf-${Date.now()}`,
          role: 'assistant',
          content: analysis,
          timestamp: new Date(),
        },
      ])
      showToast('PDF analyzed — see response below', 'info')
    },
    [storeSetMessages, showToast],
  )

  const handleDismissError = useCallback(() => {
    storeSetCurrentError(null)
  }, [storeSetCurrentError])

  const handleRemoveReadFile = useCallback(() => {
    setReadFileData(null)
    storeSetMessages((prev) => prev.filter((m) => !m.id.startsWith('file-')))
  }, [storeSetMessages, setReadFileData])

  const handleApproveTool = useCallback(
    (approved: boolean) => {
      storeHandleToolApproval(approved)
    },
    [storeHandleToolApproval],
  )

  return (
    <>
      {currentError && (
        <ErrorBanner
          error={currentError}
          onRetry={controller.chat.handleRetry}
          onDismiss={handleDismissError}
        />
      )}

      {/* File-QA attach flow — always available, never mode-gated. */}
      <ReadFileSection
        readLoading={readLoading}
        readFileData={readFileData}
        onFileSelected={handleReadFile}
        onRemove={handleRemoveReadFile}
      />

      {/* ChatStore is the single write path for input since card
          20261116_124020: the controller's setInput IS the store action, so
          one write here reaches every reader (composer, drafts, send). */}
      <ImageDropZone
        onImageDropped={handleImageDropped}
        onTextDropped={handleTextDropped}
        onPDFDropped={handlePDFDropped}
      >
        {loading && (
          <div className="px-4 py-2">
            <ContextInjectionBar onInject={controller.chat.injectContext} disabled={!loading} />
          </div>
        )}
        {pendingToolApproval && (
          <div className="px-4 py-2">
            <ToolApprovalDialog
              toolName={pendingToolApproval.toolName}
              args={pendingToolApproval.args}
              onApprove={handleApproveTool}
            />
          </div>
        )}
        <ChatArea
          messages={messages}
          loading={loading}
          sessionLoading={sessionLoading}
          model={model.model}
          health={health}
          suggestions={suggestions}
          onRefreshHealth={refreshHealth}
          onCopy={controller.chat.handleCopy}
          onRegenerate={controller.chat.handleRegenerate}
          onRegenerateWithOptions={controller.chat.handleRegenerateWithOptions}
          onThumbsUp={controller.chat.handleThumbsUp}
          onThumbsDown={controller.chat.handleThumbsDown}
          onEdit={controller.chat.handleEditMessage}
          searchQuery={ui.searchQuery}
          onSuggestionClick={controller.chat.handleSuggestionClick}
          toolEvents={toolEvents}
          streamingStatus={pendingToolApproval ? 'tool_call' : 'generating'}
          streamingToolName={pendingToolApproval?.toolName}
          ragVerification={ragVerification}
          value={input}
          onChange={(v: string) => storeSetInput(v)}
          onSend={handleWriteSend}
          onStop={handleStop}
          onCancel={handleCancel}
          images={images}
          onAddImage={controller.chat.handleAddImage}
          onRemoveImage={controller.chat.handleRemoveImage}
          onAudioRecorded={handleAudioRecorded}
          onAudioTranscript={handleAudioTranscript}
          onGeneratedImage={handleGeneratedImage}
          onPDFError={handlePDFError}
          onPDFAnalysis={handlePDFAnalysis}
          onExecuteCommand={handleExecuteCommand}
          isBookmarked={isBookmarked}
          onBookmark={handleToggleBookmark}
          onDelete={handleDeleteMessage}
          onSaveToKnowledge={handleSaveToKnowledge}
          onReact={storeHandleReact}
          onPin={storeHandlePin}
          collapsibleLength={collapsibleLength}
          temperature={model.temperature}
          contextLayers={contextLayers}
          noteMap={controller.noteMap}
          onAddNote={controller.onAddNote}
          selectionMode={selectionMode}
          selectedMessageIds={selectedMessageIds}
          onToggleSelection={storeToggleMessageSelection}
          hasThread={controller.hasThread}
          onThread={controller.onStartThread}
          onQuickReply={controller.onQuickReply}
        />
      </ImageDropZone>
    </>
  )
})

export const ChatSearchSection = memo(function ChatSearchSection({
  controller,
}: ChatPageSectionProps) {
  const { chat, ui } = controller
  const messages = useChatStore((s) => s.messages)
  const viewerMessages = useMemo(
    () =>
      messages.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        timestamp:
          typeof m.timestamp === 'number' ? m.timestamp : m.timestamp?.getTime() || Date.now(),
      })),
    [messages],
  )
  return (
    <>
      <ConversationViewer
        isOpen={ui.showConversationViewer}
        onClose={() => ui.setShowConversationViewer(false)}
        messages={viewerMessages}
        title="Current Conversation"
      />

      <ConversationSearch
        open={ui.showConversationSearch}
        onClose={() => ui.setShowConversationSearch(false)}
        onNavigate={(sessionId) => chat.loadSession(sessionId)}
      />
    </>
  )
})

export const ChatToolPanelInline = memo(function ChatToolPanelInline({
  controller,
}: ChatPageSectionProps) {
  const { ui, chat, bookmarks, removeBookmark, clearAll } = controller
  const messages = useChatStore((s) => s.messages)
  return (
    <ChatToolPanel
      open={ui.toolPanelOpen}
      onClose={() => ui.setToolPanelOpen(false)}
      sessionId={chat.sessionIdRef.current}
      bookmarks={bookmarks}
      onRemoveBookmark={removeBookmark}
      onClearBookmarks={clearAll}
      messages={messages}
    />
  )
})

export const ChatDialogSection = memo(function ChatDialogSection({
  controller,
}: ChatPageSectionProps) {
  const messages = useChatStore((s) => s.messages)
  const {
    ui,
    chat,
    model,
    bookmarks,
    removeBookmark,
    clearAll,
    systemPromptOpen,
    setSystemPromptOpen,
    customSystemPrompt,
    handleSaveSystemPrompt,
    noteDialogOpen,
    setNoteDialogOpen,
    noteDialogNote,
    onSaveNote,
    onDeleteNote,
    shortcutsOpen,
    setShortcutsOpen,
    templatesOpen,
    setTemplatesOpen,
    statsOpen,
    setStatsOpen,
    activeThreadMessageId,
    activeThread,
    activeThreadMessages,
    onStartThread,
    onReplyInThread,
    onCloseThread,
  } = controller

  return (
    <>
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
          onClose={() => {
            ui.setVoiceMode(false)
          }}
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

      {noteDialogOpen && (
        <NoteDialog
          open={noteDialogOpen}
          onOpenChange={setNoteDialogOpen}
          note={noteDialogNote}
          onSave={onSaveNote}
          onDelete={onDeleteNote}
        />
      )}

      {activeThreadMessageId && activeThread && (
        <ThreadPanel
          parentMessage={messages.find((m) => m.id === activeThreadMessageId)!}
          threadMessages={activeThreadMessages}
          onSend={(content) => onReplyInThread(activeThread.id, content)}
          onClose={onCloseThread}
          className="w-80"
        />
      )}

      <KeyboardShortcutsPanel open={shortcutsOpen} onClose={() => setShortcutsOpen(false)} />

      <TemplateDialog
        open={templatesOpen}
        onClose={() => setTemplatesOpen(false)}
        onSelect={(content) => {
          chat.sendMessage(content)
          setTemplatesOpen(false)
        }}
      />

      <ChatStatsPanel open={statsOpen} onClose={() => setStatsOpen(false)} messages={messages} />
    </>
  )
})
