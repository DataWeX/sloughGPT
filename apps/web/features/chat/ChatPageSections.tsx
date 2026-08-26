'use client'

import dynamicNext from 'next/dynamic'
import { memo, useCallback, useMemo } from 'react'

import type { ChatPageController } from '@/features/chat/hooks/useChatPageController'
import { generationConfigController } from '@/lib/generation-config-controller'
import { ChatArea, ErrorBanner } from '@/features/chat/components'
import { ContextInjectionBar } from '@/features/chat/components/ContextInjectionBar'
import { ImageDropZone } from '@/features/chat/components/layout/ImageDropZone'
import { ModeBar } from '@/features/chat/components/toolbar/ModeBar'
import { ChatToolbar } from '@/features/chat/components/toolbar/ChatToolbar'
import { logger } from '@/lib/dev-log'
import { ChatToolbarProvider } from '@/features/chat/contexts/ChatToolbarContext'
import { useAppStore } from '@/lib/store'

const VoiceChatMode = dynamicNext(() => import('@/features/chat/components/input/VoiceChatMode').then(m => m.VoiceChatMode), { ssr: false })
const ConversationViewer = dynamicNext(() => import('@/features/chat/components/sidebar/ConversationViewer').then(m => m.ConversationViewer), { ssr: false })
const ConversationSearch = dynamicNext(() => import('@/features/chat/components/sidebar/ConversationSearch').then(m => m.ConversationSearch), { ssr: false })
const ChatSettings = dynamicNext(() => import('@/features/chat/components/dialogs/ChatSettings').then(m => m.ChatSettings), { ssr: false })
const ConversationSidebar = dynamicNext(() => import('@/features/chat/components/sidebar/ConversationSidebar').then(m => m.ConversationSidebar), { ssr: false })
const ChatToolPanel = dynamicNext(() => import('@/features/chat/components/panels/ChatToolPanel').then(m => m.ChatToolPanel), { ssr: false })
const DownloadDialog = dynamicNext(() => import('@/features/chat/components/dialogs/DownloadDialog').then(m => m.DownloadDialog), { ssr: false })
const SystemPromptDialog = dynamicNext(() => import('@/features/chat/components/dialogs/SystemPromptDialog').then(m => m.SystemPromptDialog), { ssr: false })
const ReadFileSection = dynamicNext(() => import('@/features/chat/components/dialogs/ReadFileSection'), { ssr: false })
const NoteDialog = dynamicNext(() => import('@/features/chat/components/dialogs/NoteDialog').then(m => m.NoteDialog), { ssr: false })
const ThreadPanel = dynamicNext(() => import('@/features/chat/components/ThreadPanel').then(m => m.ThreadPanel), { ssr: false })
const KeyboardShortcutsPanel = dynamicNext(() => import('@/features/chat/components/dialogs/KeyboardShortcutsPanel').then(m => m.KeyboardShortcutsPanel), { ssr: false })
const TemplateDialog = dynamicNext(() => import('@/features/chat/components/dialogs/TemplateDialog').then(m => m.TemplateDialog), { ssr: false })
const ChatStatsPanel = dynamicNext(() => import('@/features/chat/components/dialogs/ChatStatsPanel').then(m => m.ChatStatsPanel), { ssr: false })

interface ChatPageSectionProps {
  controller: ChatPageController
}

export const ChatSidebarSection = memo(function ChatSidebarSection({ controller }: ChatPageSectionProps) {
  const { chat, ui, convCollapsed, toggleConv } = controller
  return (
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
      onDuplicateConversation={(id) => chat.duplicateSession(id)}
      open={ui.sidebarOpen}
      onClose={() => ui.setSidebarOpen(false)}
      collapsed={convCollapsed}
      onToggleCollapse={toggleConv}
    />
  )
})

export const ChatToolbarSection = memo(function ChatToolbarSection({ controller }: ChatPageSectionProps) {
  return (
    <ChatToolbarProvider value={controller.toolbarValue}>
      <ChatToolbar />
    </ChatToolbarProvider>
  )
})

export const ChatSettingsSection = memo(function ChatSettingsSection({ controller }: ChatPageSectionProps) {
  const { ui, model, chat, clearChat } = controller
  const settings = useAppStore(state => state.settings)
  const updateSettings = useAppStore(state => state.updateSettings)

  const handleTemperatureChange = useCallback((temp: number) => {
    model.setTemperature(temp)
    generationConfigController.update({ temperature: temp }).catch(e => { logger.warning('Could not generation config temperature save', { exception: String(e) }) })
  }, [model])

  const handleMaxTokensChange = useCallback((tokens: number) => {
    model.setMaxTokens(tokens)
    generationConfigController.update({ max_new_tokens: tokens }).catch(e => { logger.warning('Could not generation config max_tokens save', { exception: String(e) }) })
  }, [model])

  const handleAutoApproveToolsChange = useCallback((value: boolean) => {
    updateSettings({ autoApproveTools: value })
  }, [updateSettings])

  if (!ui.showSettings) return null
  return (
    <ChatSettings
      isOpen={true}
      model={model.model}
      temperature={model.temperature}
      maxTokens={model.maxTokens}
      autoApproveTools={settings.autoApproveTools}
      onModelChange={model.setModel}
      availableModels={model.availableModels}
      onTemperatureChange={handleTemperatureChange}
      onMaxTokensChange={handleMaxTokensChange}
      onAutoApproveToolsChange={handleAutoApproveToolsChange}
      onClear={clearChat}
      hasMessages={chat.messages.length > 0}
    />
  )
})

export const ChatChatSection = memo(function ChatChatSection({ controller }: ChatPageSectionProps) {
  const {
    chat, ui, model, health, suggestions, refreshHealth, showToast,
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
    readFileData, setReadFileData, readLoading, handleReadFile,
    handleWriteSend,
    handleExecuteCommand,
    handleImageDropped, handleTextDropped, handlePDFDropped,
    isBookmarked, handleToggleBookmark, handleDeleteMessage, handleSaveToKnowledge,
    collapsibleLength,
    contextLayers,
    handleReact,
    handlePin,
  } = controller

  const handleStop = useCallback(() => {
    if (chat.loadingRef.current) {
      chat.loadingRef.current.abort()
    }
    chat.setLoading(false)
  }, [chat])

  const handleCancel = useCallback(async () => {
    const sessionId = chat.sessionIdRef.current
    if (sessionId) {
      const { chatController } = await import('@/lib/chat-controller')
      chatController.cancelStream(sessionId).catch(() => {})
    }
    handleStop()
  }, [chat, handleStop])

  const handleAudioRecorded = useCallback(async (blob: Blob) => {
    try {
      const { chatController } = await import('@/lib/chat-controller')
      const sessionId = chat.sessionIdRef.current
      if (!sessionId) return
      const result = await chatController.sendVoiceMessage(sessionId, blob)
      if (result.audio_path) {
        const audioUrl = chatController.getVoiceAudioUrl(sessionId, `voice-${Date.now()}`)
        chat.setMessages(prev => [...prev, {
          id: `voice-${Date.now()}`,
          role: 'user',
          content: result.transcript || '(voice message)',
          timestamp: new Date(),
          audio: {
            id: `audio-${Date.now()}`,
            url: audioUrl,
            durationMs: result.audio_duration_ms,
          },
        }])
      }
    } catch (err) {
      showToast('Could not save voice message', 'error')
    }
  }, [chat, showToast])

  const handleAudioTranscript = useCallback((text: string) => {
    chat.setInput(prev => prev ? `${prev} ${text}` : text)
  }, [chat])

  const handleGeneratedImage = useCallback((dataUrl: string, prompt: string) => {
    chat.setMessages(prev => [...prev, {
      id: `img-${Date.now()}`, role: 'user',
      content: `[Generate image: ${prompt}]`, timestamp: new Date(),
      images: [{ id: `gen-${Date.now()}`, dataUrl, name: 'generated.png' }],
    }])
    showToast('Image generated — see message above', 'info')
  }, [chat, showToast])

  const handlePDFError = useCallback((error: string) => {
    showToast(`PDF analysis failed: ${error}`, 'error')
  }, [showToast])

  const handlePDFAnalysis = useCallback((analysis: string, filename: string) => {
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
  }, [chat, showToast])

  return (
    <>
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
        rewriteStyle={rewriteStyle}
        decideStructure={decideStructure}
        difficulty={explainDifficulty}
        langPair={translateLangPair}
        brainstormTopic={brainstormTopic}
        wellnessType={wellnessType}
        createStyle={createStyle}
        onModeChange={setChatMode}
        onToneChange={setWriteTone}
        onTypeChange={setWriteType}
        onRewriteStyleChange={setRewriteStyle}
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

      <ImageDropZone
        onImageDropped={handleImageDropped}
        onTextDropped={handleTextDropped}
        onPDFDropped={handlePDFDropped}
      >
        {chat.loading && (
          <div className="px-4 py-2">
            <ContextInjectionBar
              onInject={chat.injectContext}
              disabled={!chat.loading}
            />
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
          onRegenerateWithOptions={chat.handleRegenerateWithOptions}
          onThumbsUp={chat.handleThumbsUp}
          onThumbsDown={chat.handleThumbsDown}
          onEdit={chat.handleEditMessage}
          searchQuery={ui.searchQuery}
          onSuggestionClick={chat.handleSuggestionClick}
          toolEvents={chat.toolEvents}
          streamingStatus={chat.pendingToolApproval ? 'tool_call' : 'generating'}
          streamingToolName={chat.pendingToolApproval?.toolName}
          ragVerification={chat.ragVerification}
          value={chat.input}
          onChange={chat.setInput}
          onSend={handleWriteSend}
          onStop={handleStop}
          onCancel={handleCancel}
          images={chat.images}
          onAddImage={chat.handleAddImage}
          onRemoveImage={chat.handleRemoveImage}
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
          onReact={handleReact}
          onPin={handlePin}
          collapsibleLength={collapsibleLength}
          temperature={model.temperature}
          contextLayers={contextLayers}
          noteMap={controller.noteMap}
          onAddNote={controller.onAddNote}
          selectionMode={controller.selectionMode}
          selectedMessageIds={controller.selectedMessageIds}
          onToggleSelection={controller.toggleMessageSelection}
          hasThread={controller.hasThread}
          onThread={controller.onStartThread}
        />
      </ImageDropZone>
    </>
  )
})

export const ChatSearchSection = memo(function ChatSearchSection({ controller }: ChatPageSectionProps) {
  const { chat, ui } = controller
  return (
    <>
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
    </>
  )
})

export const ChatDialogSection = memo(function ChatDialogSection({ controller }: ChatPageSectionProps) {
  const {
    ui, chat, model, bookmarks, removeBookmark, clearAll,
    systemPromptOpen, setSystemPromptOpen, customSystemPrompt, handleSaveSystemPrompt,
    setChatMode,
    noteDialogOpen, setNoteDialogOpen, noteDialogNote, onSaveNote, onDeleteNote,
    shortcutsOpen, setShortcutsOpen,
    templatesOpen, setTemplatesOpen,
    statsOpen, setStatsOpen,
    activeThreadMessageId, activeThread, activeThreadMessages, onStartThread, onReplyInThread, onCloseThread,
  } = controller

  return (
    <>
      {ui.toolPanelOpen && (
        <ChatToolPanel
          open={true}
          onClose={() => ui.setToolPanelOpen(false)}
          sessionId={chat.sessionIdRef.current}
          bookmarks={bookmarks}
          onRemoveBookmark={removeBookmark}
          onClearBookmarks={clearAll}
          messages={chat.messages}
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
          parentMessage={chat.messages.find(m => m.id === activeThreadMessageId)!}
          threadMessages={activeThreadMessages}
          onSend={(content) => onReplyInThread(activeThread.id, content)}
          onClose={onCloseThread}
          className="w-80"
        />
      )}

      <KeyboardShortcutsPanel
        open={shortcutsOpen}
        onClose={() => setShortcutsOpen(false)}
      />

      <TemplateDialog
        open={templatesOpen}
        onClose={() => setTemplatesOpen(false)}
        onSelect={(content) => {
          chat.sendMessage(content)
          setTemplatesOpen(false)
        }}
      />

      <ChatStatsPanel
        open={statsOpen}
        onClose={() => setStatsOpen(false)}
        messages={chat.messages}
      />
    </>
  )
})
