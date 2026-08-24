'use client'

import dynamicNext from 'next/dynamic'

import type { ChatPageController } from '@/features/chat/hooks/useChatPageController'
import { generationConfigController } from '@/lib/generation-config-controller'
import { ChatArea, ErrorBanner } from '@/features/chat/components'
import { ImageDropZone } from '@/features/chat/components/layout/ImageDropZone'
import { ModeBar } from '@/features/chat/components/toolbar/ModeBar'
import { ChatToolbar } from '@/features/chat/components/toolbar/ChatToolbar'
import { logger } from '@/lib/dev-log'
import { ChatToolbarProvider } from '@/features/chat/contexts/ChatToolbarContext'

const VoiceChatMode = dynamicNext(() => import('@/features/chat/components/input/VoiceChatMode').then(m => m.VoiceChatMode), { ssr: false })
const ConversationViewer = dynamicNext(() => import('@/features/chat/components/sidebar/ConversationViewer').then(m => m.ConversationViewer), { ssr: false })
const ConversationSearch = dynamicNext(() => import('@/features/chat/components/sidebar/ConversationSearch').then(m => m.ConversationSearch), { ssr: false })
const ChatSettings = dynamicNext(() => import('@/features/chat/components/dialogs/ChatSettings').then(m => m.ChatSettings), { ssr: false })
const ConversationSidebar = dynamicNext(() => import('@/features/chat/components/sidebar/ConversationSidebar').then(m => m.ConversationSidebar), { ssr: false })
const ChatToolPanel = dynamicNext(() => import('@/features/chat/components/panels/ChatToolPanel').then(m => m.ChatToolPanel), { ssr: false })
const DownloadDialog = dynamicNext(() => import('@/features/chat/components/dialogs/DownloadDialog').then(m => m.DownloadDialog), { ssr: false })
const SystemPromptDialog = dynamicNext(() => import('@/features/chat/components/dialogs/SystemPromptDialog').then(m => m.SystemPromptDialog), { ssr: false })
const ReadFileSection = dynamicNext(() => import('@/features/chat/components/dialogs/ReadFileSection'), { ssr: false })

interface ChatPageSectionProps {
  controller: ChatPageController
}

export function ChatSidebarSection({ controller }: ChatPageSectionProps) {
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
}

export function ChatToolbarSection({ controller }: ChatPageSectionProps) {
  return (
    <ChatToolbarProvider value={controller.toolbarValue}>
      <ChatToolbar />
    </ChatToolbarProvider>
  )
}

export function ChatSettingsSection({ controller }: ChatPageSectionProps) {
  const { ui, model, chat, clearChat } = controller
  if (!ui.showSettings) return null
  return (
    <ChatSettings
      isOpen={true}
      model={model.model}
      temperature={model.temperature}
      maxTokens={model.maxTokens}
      onModelChange={model.setModel}
      availableModels={model.availableModels}
      onTemperatureChange={(temp) => {
        model.setTemperature(temp)
        generationConfigController.update({ temperature: temp }).catch(e => { logger.warning('Could not generation config temperature save', { exception: String(e) }) })
      }}
      onMaxTokensChange={(tokens) => {
        model.setMaxTokens(tokens)
        generationConfigController.update({ max_new_tokens: tokens }).catch(e => { logger.warning('Could not generation config max_tokens save', { exception: String(e) }) })
      }}
      onClear={clearChat}
      hasMessages={chat.messages.length > 0}
    />
  )
}

export function ChatChatSection({ controller }: ChatPageSectionProps) {
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
  } = controller

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
          ragVerification={chat.ragVerification}
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
          onSaveToKnowledge={handleSaveToKnowledge}
          collapsibleLength={collapsibleLength}
        />
      </ImageDropZone>
    </>
  )
}

export function ChatSearchSection({ controller }: ChatPageSectionProps) {
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
}

export function ChatDialogSection({ controller }: ChatPageSectionProps) {
  const {
    ui, chat, model, bookmarks, removeBookmark, clearAll,
    systemPromptOpen, setSystemPromptOpen, customSystemPrompt, handleSaveSystemPrompt,
    setChatMode,
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
    </>
  )
}
