'use client'

import { useState } from 'react'

import { cn, Button } from '@sloughgpt/strui'
import { IconX, IconEye, IconSettings, IconDocument, IconSparkle, IconCode, IconBolt, IconChart, IconDownload } from '@sloughgpt/strui'
import { useChatContext } from '@/features/chat/contexts/ChatContext'
import { KnowledgeTab } from './KnowledgeTab'
import { MemoryTab } from './MemoryTab'
import { ContextTab } from './ContextTab'
import { VisionTabContent } from './VisionTabContent'
import { QuickPrompts } from './../input/QuickPrompts'
import { ChatBookmarksPanel } from './ChatBookmarksPanel'
import { ChatSessionStatsCard } from './ChatSessionStatsCard'
import { ConversationSummary } from './../ConversationSummary'
import { ConversationStats } from './../ConversationStats'
import { ConversationExport } from './../ConversationExport'

interface ChatToolPanelProps {
  open: boolean
  onClose: () => void
  sessionId: string | null
  bookmarks?: import('@/features/chat/hooks/useChatBookmarks').BookmarkedMessage[]
  onRemoveBookmark?: (id: string) => void
  onClearBookmarks?: () => void
  messages?: import('@/lib/chat-utils').ChatMessage[]
}

export function ChatToolPanel({ open, onClose, sessionId, bookmarks = [], onRemoveBookmark, onClearBookmarks, messages = [] }: ChatToolPanelProps) {
  const [showVision, setShowVision] = useState(false)
  const ctx = useChatContext()

  return (
    <div
      id="chat-tool-panel"
      className={cn(
        'border-l border-border/50 bg-background overflow-hidden transition-all duration-200 flex flex-col',
        open ? 'w-[var(--tool-panel-width)] min-w-[var(--tool-panel-width)] lg:relative lg:w-[var(--tool-panel-width)] fixed right-0 top-0 bottom-0 z-50 shadow-xl lg:shadow-none' : 'w-0 min-w-0',
      )}
    >
      {open && (
        <>
          {/* ── Header ── */}
          <div className="flex items-center justify-between px-3 py-2 border-b border-border/50 shrink-0">
            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
              {showVision ? 'Vision' : 'Tools'}
            </span>
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => setShowVision(!showVision)}
                className={cn(
                  'h-6 w-6 flex items-center justify-center rounded hover:bg-muted/60 transition-colors',
                  showVision && 'text-primary',
                )}
                title={showVision ? 'Show tools' : 'Show vision'}
                aria-label={showVision ? 'Show tools panel' : 'Show vision panel'}
              >
                <IconEye className="h-3.5 w-3.5" />
              </button>
              <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose} aria-label="Close tools panel">
                <IconX className="h-3.5 w-3.5" />
              </Button>
            </div>
          </div>

          {/* ── Content ── */}
          <div className="flex-1 overflow-y-auto p-3 space-y-4 text-xs scrollbar-thin">
            {showVision ? (
              <VisionTabContent
                visionImagesLearned={ctx.visionCaps?.images_learned}
                visionTrained={ctx.visionCaps?.trained}
                visionStatus={ctx.visionCaps?.status}
                visionCaptionHistory={ctx.visionCaptionHistory}
                visionVocabSize={ctx.visionVocabSize}
                sessionId={sessionId}
                onGeneratedImage={(dataUrl, prompt) => {
                  const event = new CustomEvent('insert-generated-image', { detail: { dataUrl, prompt } })
                  window.dispatchEvent(event)
                }}
                onSendText={(text) => {
                  window.dispatchEvent(new CustomEvent('send-text', { detail: { text } }))
                }}
              />
            ) : (
              <>
                <ChatSessionStatsCard sessionId={sessionId} />
                <section aria-label="Knowledge">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconDocument className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Knowledge</span>
                  </div>
                  <KnowledgeTab
                    onOpenConversationViewer={ctx.onOpenConversationViewer}
                    onOpenSettings={ctx.onOpenSettings}
                    onOpenShortcuts={ctx.onOpenShortcuts}
                  />
                </section>
                <section aria-label="Memory">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconSparkle className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Memory</span>
                  </div>
                  <MemoryTab />
                </section>
                <section aria-label="Context">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconCode className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Context</span>
                  </div>
                  <ContextTab />
                </section>
                <section aria-label="Quick Prompts">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconBolt className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Quick Prompts</span>
                  </div>
                  <QuickPrompts onUsePrompt={(text) => ctx.setInput(text)} />
                </section>
                <section aria-label="Summary">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconDocument className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Summary</span>
                  </div>
                  <ConversationSummary messages={messages} model={ctx.model} />
                </section>
                <section aria-label="Statistics">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconChart className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Statistics</span>
                  </div>
                  <ConversationStats messages={messages} />
                </section>
                <section aria-label="Export">
                  <div className="flex items-center gap-1.5 mb-2">
                    <IconDownload className="h-3 w-3 text-muted-foreground" />
                    <span className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider">Export</span>
                  </div>
                  <ConversationExport messages={messages} model={ctx.model} />
                </section>
                <section aria-label="Bookmarks">
                  <ChatBookmarksPanel
                    bookmarks={bookmarks}
                    onRemove={onRemoveBookmark || (() => {})}
                    onClear={onClearBookmarks || (() => {})}
                    onJumpToMessage={(id) => {
                      const el = document.getElementById(`msg-${id}`)
                      el?.scrollIntoView({ behavior: 'smooth', block: 'center' })
                    }}
                  />
                </section>
              </>
            )}
          </div>

          {/* ── Footer actions ── */}
          <div className="border-t border-border/50 p-2 flex gap-1 shrink-0">
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={ctx.onOpenConversationViewer}>
              <IconEye className="h-3 w-3 mr-1" />
              Log
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={ctx.onOpenSettings}>
              <IconSettings className="h-3 w-3 mr-1" />
              Settings
            </Button>
            <Button variant="ghost" size="sm" className="text-[10px] h-7 flex-1" onClick={ctx.onOpenShortcuts}>
              <IconBolt className="h-3 w-3 mr-1" />
              Keys
            </Button>
          </div>
        </>
      )}
    </div>
  )
}
