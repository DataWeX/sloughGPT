'use client'
export const dynamic = 'force-dynamic'

import { useCallback, useState } from 'react'

import { modelController } from '@/lib/controllers'
import { useToastStore } from '@/lib/toast-store'
import { useChatPageController } from '@/features/chat/hooks/useChatPageController'
import { ChatProvider } from '@/features/chat/contexts/ChatContext'
import {
  ChatSidebarSection,
  ChatToolbarSection,
  ChatSettingsSection,
  ChatChatSection,
  ChatSearchSection,
  ChatDialogSection,
  ChatToolPanelInline,
} from '@/features/chat/ChatPageSections'
import { ConsciousnessChatPanel } from '@/features/chat/components/ConsciousnessChatPanel'

export default function ChatPage() {
  const [consciousnessOpen, setConsciousnessOpen] = useState(false)

  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info' = 'success') => {
    const store = useToastStore.getState()
    const exists = store.toasts.some(t => t.message === message && t.type === type)
    if (!exists) store.addToast(message, type)
  }, [])

  const refreshHealth = useCallback(async () => {
    await modelController.getHealth()
  }, [])

  const controller = useChatPageController(showToast, refreshHealth)
  const { healthValue, modelValue, uiValue } = controller

  return (
    <ChatProvider health={healthValue} model={modelValue} ui={uiValue}>
    <a href="#chat-messages" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:px-4 focus:py-2 focus:bg-background focus:border focus:rounded-lg focus:shadow-lg">
      Skip to messages
    </a>
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <ChatSidebarSection controller={controller} />
      <main className="flex flex-1 min-h-0 overflow-hidden rounded-none lg:rounded-xl border border-border/40 bg-[rgb(var(--chat-bg))] shadow-sm relative" aria-label="Chat">
        <div className="flex flex-col flex-1 min-h-0 min-w-0 max-w-full overflow-hidden">
          <ChatToolbarSection
            controller={controller}
            consciousnessOpen={consciousnessOpen}
            onConsciousnessToggle={() => setConsciousnessOpen(v => !v)}
          />
          <ChatSettingsSection controller={controller} />
          <ChatChatSection controller={controller} />
          <ChatSearchSection controller={controller} />
        </div>
        <ChatToolPanelInline controller={controller} />
        <ConsciousnessChatPanel
          open={consciousnessOpen}
          onClose={() => setConsciousnessOpen(false)}
        />
      </main>
      <ChatDialogSection controller={controller} />
    </div>
    </ChatProvider>
  )
}
