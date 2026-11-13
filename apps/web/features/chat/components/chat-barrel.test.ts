import { describe, it, expect } from 'vitest'

import * as chatBarrel from './index'
import { ChatSettings } from './dialogs/ChatSettings'
import { ChatScreen } from './layout/ChatScreen'
import { MessageBubble } from './messages/MessageBubble'
import { MessageActions } from './messages/MessageActions'
import { ChatInput } from './input/ChatInput'
import { EmptyState } from './messages/EmptyState'
import { ToastContainer } from './feedback/Toast'
import { ErrorBanner } from './feedback/ErrorBanner'
import { SystemBanner } from './messages/SystemBanner'
import { VoiceInput } from './input/VoiceInput'
import { ImageUpload } from './input/ImageUpload'
import { Markdown } from './messages/Markdown'
import { ChatInputRow } from './input/ChatInputRow'
import { ChatInputField } from './input/ChatInputField'
import { ChatInputAccessories } from './input/ChatInputAccessories'
import { ChatSendButton } from './input/ChatSendButton'
import { ChatSearchBar } from './toolbar/ChatSearchBar'
import { ChatArea } from './layout/ChatArea'
import { ConversationViewer } from './sidebar/ConversationViewer'
import { SoulSelectorDropdown } from './toolbar/SoulSelectorDropdown'
import { ChatMoreMenu } from './toolbar/ChatMoreMenu'
import { DownloadDialog } from './dialogs/DownloadDialog'

describe('chat barrel (components/chat/index.ts)', () => {
  it('re-exports every component with identity', () => {
    expect(chatBarrel.ChatSettings).toBe(ChatSettings)
    expect(chatBarrel.ChatScreen).toBe(ChatScreen)
    expect(chatBarrel.MessageBubble).toBe(MessageBubble)
    expect(chatBarrel.MessageActions).toBe(MessageActions)
    expect(chatBarrel.ChatInput).toBe(ChatInput)
    expect(chatBarrel.EmptyState).toBe(EmptyState)
    expect(chatBarrel.ToastContainer).toBe(ToastContainer)
    expect(chatBarrel.ErrorBanner).toBe(ErrorBanner)
    expect(chatBarrel.SystemBanner).toBe(SystemBanner)
    expect(chatBarrel.VoiceInput).toBe(VoiceInput)
    expect(chatBarrel.ImageUpload).toBe(ImageUpload)
    expect(chatBarrel.Markdown).toBe(Markdown)
    expect(chatBarrel.ChatInputRow).toBe(ChatInputRow)
    expect(chatBarrel.ChatInputField).toBe(ChatInputField)
    expect(chatBarrel.ChatInputAccessories).toBe(ChatInputAccessories)
    expect(chatBarrel.ChatSendButton).toBe(ChatSendButton)
    expect(chatBarrel.ChatSearchBar).toBe(ChatSearchBar)
    expect(chatBarrel.ChatArea).toBe(ChatArea)
    expect(chatBarrel.ConversationViewer).toBe(ConversationViewer)
    expect(chatBarrel.SoulSelectorDropdown).toBe(SoulSelectorDropdown)
    expect(chatBarrel.ChatMoreMenu).toBe(ChatMoreMenu)
    expect(chatBarrel.DownloadDialog).toBe(DownloadDialog)
  })

  it('all value exports are components (functions)', () => {
    const valueExports = [
      'ChatSettings', 'ChatScreen', 'MessageBubble', 'MessageActions', 'ChatInput',
      'EmptyState', 'ToastContainer', 'ErrorBanner', 'getErrorInfo', 'SystemBanner',
      'VoiceInput', 'ImageUpload', 'ImagePreview', 'Markdown', 'ChatInputRow',
      'ChatInputField', 'ChatInputAccessories', 'ChatSendButton', 'ChatSearchBar',
      'ChatArea', 'ConversationViewer', 'SoulSelectorDropdown', 'ChatMoreMenu', 'DownloadDialog',
    ]
    for (const name of valueExports) {
      const value = (chatBarrel as Record<string, unknown>)[name]
      expect(value != null, `export ${name}`).toBe(true)
      expect(['function', 'object'], `export ${name}`).toContain(typeof value)
    }
  })

  it('does not accidentally rename the exported MessageActions', () => {
    const name = chatBarrel.MessageActions.name
    expect(!name || name === 'MessageActions' || name === 'ForwardRef', `unexpected name: ${name}`).toBe(true)
  })
})
