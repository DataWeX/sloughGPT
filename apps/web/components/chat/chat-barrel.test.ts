import { describe, it, expect } from 'vitest'

import * as chatBarrel from './index'
import { ChatSettings } from './ChatSettings'
import { ChatScreen } from './ChatScreen'
import { MessageBubble } from './MessageBubble'
import { MessageActions } from './MessageActions'
import { ChatInput } from './ChatInput'
import { EmptyState } from './EmptyState'
import { ToastContainer } from './Toast'
import { ErrorBanner } from './ErrorBanner'
import { SystemBanner } from './SystemBanner'
import { VoiceInput } from './VoiceInput'
import { ImageUpload } from './ImageUpload'
import { Markdown } from './Markdown'
import { ChatInputRow } from './ChatInputRow'
import { ChatInputField } from './ChatInputField'
import { ChatInputAccessories } from './ChatInputAccessories'
import { ChatSendButton } from './ChatSendButton'
import { ChatSearchBar } from './ChatSearchBar'
import { ChatArea } from './ChatArea'
import { ConversationViewer } from './ConversationViewer'
import { SoulSelectorDropdown } from './SoulSelectorDropdown'
import { ChatMoreMenu } from './ChatMoreMenu'
import { DownloadDialog } from './DownloadDialog'

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
