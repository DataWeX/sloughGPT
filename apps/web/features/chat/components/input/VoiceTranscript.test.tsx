import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...a: any[]) => a.filter(Boolean).join(' '),
}))

import { VoiceTranscript } from './VoiceTranscript'
import type { VoiceExchange } from '@/features/chat/hooks/useVoiceChat'

const baseExchange: VoiceExchange = {
  id: '1',
  userText: 'Hello',
  assistantText: 'Hi there!',
  timestamp: Date.now(),
}

afterEach(cleanup)

describe('VoiceTranscript', () => {
  it('renders without crashing with empty conversation', () => {
    const { container } = render(<VoiceTranscript conversation={[]} responseText="" isSpeaking={false} />)
    expect(container.firstChild).toBeDefined()
  })

  it('renders user and assistant text for each exchange', () => {
    const conversation: VoiceExchange[] = [
      baseExchange,
      { ...baseExchange, id: '2', userText: 'How are you?', assistantText: 'I am fine.' },
    ]
    render(<VoiceTranscript conversation={conversation} responseText="" isSpeaking={false} />)
    expect(screen.getByText('Hello')).toBeDefined()
    expect(screen.getByText('Hi there!')).toBeDefined()
    expect(screen.getByText('How are you?')).toBeDefined()
    expect(screen.getByText('I am fine.')).toBeDefined()
  })

  it('shows final responseText when not speaking', () => {
    render(
      <VoiceTranscript
        conversation={[]}
        responseText="This is the final answer"
        isSpeaking={false}
      />,
    )
    expect(screen.getByText('This is the final answer')).toBeDefined()
  })

  it('hides final responseText while speaking', () => {
    render(
      <VoiceTranscript
        conversation={[]}
        responseText="Still generating..."
        isSpeaking={true}
      />,
    )
    expect(screen.queryByText('Still generating...')).toBeNull()
  })

  it('forwards ref correctly', () => {
    const ref = { current: null }
    render(
      <VoiceTranscript
        conversation={[]}
        responseText=""
        isSpeaking={false}
        ref={ref as any}
      />,
    )
    expect(ref.current).toBeInstanceOf(HTMLDivElement)
  })
})
