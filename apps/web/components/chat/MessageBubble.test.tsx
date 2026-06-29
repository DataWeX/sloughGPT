/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeAll } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { MessageBubble } from './MessageBubble'

beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
})

vi.mock('./Markdown', () => ({
  Markdown: ({ content, className }: { content: string; className?: string }) => (
    <div data-testid="markdown" className={className}>{content}</div>
  ),
}))

vi.mock('./MessageActions', () => ({
  MessageActions: (props: Record<string, unknown>) => (
    <div data-testid="message-actions" data-props={JSON.stringify(props)} />
  ),
}))

afterEach(cleanup)

describe('formatTime', () => {
  it('returns "just now" for recent timestamp', () => {
    const { container } = render(
      <MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp />
    )
    expect(container.textContent).toContain('just now')
  })

  it('returns "Xm ago" for minutes-old timestamp', () => {
    const past = new Date(Date.now() - 5 * 60000)
    const { container } = render(
      <MessageBubble content="hi" role="assistant" timestamp={past} showTimestamp />
    )
    expect(container.textContent).toContain('5m ago')
  })

  it('returns localized time for older messages', () => {
    const past = new Date(Date.now() - 120 * 60000)
    const { container } = render(
      <MessageBubble content="hi" role="assistant" timestamp={past} showTimestamp />
    )
    expect(container.textContent).not.toContain('just now')
    expect(container.textContent).not.toContain('m ago')
  })
})

describe('highlightText', () => {
  it('renders original text when no search query', () => {
    const { container } = render(
      <MessageBubble content="hello world" role="user" timestamp={new Date()} showTimestamp />
    )
    expect(container.textContent).toContain('hello world')
  })

  it('renders highlight mark for matching query', () => {
    const { container } = render(
      <MessageBubble content="hello world testing" role="user" timestamp={new Date()} searchQuery="world" showTimestamp />
    )
    expect(container.querySelector('mark')).toBeInTheDocument()
  })

  it('highlights case-insensitive', () => {
    const { container } = render(
      <MessageBubble content="Hello World" role="user" timestamp={new Date()} searchQuery="hello" showTimestamp />
    )
    const marks = container.querySelectorAll('mark')
    expect(marks.length).toBeGreaterThan(0)
  })
})

describe('MessageBubble', () => {
  it('renders user message content', () => {
    render(<MessageBubble content="user text" role="user" timestamp={new Date()} showTimestamp />)
    expect(screen.getByText('user text')).toBeInTheDocument()
  })

  it('renders assistant message via Markdown', () => {
    render(<MessageBubble content="assistant text" role="assistant" timestamp={new Date()} showTimestamp />)
    expect(screen.getByTestId('markdown')).toHaveTextContent('assistant text')
  })

  it('has accessible label for user role', () => {
    const { container } = render(<MessageBubble content="hi" role="user" timestamp={new Date()} showTimestamp />)
    const article = container.querySelector('[role="article"]')
    expect(article).toHaveAttribute('aria-label', 'Message from You')
  })

  it('has accessible label for assistant role', () => {
    const { container } = render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp />)
    const article = container.querySelector('[role="article"]')
    expect(article).toHaveAttribute('aria-label', 'Message from Assistant')
  })

  it('shows role indicator "You" for user', () => {
    render(<MessageBubble content="hi" role="user" timestamp={new Date()} showTimestamp />)
    expect(screen.getByText('You')).toBeInTheDocument()
  })

  it('shows role indicator "Assistant" for assistant', () => {
    render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp />)
    expect(screen.getByText('Assistant')).toBeInTheDocument()
  })

  it('shows model name when provided', () => {
    render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} model="gpt2" showTimestamp />)
    expect(screen.getByText('gpt2')).toBeInTheDocument()
  })

  it('does not show model name for user messages', () => {
    render(<MessageBubble content="hi" role="user" timestamp={new Date()} model="gpt2" showTimestamp />)
    expect(screen.queryByText('gpt2')).not.toBeInTheDocument()
  })

  it('shows timestamp when showTimestamp is true', () => {
    const { container } = render(<MessageBubble content="hi" role="user" timestamp={new Date()} showTimestamp />)
    expect(container.textContent).toContain('just now')
  })

  it('hides timestamp when showTimestamp is false', () => {
    const { container } = render(<MessageBubble content="hi" role="user" timestamp={new Date()} showTimestamp={false} />)
    expect(container.textContent).not.toContain('just now')
  })

  it('shows loading dots when content is empty and assistant', () => {
    const { container } = render(<MessageBubble content="" role="assistant" timestamp={new Date()} showTimestamp />)
    const dots = container.querySelector('.animate-bounce')
    expect(dots).toBeInTheDocument()
  })

  it('shows streaming cursor when isStreaming', () => {
    const { container } = render(<MessageBubble content="streaming" role="assistant" timestamp={new Date()} showTimestamp isStreaming />)
    expect(container.textContent).toContain('▊')
  })

  it('applies polite aria-live when streaming', () => {
    const { container } = render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp isStreaming />)
    const article = container.querySelector('[role="article"]')
    expect(article).toHaveAttribute('aria-live', 'polite')
  })

  it('sets id attribute when messageId provided', () => {
    const { container } = render(<MessageBubble content="hi" role="user" timestamp={new Date()} messageId="abc" showTimestamp />)
    expect(container.querySelector('#msg-abc')).toBeInTheDocument()
  })

  it('renders MessageActions for assistant when content exists and not streaming', () => {
    render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp onCopy={vi.fn()} />)
    expect(screen.getByTestId('message-actions')).toBeInTheDocument()
  })

  it('does not render MessageActions for assistant when streaming', () => {
    render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp onCopy={vi.fn()} isStreaming />)
    expect(screen.queryByTestId('message-actions')).not.toBeInTheDocument()
  })

  it('does not render MessageActions for assistant with empty content', () => {
    render(<MessageBubble content="" role="assistant" timestamp={new Date()} showTimestamp onCopy={vi.fn()} />)
    expect(screen.queryByTestId('message-actions')).not.toBeInTheDocument()
  })

  it('renders MessageActions for user with onEdit', () => {
    render(<MessageBubble content="hi" role="user" timestamp={new Date()} showTimestamp onEdit={vi.fn()} />)
    expect(screen.getByTestId('message-actions')).toBeInTheDocument()
  })

  it('renders images when provided', () => {
    const images = [{ id: 'img1', dataUrl: 'data:image/png;base64,abc', name: 'test.png', file: new File([], 'test.png') }]
    const { container } = render(
      <MessageBubble content="with image" role="user" timestamp={new Date()} images={images} showTimestamp />
    )
    const img = container.querySelector('img')
    expect(img).toBeInTheDocument()
    expect(img).toHaveAttribute('src', 'data:image/png;base64,abc')
    expect(img).toHaveAttribute('alt', 'test.png')
  })

  it('transitions to visible state on mount', () => {
    const { container } = render(<MessageBubble content="hi" role="user" timestamp={new Date()} showTimestamp />)
    const outer = container.firstChild as HTMLElement
    expect(outer.className).toContain('opacity-100')
    expect(outer.className).toContain('translate-y-0')
  })

  it('accepts custom aria-live for assistant', () => {
    const { container } = render(<MessageBubble content="hi" role="assistant" timestamp={new Date()} showTimestamp aria-live="assertive" />)
    const article = container.querySelector('[role="article"]')
    expect(article).toHaveAttribute('aria-live', 'assertive')
  })

  describe('message folding', () => {
    const longContent = 'A'.repeat(600)
    const shortContent = 'A'.repeat(50)

    it('collapses long content when collapsibleLength is set', () => {
      const { container } = render(
        <MessageBubble content={longContent} role="assistant" timestamp={new Date()} showTimestamp collapsibleLength={200} />
      )
      const markdown = container.querySelector('[data-testid="markdown"]')
      expect(markdown?.textContent?.length).toBe(200)
    })

    it('does not collapse short content', () => {
      const { container } = render(
        <MessageBubble content={shortContent} role="assistant" timestamp={new Date()} showTimestamp collapsibleLength={200} />
      )
      const markdown = container.querySelector('[data-testid="markdown"]')
      expect(markdown?.textContent?.length).toBe(50)
    })

    it('shows "Show more" button when collapsed', () => {
      render(
        <MessageBubble content={longContent} role="assistant" timestamp={new Date()} showTimestamp collapsibleLength={200} />
      )
      expect(screen.getByText(/Show more/)).toBeInTheDocument()
    })

    it('does not show "Show more" when collapsibleLength is 0', () => {
      const { container } = render(
        <MessageBubble content={longContent} role="assistant" timestamp={new Date()} showTimestamp collapsibleLength={0} />
      )
      expect(screen.queryByText(/Show more/)).not.toBeInTheDocument()
      const markdown = container.querySelector('[data-testid="markdown"]')
      expect(markdown?.textContent?.length).toBe(600)
    })

    it('shows remaining char count in collapse label', () => {
      render(
        <MessageBubble content={longContent} role="assistant" timestamp={new Date()} showTimestamp collapsibleLength={200} />
      )
      expect(screen.getByText(/400 more/)).toBeInTheDocument()
    })

    it('collapses user messages too', () => {
      const { container } = render(
        <MessageBubble content={longContent} role="user" timestamp={new Date()} showTimestamp collapsibleLength={200} />
      )
      expect(container.textContent).toContain('Show more')
    })
  })
})
