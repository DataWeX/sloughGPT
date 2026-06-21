/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import { Markdown } from './Markdown'

afterEach(cleanup)

describe('Markdown', () => {
  it('renders plain text', () => {
    const { container } = render(<Markdown content="Hello world" />)
    expect(container.textContent).toContain('Hello world')
  })

  it('renders paragraph for plain text', () => {
    render(<Markdown content="Hello world" />)
    const p = screen.getByText('Hello world')
    expect(p.tagName).toBe('P')
  })

  it('renders heading 1', () => {
    render(<Markdown content="# Title" />)
    const h1 = screen.getByText('Title')
    expect(h1.tagName).toBe('H1')
    expect(h1.className).toContain('text-base')
  })

  it('renders heading 2', () => {
    render(<Markdown content="## Section" />)
    const h2 = screen.getByText('Section')
    expect(h2.tagName).toBe('H2')
    expect(h2.className).toContain('text-sm')
  })

  it('renders heading 3', () => {
    render(<Markdown content="### Sub" />)
    const h3 = screen.getByText('Sub')
    expect(h3.tagName).toBe('H3')
    expect(h3.className).toContain('text-xs')
  })

  it('renders heading 6', () => {
    render(<Markdown content="###### Tiny" />)
    const h6 = screen.getByText('Tiny')
    expect(h6.tagName).toBe('H6')
  })

  it('renders bold text', () => {
    render(<Markdown content="hello **world** here" />)
    expect(screen.getByText('world').tagName).toBe('STRONG')
  })

  it('renders italic text', () => {
    render(<Markdown content="hello *world* here" />)
    expect(screen.getByText('world').tagName).toBe('EM')
  })

  it('renders inline code', () => {
    const { container } = render(<Markdown content="use `code` here" />)
    const code = container.querySelector('code')
    expect(code).toBeInTheDocument()
    expect(code?.textContent).toBe('code')
  })

  it('renders link', () => {
    render(<Markdown content="click [here](https://example.com) now" />)
    const link = screen.getByText('here')
    expect(link.tagName).toBe('A')
    expect(link).toHaveAttribute('href', 'https://example.com')
    expect(link).toHaveAttribute('target', '_blank')
  })

  it('renders link with rel noopener', () => {
    render(<Markdown content="[link](https://x.com)" />)
    const link = screen.getByText('link')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('renders unordered list', () => {
    const { container } = render(<Markdown content={'- item1\n- item2\n- item3'} />)
    const ul = container.querySelector('ul')
    expect(ul).toBeInTheDocument()
    expect(ul?.className).toContain('list-disc')
    const items = container.querySelectorAll('li')
    expect(items).toHaveLength(3)
    expect(items[0].textContent).toBe('item1')
    expect(items[2].textContent).toBe('item3')
  })

  it('renders ordered list', () => {
    const { container } = render(<Markdown content={'1. first\n2. second\n3. third'} />)
    const ol = container.querySelector('ol')
    expect(ol).toBeInTheDocument()
    expect(ol?.className).toContain('list-decimal')
    const items = container.querySelectorAll('li')
    expect(items).toHaveLength(3)
  })

  it('renders blockquote', () => {
    const { container } = render(<Markdown content="> quoted text" />)
    const bq = container.querySelector('blockquote')
    expect(bq).toBeInTheDocument()
    expect(bq?.className).toContain('border-l-2')
    expect(bq?.textContent).toBe('quoted text')
  })

  it('renders multi-line blockquote', () => {
    const { container } = render(<Markdown content={'> line1\n> line2\n> line3'} />)
    const bq = container.querySelector('blockquote')
    expect(bq).toBeInTheDocument()
    expect(bq?.textContent).toContain('line1')
    expect(bq?.textContent).toContain('line2')
  })

  it('renders horizontal rule', () => {
    const { container } = render(<Markdown content="---" />)
    const hr = container.querySelector('hr')
    expect(hr).toBeInTheDocument()
  })

  it('renders code block with language', () => {
    const { container } = render(<Markdown content={"```python\nprint('hi')\n```"} />)
    const pre = container.querySelector('pre')
    expect(pre).toBeInTheDocument()
    expect(pre?.textContent).toContain("print('hi')")
    expect(container.textContent).toContain('python')
  })

  it('renders code block without language', () => {
    const { container } = render(<Markdown content={"```\nplain code\n```"} />)
    const pre = container.querySelector('pre')
    expect(pre).toBeInTheDocument()
    expect(container.textContent).toContain('code')
  })

  it('code block has copy button', () => {
    const { container } = render(<Markdown content={"```js\nvar x = 1\n```"} />)
    const copyBtn = container.querySelector('button')
    expect(copyBtn).toBeInTheDocument()
    expect(copyBtn).toHaveAttribute('aria-label', 'Copy code')
  })

  it('code block copy button changes to Copied on click', async () => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    })
    const { container } = render(<Markdown content={"```js\nvar x = 1\n```"} />)
    const copyBtn = container.querySelector('button')!
    fireEvent.click(copyBtn)
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith('var x = 1')
    await vi.waitFor(() => {
      expect(screen.getByText('Copied')).toBeInTheDocument()
    })
  })

  it('renders bold text inline', () => {
    render(<Markdown content="hello **world** here" />)
    const strong = screen.getByText('world')
    expect(strong.tagName).toBe('STRONG')
    expect(strong.closest('p')?.textContent).toBe('hello world here')
  })

  it('renders italic text', () => {
    render(<Markdown content="hello *world* here" />)
    expect(screen.getByText('world').tagName).toBe('EM')
  })

  it('renders inline code mid-sentence', () => {
    render(<Markdown content="use `code` here please" />)
    expect(screen.getByText('code').tagName).toBe('CODE')
  })

  it('renders empty content without error', () => {
    const { container } = render(<Markdown content="" />)
    const div = container.querySelector('div')
    expect(div).toBeInTheDocument()
  })

  it('applies custom className', () => {
    const { container } = render(<Markdown content="text" className="custom-md" />)
    expect(container.firstChild).toHaveClass('custom-md')
  })

  it('renders link inside paragraph', () => {
    render(<Markdown content="click [here](https://example.com) now" />)
    const link = screen.getByText('here')
    expect(link.tagName).toBe('A')
    expect(link).toHaveAttribute('href', 'https://example.com')
  })
})
