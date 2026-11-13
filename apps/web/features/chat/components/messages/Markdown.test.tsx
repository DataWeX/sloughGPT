/**
 */
import React from 'react'
import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'

vi.mock('./CodeBlock', () => {
  return {
    CodeBlock: (props: any) => {
      const { language, code } = props || {}
      const [copied, setCopied] = React.useState(false)
      return (
        <div>
          <span>{language || 'code'}</span>
          <button
            aria-label="Copy code"
            onClick={() => navigator.clipboard.writeText(code).then(() => setCopied(true))}
          >
            {copied ? 'Copied' : 'Copy'}
          </button>
          <pre>{code}</pre>
        </div>
      )
    },
  }
})

vi.mock('@/vite/next-compat/dynamic', () => ({
  default: (factory: () => Promise<any>) => {
    let _resolved: any = null
    factory().then((mod: any) => {
      _resolved = typeof mod === 'function' ? mod : mod.default || mod.CodeBlock
    })
    return function ResolvedDynamic(props: any) {
      if (!_resolved) return null
      return <_resolved {...props} />
    }
  },
}))

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

  it('renders code block with language', async () => {
    const { container } = render(<Markdown content={"```python\nprint('hi')\n```"} />)
    await waitFor(() => {
      const pre = container.querySelector('pre')
      expect(pre).toBeInTheDocument()
    })
    const pre = container.querySelector('pre')
    expect(pre?.textContent).toContain("print('hi')")
    expect(container.textContent).toContain('python')
  })

  it('renders code block without language', () => {
    const { container } = render(<Markdown content={'```\nplain code\n```'} />)
    const pre = container.querySelector('pre')
    expect(pre).toBeInTheDocument()
    expect(container.textContent).toContain('code')
  })

  it('code block has copy button', () => {
    const { container } = render(<Markdown content={'```js\nvar x = 1\n```'} />)
    const copyBtn = container.querySelector('button')
    expect(copyBtn).toBeInTheDocument()
    expect(copyBtn).toHaveAttribute('aria-label', 'Copy code')
  })

  it('code block copy button changes to Copied on click', async () => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    })
    const { container } = render(<Markdown content={'```js\nvar x = 1\n```'} />)
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

  // ── Images ────────────────────────────────────────────────────────────
  it('renders image', () => {
    render(<Markdown content="![a cat](https://example.com/cat.png)" />)
    const img = screen.getByAltText('a cat')
    expect(img.tagName).toBe('IMG')
    expect(img).toHaveAttribute('src', 'https://example.com/cat.png')
  })

  it('renders image inline with surrounding text', () => {
    render(<Markdown content="look ![pic](https://example.com/p.png) here" />)
    expect(screen.getByAltText('pic')).toBeInTheDocument()
    expect(screen.getByText(/look/)).toBeInTheDocument()
    expect(screen.getByText(/here/)).toBeInTheDocument()
  })

  it('renders image with empty alt text', () => {
    // alt="" makes the img role="presentation", so query the DOM directly
    // instead of getByRole('img').
    const { container } = render(<Markdown content="![](https://example.com/x.png)" />)
    const img = container.querySelector('img')
    expect(img).not.toBeNull()
    expect(img).toHaveAttribute('src', 'https://example.com/x.png')
    expect(img).toHaveAttribute('alt', '')
  })

  it('renders relative image src', () => {
    render(<Markdown content="![rel](/static/logo.png)" />)
    expect(screen.getByAltText('rel')).toHaveAttribute('src', '/static/logo.png')
  })

  it('does not render image as a link', () => {
    // Regression: the link regex \[(.+?)\]\((.+?)\) matches INSIDE
    // ![alt](url), so before image support this rendered a literal "!"
    // followed by an <a> whose text was the alt.
    render(<Markdown content="![cat](https://example.com/c.png)" />)
    expect(screen.queryByRole('link')).toBeNull()
    expect(screen.getByAltText('cat')).toBeInTheDocument()
  })

  it('renders literal text for unsafe image url', () => {
    // javascript: and data:text/html must never become an img src.
    render(<Markdown content="![bad](javascript:alert(1))" />)
    expect(screen.queryByAltText('bad')).toBeNull()
  })

  it('refuses scriptable data url but allows raster data url', () => {
    const { rerender } = render(
      <Markdown content="![svg](data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=)" />,
    )
    expect(screen.queryByAltText('svg')).toBeNull()

    rerender(<Markdown content="![png](data:image/png;base64,iVBORw0KGgo=)" />)
    expect(screen.getByAltText('png')).toBeInTheDocument()
  })

  it('does not treat a link as an image', () => {
    render(<Markdown content="[not an image](https://example.com)" />)
    expect(screen.queryByRole('img')).toBeNull()
    expect(screen.getByText('not an image')).toBeInTheDocument()
  })
})
