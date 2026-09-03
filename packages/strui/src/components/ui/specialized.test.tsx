import { renderToStaticMarkup } from 'react-dom/server'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import {
  Avatar,
  AvatarGroup,
  CardDeck,
  Divider,
  EmptyState,
  Pagination,
  ProgressBar,
  SearchField,
  Spinner,
} from './specialized'

afterEach(() => {
  cleanup()
})

describe('Avatar', () => {
  it('renders an image when src is provided', () => {
    const html = renderToStaticMarkup(<Avatar src="/a.png" alt="Alice" fallback="Alice" />)
    expect(html).toContain('<img')
    expect(html).toContain('src="/a.png"')
    expect(html).toContain('alt="Alice"')
  })

  it('falls back to the fallback text for alt', () => {
    const html = renderToStaticMarkup(<Avatar src="/a.png" fallback="Alice" />)
    expect(html).toContain('alt="Alice"')
  })

  it('renders fallback initials when no src', () => {
    const html = renderToStaticMarkup(<Avatar fallback="alice wong" />)
    expect(html).not.toContain('<img')
    expect(html).toContain('AL')
  })

  it('applies size classes', () => {
    expect(renderToStaticMarkup(<Avatar fallback="A" size="xs" />)).toContain('h-5 w-5')
    expect(renderToStaticMarkup(<Avatar fallback="A" size="lg" />)).toContain('h-10 w-10')
    expect(renderToStaticMarkup(<Avatar fallback="A" />)).toContain('h-8 w-8')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Avatar fallback="A" className="my-avatar" />)
    expect(html).toContain('my-avatar')
  })
})

describe('AvatarGroup', () => {
  const avatars = [
    { src: '/1.png', fallback: 'One' },
    { src: '/2.png', fallback: 'Two' },
    { src: '/3.png', fallback: 'Three' },
    { src: '/4.png', fallback: 'Four' },
    { src: '/5.png', fallback: 'Five' },
  ]

  it('renders up to max avatars', () => {
    const html = renderToStaticMarkup(<AvatarGroup avatars={avatars} max={3} />)
    expect(html).toContain('src="/1.png"')
    expect(html).toContain('src="/3.png"')
    expect(html).not.toContain('src="/4.png"')
  })

  it('shows the overflow count when avatars exceed max', () => {
    const html = renderToStaticMarkup(<AvatarGroup avatars={avatars} max={3} />)
    expect(html).toContain('+2')
  })

  it('does not show an overflow label within max', () => {
    const html = renderToStaticMarkup(<AvatarGroup avatars={avatars.slice(0, 2)} max={3} />)
    expect(html).not.toContain('+')
  })

  it('defaults max to 4', () => {
    const html = renderToStaticMarkup(<AvatarGroup avatars={avatars} />)
    expect(html).toContain('src="/4.png"')
    expect(html).not.toContain('src="/5.png"')
    expect(html).toContain('+1')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<AvatarGroup avatars={avatars.slice(0, 2)} className="my-group" />)
    expect(html).toContain('my-group')
  })
})

describe('ProgressBar', () => {
  it('renders a progressbar with aria attributes', () => {
    const html = renderToStaticMarkup(<ProgressBar value={50} label="Uploading" />)
    expect(html).toContain('role="progressbar"')
    expect(html).toContain('aria-valuenow="50"')
    expect(html).toContain('aria-valuemin="0"')
    expect(html).toContain('aria-valuemax="100"')
    expect(html).toContain('Uploading')
  })

  it('renders the width percentage', () => {
    const html = renderToStaticMarkup(<ProgressBar value={50} />)
    expect(html).toContain('width:50%')
  })

  it('clamps the value above 100', () => {
    const html = renderToStaticMarkup(<ProgressBar value={150} />)
    expect(html).toContain('width:100%')
  })

  it('clamps the value below 0', () => {
    const html = renderToStaticMarkup(<ProgressBar value={-20} />)
    expect(html).toContain('width:0%')
  })

  it('uses max for the percentage and aria', () => {
    const html = renderToStaticMarkup(<ProgressBar value={50} max={200} />)
    expect(html).toContain('width:25%')
    expect(html).toContain('aria-valuemax="200"')
  })

  it('renders label and value text', () => {
    const html = renderToStaticMarkup(<ProgressBar value={50} label="Uploading" />)
    expect(html).toContain('Uploading')
    expect(html).toContain('50%')
  })

  it('hides the value text when showValue is false', () => {
    const html = renderToStaticMarkup(<ProgressBar value={50} showValue={false} />)
    expect(html).not.toContain('justify-between')
  })

  it('applies variant classes', () => {
    expect(renderToStaticMarkup(<ProgressBar value={50} variant="success" />)).toContain('bg-success')
    expect(renderToStaticMarkup(<ProgressBar value={50} variant="error" />)).toContain('bg-destructive')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<ProgressBar value={50} className="my-progress" />)
    expect(html).toContain('my-progress')
  })
})

describe('Spinner', () => {
  it('renders a status with a Loading label', () => {
    const html = renderToStaticMarkup(<Spinner />)
    expect(html).toContain('role="status"')
    expect(html).toContain('aria-label="Loading"')
  })

  it('defaults to md size', () => {
    expect(renderToStaticMarkup(<Spinner />)).toContain('w-6 h-6')
  })

  it('supports size variants', () => {
    expect(renderToStaticMarkup(<Spinner size="xs" />)).toContain('w-3 h-3')
    expect(renderToStaticMarkup(<Spinner size="lg" />)).toContain('w-8 h-8')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Spinner className="my-spinner" />)
    expect(html).toContain('my-spinner')
  })
})

describe('Divider', () => {
  it('renders a horizontal divider by default', () => {
    const html = renderToStaticMarkup(<Divider />)
    expect(html).toContain('h-px')
    expect(html).toContain('role="separator"')
  })

  it('renders a vertical divider', () => {
    const html = renderToStaticMarkup(<Divider orientation="vertical" />)
    expect(html).toContain('w-px')
    expect(html).not.toContain('h-px')
  })

  it('renders a labeled divider', () => {
    const html = renderToStaticMarkup(<Divider label="Or" />)
    expect(html).toContain('Or')
    expect(html).toContain('role="separator"')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Divider className="my-divider" />)
    expect(html).toContain('my-divider')
  })
})

describe('CardDeck', () => {
  it('renders title, description and children', () => {
    const html = renderToStaticMarkup(
      <CardDeck title="Models" description="All available">
        <span data-child="row">Row</span>
      </CardDeck>,
    )
    expect(html).toContain('Models')
    expect(html).toContain('All available')
    expect(html).toContain('data-child="row"')
  })

  it('renders a footer', () => {
    const html = renderToStaticMarkup(
      <CardDeck footer={<button data-action="more">More</button>}><span /></CardDeck>,
    )
    expect(html).toContain('data-action="more"')
  })

  it('omits the header when there is no title or description', () => {
    const html = renderToStaticMarkup(<CardDeck><span /></CardDeck>)
    expect(html).not.toContain('font-medium')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<CardDeck className="my-deck"><span /></CardDeck>)
    expect(html).toContain('my-deck')
  })
})

describe('EmptyState', () => {
  it('renders title and description', () => {
    const html = renderToStaticMarkup(<EmptyState title="No results" description="Try another query" />)
    expect(html).toContain('No results')
    expect(html).toContain('Try another query')
  })

  it('renders icon and action', () => {
    const html = renderToStaticMarkup(
      <EmptyState title="Empty" icon={<span data-icon="e">E</span>} action={<button data-action="create">Create</button>} />,
    )
    expect(html).toContain('data-icon="e"')
    expect(html).toContain('data-action="create"')
  })

  it('applies size padding', () => {
    expect(renderToStaticMarkup(<EmptyState title="A" size="sm" />)).toContain('py-8 px-3')
    expect(renderToStaticMarkup(<EmptyState title="A" size="lg" />)).toContain('py-16 px-6')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<EmptyState title="A" className="my-state" />)
    expect(html).toContain('my-state')
  })
})

describe('SearchField', () => {
  it('renders a search input with the default placeholder', () => {
    const html = renderToStaticMarkup(<SearchField value="" onChange={() => {}} />)
    expect(html).toContain('type="search"')
    expect(html).toContain('placeholder="Search…"')
  })

  it('uses a custom placeholder', () => {
    const html = renderToStaticMarkup(<SearchField value="" onChange={() => {}} placeholder="Find models" />)
    expect(html).toContain('placeholder="Find models"')
  })

  it('fires onChange when typing', () => {
    const onChange = vi.fn()
    render(<SearchField value="" onChange={onChange} />)
    fireEvent.change(screen.getByRole('searchbox'), { target: { value: 'abc' } })
    expect(onChange).toHaveBeenCalledWith('abc')
  })

  it('clears the value via the clear button', () => {
    const onChange = vi.fn()
    render(<SearchField value="abc" onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Clear search'))
    expect(onChange).toHaveBeenCalledWith('')
  })

  it('omits the clear button when the value is empty', () => {
    const html = renderToStaticMarkup(<SearchField value="" onChange={() => {}} />)
    expect(html).not.toContain('Clear search')
  })

  it('passes className to the wrapper', () => {
    const html = renderToStaticMarkup(<SearchField value="" onChange={() => {}} className="my-field" />)
    expect(html).toContain('my-field')
  })
})

describe('Pagination', () => {
  it('renders page info and navigation buttons', () => {
    const html = renderToStaticMarkup(<Pagination page={2} total={100} pageSize={10} onChange={() => {}} />)
    expect(html).toContain('Page 2 of 10 (100 items)')
    expect(html).toContain('aria-label="Previous page"')
    expect(html).toContain('aria-label="Next page"')
  })

  it('fires onChange with the previous page', () => {
    const onChange = vi.fn()
    render(<Pagination page={2} total={100} pageSize={10} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Previous page'))
    expect(onChange).toHaveBeenCalledWith(1)
  })

  it('fires onChange with the next page', () => {
    const onChange = vi.fn()
    render(<Pagination page={2} total={100} pageSize={10} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Next page'))
    expect(onChange).toHaveBeenCalledWith(3)
  })

  it('disables previous on the first page', () => {
    const onChange = vi.fn()
    render(<Pagination page={1} total={100} pageSize={10} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Previous page'))
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Previous page').getAttribute('disabled')).not.toBeNull()
  })

  it('disables next on the last page', () => {
    const onChange = vi.fn()
    render(<Pagination page={10} total={100} pageSize={10} onChange={onChange} />)
    fireEvent.click(screen.getByLabelText('Next page'))
    expect(onChange).not.toHaveBeenCalled()
    expect(screen.getByLabelText('Next page').getAttribute('disabled')).not.toBeNull()
  })

  it('renders nothing when there is a single page', () => {
    const html = renderToStaticMarkup(<Pagination page={1} total={5} pageSize={10} onChange={() => {}} />)
    expect(html).toBe('')
  })
})
