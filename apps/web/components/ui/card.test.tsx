/**
 * @vitest-environment jsdom
 */
import { describe, expect, it, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from './card'

afterEach(cleanup)

describe('Card', () => {
  it('renders children', () => {
    render(<Card><span data-testid="child" /></Card>)
    expect(screen.getByTestId('child')).toBeInTheDocument()
  })

  it('has border and rounded classes', () => {
    const { container } = render(<Card><div /></Card>)
    const el = container.firstChild as HTMLElement
    expect(el.className).toContain('rounded-lg')
    expect(el.className).toContain('border')
    expect(el.className).toContain('shadow-sm')
  })

  it('renders as div by default', () => {
    const { container } = render(<Card><div /></Card>)
    expect(container.firstChild?.nodeName).toBe('DIV')
  })

  it('applies custom className', () => {
    const { container } = render(<Card className="my-card"><div /></Card>)
    expect(container.firstChild).toHaveClass('my-card')
  })

  it('forwards ref', () => {
    const ref = vi.fn()
    render(<Card ref={ref}><div /></Card>)
    expect(ref).toHaveBeenCalled()
  })
})

describe('CardHeader', () => {
  it('renders children', () => {
    render(<CardHeader><h2>Header</h2></CardHeader>)
    expect(screen.getByText('Header')).toBeInTheDocument()
  })

  it('has flex column layout', () => {
    const { container } = render(<CardHeader><div /></CardHeader>)
    expect(container.firstChild).toHaveClass('flex')
    expect(container.firstChild).toHaveClass('flex-col')
  })

  it('applies custom className', () => {
    const { container } = render(<CardHeader className="p-4"><div /></CardHeader>)
    expect(container.firstChild).toHaveClass('p-4')
  })
})

describe('CardTitle', () => {
  it('renders text', () => {
    render(<CardTitle>Title</CardTitle>)
    expect(screen.getByText('Title')).toBeInTheDocument()
  })

  it('renders as h2 element', () => {
    render(<CardTitle>Title</CardTitle>)
    expect(screen.getByRole('heading', { level: 2 })).toBeInTheDocument()
  })

  it('has font-semibold class', () => {
    const { container } = render(<CardTitle>Title</CardTitle>)
    expect(container.firstChild).toHaveClass('font-semibold')
  })

  it('applies custom className', () => {
    const { container } = render(<CardTitle className="text-base">Title</CardTitle>)
    expect(container.firstChild).toHaveClass('text-base')
  })
})

describe('CardDescription', () => {
  it('renders text', () => {
    render(<CardDescription>Description</CardDescription>)
    expect(screen.getByText('Description')).toBeInTheDocument()
  })

  it('renders as p element', () => {
    const { container } = render(<CardDescription>Desc</CardDescription>)
    expect(container.firstChild?.nodeName).toBe('P')
  })

  it('has muted foreground class', () => {
    const { container } = render(<CardDescription>Desc</CardDescription>)
    expect(container.firstChild).toHaveClass('text-muted-foreground')
  })

  it('applies custom className', () => {
    const { container } = render(<CardDescription className="italic">Desc</CardDescription>)
    expect(container.firstChild).toHaveClass('italic')
  })
})

describe('CardContent', () => {
  it('renders children', () => {
    render(<CardContent><span data-testid="content" /></CardContent>)
    expect(screen.getByTestId('content')).toBeInTheDocument()
  })

  it('has pt-0 class', () => {
    const { container } = render(<CardContent>content</CardContent>)
    expect(container.firstChild).toHaveClass('pt-0')
  })

  it('renders as div', () => {
    const { container } = render(<CardContent>content</CardContent>)
    expect(container.firstChild?.nodeName).toBe('DIV')
  })

  it('applies custom className', () => {
    const { container } = render(<CardContent className="px-0">content</CardContent>)
    expect(container.firstChild).toHaveClass('px-0')
  })
})

describe('CardFooter', () => {
  it('renders children', () => {
    render(<CardFooter><span data-testid="footer" /></CardFooter>)
    expect(screen.getByTestId('footer')).toBeInTheDocument()
  })

  it('has flex items-center class', () => {
    const { container } = render(<CardFooter>footer</CardFooter>)
    expect(container.firstChild).toHaveClass('flex')
    expect(container.firstChild).toHaveClass('items-center')
  })

  it('renders as div', () => {
    const { container } = render(<CardFooter>footer</CardFooter>)
    expect(container.firstChild?.nodeName).toBe('DIV')
  })

  it('applies custom className', () => {
    const { container } = render(<CardFooter className="justify-end">footer</CardFooter>)
    expect(container.firstChild).toHaveClass('justify-end')
  })
})
