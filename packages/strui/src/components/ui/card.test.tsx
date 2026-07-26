import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter, cardVariants } from './card'

describe('Card', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<Card>Content</Card>)
    expect(html).toContain('Content')
  })

  it('default variant has border-border', () => {
    const cls = cardVariants({ variant: 'default' })
    expect(cls).toContain('border-border')
  })

  it('interactive variant has hover classes', () => {
    const cls = cardVariants({ variant: 'interactive' })
    expect(cls).toContain('hover:-translate-y-0.5')
    expect(cls).toContain('hover:shadow-md')
    expect(cls).toContain('cursor-pointer')
  })

  it('selected variant has primary border', () => {
    const cls = cardVariants({ variant: 'selected' })
    expect(cls).toContain('border-primary/40')
    expect(cls).toContain('bg-primary/5')
  })

  it('loading variant has animate-pulse', () => {
    const cls = cardVariants({ variant: 'loading' })
    expect(cls).toContain('animate-pulse')
    expect(cls).toContain('bg-muted/50')
  })

  it('error variant has destructive border', () => {
    const cls = cardVariants({ variant: 'error' })
    expect(cls).toContain('border-destructive/40')
    expect(cls).toContain('bg-destructive/5')
  })

  it('accepts custom className', () => {
    const html = renderToStaticMarkup(<Card className="my-class">X</Card>)
    expect(html).toContain('my-class')
  })

  it('has rounded-lg', () => {
    const cls = cardVariants()
    expect(cls).toContain('rounded-lg')
  })
})

describe('CardHeader', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<CardHeader>Title area</CardHeader>)
    expect(html).toContain('Title area')
  })

  it('has padding', () => {
    const html = renderToStaticMarkup(<CardHeader>X</CardHeader>)
    expect(html).toContain('p-5')
  })
})

describe('CardTitle', () => {
  it('renders as h3 by default', () => {
    const html = renderToStaticMarkup(<CardTitle>My Title</CardTitle>)
    expect(html).toContain('<h3')
    expect(html).toContain('My Title')
  })

  it('uses font-medium not font-semibold', () => {
    const html = renderToStaticMarkup(<CardTitle>Title</CardTitle>)
    expect(html).toContain('font-medium')
    expect(html).not.toContain('font-semibold')
  })

  it('uses text-base', () => {
    const html = renderToStaticMarkup(<CardTitle>Title</CardTitle>)
    expect(html).toContain('text-base')
  })

  it('renders as custom element with as prop', () => {
    const html = renderToStaticMarkup(<CardTitle as="h2">Title</CardTitle>)
    expect(html).toContain('<h2')
  })
})

describe('CardDescription', () => {
  it('renders as p element', () => {
    const html = renderToStaticMarkup(<CardDescription>Desc</CardDescription>)
    expect(html).toContain('<p')
    expect(html).toContain('Desc')
  })

  it('uses muted-foreground', () => {
    const html = renderToStaticMarkup(<CardDescription>D</CardDescription>)
    expect(html).toContain('text-muted-foreground')
  })
})

describe('CardContent', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<CardContent>Body</CardContent>)
    expect(html).toContain('Body')
  })

  it('has padding', () => {
    const html = renderToStaticMarkup(<CardContent>X</CardContent>)
    expect(html).toContain('p-5')
  })
})

describe('CardFooter', () => {
  it('renders children', () => {
    const html = renderToStaticMarkup(<CardFooter>Actions</CardFooter>)
    expect(html).toContain('Actions')
  })

  it('has padding and flex', () => {
    const html = renderToStaticMarkup(<CardFooter>X</CardFooter>)
    expect(html).toContain('p-5')
    expect(html).toContain('flex')
  })
})
