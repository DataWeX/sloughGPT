/**
 */
import { describe, expect, it, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import { Tabs, TabsList, TabsTrigger, TabsContent } from './tabs'

afterEach(cleanup)

describe('Tabs', () => {
  it('renders trigger with label', () => {
    render(
      <Tabs defaultValue="tab1">
        <TabsList>
          <TabsTrigger value="tab1">First</TabsTrigger>
          <TabsTrigger value="tab2">Second</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
      </Tabs>
    )
    expect(screen.getByText('First')).toBeInTheDocument()
    expect(screen.getByText('Second')).toBeInTheDocument()
  })

  it('shows active tab content', () => {
    render(
      <Tabs defaultValue="tab2">
        <TabsList>
          <TabsTrigger value="tab1">First</TabsTrigger>
          <TabsTrigger value="tab2">Second</TabsTrigger>
        </TabsList>
        <TabsContent value="tab1">Content 1</TabsContent>
        <TabsContent value="tab2">Content 2</TabsContent>
      </Tabs>
    )
    expect(screen.getByText('Content 2')).toBeInTheDocument()
    expect(screen.queryByText('Content 1')).not.toBeInTheDocument()
  })

  it('applies className to TabsList', () => {
    const { container } = render(
      <Tabs defaultValue="a">
        <TabsList className="custom-list">
          <TabsTrigger value="a">A</TabsTrigger>
        </TabsList>
      </Tabs>
    )
    const list = container.querySelector('[role="tablist"]')
    expect(list).toHaveClass('custom-list')
  })

  it('applies className to TabsTrigger', () => {
    const { container } = render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a" className="custom-trigger">A</TabsTrigger>
        </TabsList>
      </Tabs>
    )
    const trigger = container.querySelector('[role="tab"]')
    expect(trigger).toHaveClass('custom-trigger')
  })

  it('applies className to TabsContent', () => {
    const { container } = render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a">A</TabsTrigger>
        </TabsList>
        <TabsContent value="a" className="custom-content">Content</TabsContent>
      </Tabs>
    )
    const content = container.querySelector('[role="tabpanel"]')
    expect(content).toHaveClass('custom-content')
  })

  it('sets data-state on active trigger', () => {
    const { container } = render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a">A</TabsTrigger>
          <TabsTrigger value="b">B</TabsTrigger>
        </TabsList>
      </Tabs>
    )
    const triggers = container.querySelectorAll('[role="tab"]')
    expect(triggers[0]).toHaveAttribute('data-state', 'active')
    expect(triggers[1]).toHaveAttribute('data-state', 'inactive')
  })

  it('has accessible roles', () => {
    render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
      </Tabs>
    )
    expect(screen.getByRole('tablist')).toBeInTheDocument()
    expect(screen.getByRole('tab')).toBeInTheDocument()
    expect(screen.getByRole('tabpanel')).toBeInTheDocument()
  })
})
