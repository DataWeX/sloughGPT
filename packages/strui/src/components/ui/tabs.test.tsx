import { describe, expect, it, vi, afterEach } from 'vitest'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'

import { Tabs, TabsList, TabsTrigger, TabsContent } from './tabs'

function renderTabs(props: { onValueChange?: (value: string) => void } = {}) {
  return render(
    <Tabs defaultValue="a" {...props}>
      <TabsList>
        <TabsTrigger value="a">Tab A</TabsTrigger>
        <TabsTrigger value="b">Tab B</TabsTrigger>
      </TabsList>
      <TabsContent value="a">Content A</TabsContent>
      <TabsContent value="b">Content B</TabsContent>
    </Tabs>,
  )
}

describe('Tabs', () => {
  afterEach(() => {
    cleanup()
  })

  it('renders a tablist with tab triggers', () => {
    renderTabs()
    expect(screen.getByRole('tablist').getAttribute('role')).toBe('tablist')
    const tabs = screen.getAllByRole('tab')
    expect(tabs.length).toBe(2)
    expect(tabs[0].getAttribute('role')).toBe('tab')
  })

  it('activates the defaultValue tab', () => {
    renderTabs()
    const tabA = screen.getByText('Tab A')
    const tabB = screen.getByText('Tab B')
    expect(tabA.getAttribute('aria-selected')).toBe('true')
    expect(tabB.getAttribute('aria-selected')).toBe('false')
    expect(screen.getByText('Content A')).toBeTruthy()
  })

  it('does not render inactive content', () => {
    renderTabs()
    expect(screen.getByText('Content A')).toBeTruthy()
    expect(screen.queryByText('Content B')).toBeNull()
  })

  it('activates the clicked trigger and renders its content', () => {
    const onValueChange = vi.fn()
    renderTabs({ onValueChange })
    fireEvent.click(screen.getByText('Tab B'))
    expect(onValueChange).toHaveBeenCalledWith('b')
    expect(screen.getByText('Tab B').getAttribute('aria-selected')).toBe('true')
    expect(screen.getByText('Tab A').getAttribute('aria-selected')).toBe('false')
    expect(screen.getByText('Content B')).toBeTruthy()
    expect(screen.queryByText('Content A')).toBeNull()
  })

  it('marks the active content with role tabpanel', () => {
    renderTabs()
    const panel = screen.getByRole('tabpanel')
    expect(panel.getAttribute('role')).toBe('tabpanel')
    expect(panel.textContent).toBe('Content A')
  })

  it('links each trigger to its panel via aria-controls and aria-labelledby', () => {
    renderTabs()
    const tabA = screen.getByText('Tab A')
    const panelA = document.getElementById(tabA.getAttribute('aria-controls')!)!
    expect(panelA.getAttribute('role')).toBe('tabpanel')
    expect(panelA.getAttribute('aria-labelledby')).toBe(tabA.id)
    const tabB = screen.getByText('Tab B')
    expect(tabB.getAttribute('aria-controls')).toBeTruthy()
  })

  it('moves selection with arrow keys and wraps around', () => {
    const onValueChange = vi.fn()
    renderTabs({ onValueChange })
    const list = screen.getByRole('tablist')
    fireEvent.keyDown(list, { key: 'ArrowRight' })
    expect(onValueChange).toHaveBeenCalledWith('b')
    const tabB = screen.getByText('Tab B')
    expect(tabB.getAttribute('aria-selected')).toBe('true')
    fireEvent.keyDown(list, { key: 'ArrowRight' })
    expect(onValueChange).toHaveBeenCalledWith('a')
    fireEvent.keyDown(list, { key: 'ArrowLeft' })
    expect(onValueChange).toHaveBeenCalledWith('b')
  })

  it('moves to first and last tab with Home and End', () => {
    const onValueChange = vi.fn()
    render(
      <Tabs defaultValue="b" onValueChange={onValueChange}>
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b">Tab B</TabsTrigger>
          <TabsTrigger value="c">Tab C</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
        <TabsContent value="b">Content B</TabsContent>
        <TabsContent value="c">Content C</TabsContent>
      </Tabs>,
    )
    const list = screen.getByRole('tablist')
    fireEvent.keyDown(list, { key: 'Home' })
    expect(onValueChange).toHaveBeenCalledWith('a')
    fireEvent.keyDown(list, { key: 'End' })
    expect(onValueChange).toHaveBeenCalledWith('c')
  })

  it('skips disabled triggers during arrow navigation', () => {
    const onValueChange = vi.fn()
    render(
      <Tabs defaultValue="a" onValueChange={onValueChange}>
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b" disabled>
            Tab B
          </TabsTrigger>
          <TabsTrigger value="c">Tab C</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
      </Tabs>,
    )
    const list = screen.getByRole('tablist')
    fireEvent.keyDown(list, { key: 'ArrowRight' })
    expect(onValueChange).toHaveBeenCalledWith('c')
  })

  it('disabled trigger does not fire onValueChange', () => {
    const onValueChange = vi.fn()
    render(
      <Tabs defaultValue="a" onValueChange={onValueChange}>
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b" disabled>
            Tab B
          </TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
        <TabsContent value="b">Content B</TabsContent>
      </Tabs>,
    )
    const tabB = screen.getByText('Tab B')
    expect(tabB.getAttribute('aria-disabled')).toBe('true')
    fireEvent.click(tabB)
    expect(onValueChange).not.toHaveBeenCalled()
    expect(screen.getByText('Content A')).toBeTruthy()
    expect(screen.queryByText('Content B')).toBeNull()
  })

  it('respects controlled value', () => {
    const onValueChange = vi.fn()
    render(
      <Tabs value="a" onValueChange={onValueChange}>
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b">Tab B</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
        <TabsContent value="b">Content B</TabsContent>
      </Tabs>,
    )
    fireEvent.click(screen.getByText('Tab B'))
    expect(onValueChange).toHaveBeenCalledWith('b')
    expect(screen.getByText('Content A')).toBeTruthy()
    expect(screen.queryByText('Content B')).toBeNull()
  })

  it('supports onChange as an alias for onValueChange', () => {
    const onChange = vi.fn()
    render(
      <Tabs defaultValue="a" onChange={onChange}>
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b">Tab B</TabsTrigger>
        </TabsList>
        <TabsContent value="a">Content A</TabsContent>
        <TabsContent value="b">Content B</TabsContent>
      </Tabs>,
    )
    fireEvent.click(screen.getByText('Tab B'))
    expect(onChange).toHaveBeenCalledWith('b')
  })

  it('keeps inactive content in the DOM with hidden when forceMount', () => {
    const { container } = render(
      <Tabs defaultValue="a">
        <TabsList>
          <TabsTrigger value="a">Tab A</TabsTrigger>
          <TabsTrigger value="b">Tab B</TabsTrigger>
        </TabsList>
        <TabsContent value="a" forceMount>
          Content A
        </TabsContent>
        <TabsContent value="b" forceMount>
          Content B
        </TabsContent>
      </Tabs>,
    )
    const panels = container.querySelectorAll('[role="tabpanel"]')
    expect(panels.length).toBe(2)
    expect(panels[1].getAttribute('hidden')).toBe('')
  })

  it('renders triggers from the tabs shorthand', () => {
    render(
      <Tabs defaultValue="a" tabs={[
        { value: 'a', label: 'Shorthand A' },
        { value: 'b', label: 'Shorthand B', count: 3 },
      ]}>
        <TabsContent value="a">Content A</TabsContent>
      </Tabs>,
    )
    expect(screen.getByRole('tablist')).toBeTruthy()
    expect(screen.getByText('Shorthand A')).toBeTruthy()
    expect(screen.getByText('Shorthand B')).toBeTruthy()
    expect(screen.getByText('3')).toBeTruthy()
  })
})
