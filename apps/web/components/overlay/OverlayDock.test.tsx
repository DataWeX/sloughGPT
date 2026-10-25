// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'

import { OverlayDock, requestOverlayDockOpen, type OverlayWidget } from './OverlayDock'

function widget(over: Partial<OverlayWidget> = {}): OverlayWidget {
  return {
    id: 'alpha',
    label: 'Alpha widget',
    icon: <span data-testid="icon-alpha">A</span>,
    render: () => <div data-testid="panel-body">alpha body</div>,
    ...over,
  }
}

afterEach(cleanup)

describe('OverlayDock', () => {
  it('renders no region when no widgets are registered', () => {
    render(<OverlayDock widgets={[]} />)
    expect(screen.queryByTestId('overlay-dock')).toBeNull()
  })

  it('renders each widget as a labelled button', () => {
    render(<OverlayDock widgets={[widget(), widget({ id: 'beta', label: 'Beta' })]} />)
    const alpha = screen.getByTestId('overlay-widget-alpha')
    expect(alpha).toHaveAttribute('aria-label', 'Alpha widget')
    expect(alpha).toHaveAttribute('aria-haspopup', 'dialog')
    expect(alpha).toHaveAttribute('aria-expanded', 'false')
    expect(screen.getByTestId('overlay-widget-beta')).toBeDefined()
    // No panel until a widget is clicked.
    expect(screen.queryByTestId('overlay-dock-panel')).toBeNull()
  })

  it('opens the panel on click and closes on second click', () => {
    render(<OverlayDock widgets={[widget()]} />)
    const button = screen.getByTestId('overlay-widget-alpha')
    fireEvent.click(button)
    const panel = screen.getByTestId('overlay-dock-panel')
    expect(panel).toHaveAttribute('role', 'dialog')
    expect(panel).toHaveAttribute('aria-label', 'Alpha widget')
    expect(screen.getByTestId('panel-body')).toBeDefined()
    expect(button).toHaveAttribute('aria-expanded', 'true')

    fireEvent.click(button)
    expect(screen.queryByTestId('overlay-dock-panel')).toBeNull()
    expect(button).toHaveAttribute('aria-expanded', 'false')
  })

  it('keeps exactly one panel open — clicking another widget switches it', () => {
    render(
      <OverlayDock
        widgets={[
          widget(),
          widget({ id: 'beta', label: 'Beta', render: () => <div data-testid="panel-beta" /> }),
        ]}
      />,
    )
    fireEvent.click(screen.getByTestId('overlay-widget-alpha'))
    expect(screen.getByTestId('overlay-dock-panel')).toHaveAttribute('aria-label', 'Alpha widget')

    fireEvent.click(screen.getByTestId('overlay-widget-beta'))
    const dialogs = screen.getAllByRole('dialog')
    expect(dialogs).toHaveLength(1)
    expect(screen.getByTestId('panel-beta')).toBeDefined()
    expect(screen.queryByTestId('panel-body')).toBeNull()
  })

  it('closes the open panel on Escape', () => {
    render(<OverlayDock widgets={[widget()]} />)
    fireEvent.click(screen.getByTestId('overlay-widget-alpha'))
    expect(screen.getByTestId('overlay-dock-panel')).toBeDefined()

    fireEvent.keyDown(document, { key: 'Escape' })
    expect(screen.queryByTestId('overlay-dock-panel')).toBeNull()
  })

  it('closes the open panel on an outside click', () => {
    render(
      <>
        <button data-testid="outside">outside</button>
        <OverlayDock widgets={[widget()]} />
      </>,
    )
    fireEvent.click(screen.getByTestId('overlay-widget-alpha'))
    expect(screen.getByTestId('overlay-dock-panel')).toBeDefined()

    fireEvent.mouseDown(screen.getByTestId('outside'))
    expect(screen.queryByTestId('overlay-dock-panel')).toBeNull()
  })

  it('shows a count badge when the widget supplies one', () => {
    render(<OverlayDock widgets={[widget({ badge: '3', tone: 'destructive' })]} />)
    expect(screen.getByTestId('overlay-badge-alpha')).toHaveTextContent('3')
    expect(screen.getByTestId('overlay-badge-alpha').className).toContain('bg-destructive/10')
  })

  it('omits the badge when absent', () => {
    render(<OverlayDock widgets={[widget({ badge: null })]} />)
    expect(screen.queryByTestId('overlay-badge-alpha')).toBeNull()
  })

  it('tags operator surfaces on button wrapper and panel', () => {
    render(<OverlayDock widgets={[widget({ role: 'operator' })]} />)
    expect(screen.getByTestId('overlay-widget-alpha').closest('[data-role]')).toHaveAttribute(
      'data-role',
      'operator',
    )

    fireEvent.click(screen.getByTestId('overlay-widget-alpha'))
    expect(screen.getByTestId('overlay-dock-panel')).toHaveAttribute('data-role', 'operator')
  })

  it('renders the rail shell with a status LED (worst tone wins)', () => {
    render(
      <OverlayDock
        widgets={[
          widget({ tone: 'success' }),
          widget({ id: 'beta', label: 'Beta', tone: 'destructive' }),
        ]}
      />,
    )
    expect(screen.getByTestId('overlay-rail')).toBeDefined()
    const led = screen.getByTestId('overlay-led')
    // destructive tenant dominates the lamp (pulsing)
    expect(led.className).toContain('bg-destructive')
    expect(led.className).toContain('animate-pulse')
  })

  it('opens a widget by id via the overlay-dock:open event bridge', () => {
    render(<OverlayDock widgets={[widget()]} />)
    expect(screen.queryByTestId('overlay-dock-panel')).toBeNull()

    act(() => requestOverlayDockOpen('alpha'))
    const panel = screen.getByTestId('overlay-dock-panel')
    expect(panel).toHaveAttribute('aria-label', 'Alpha widget')

    // unknown ids are ignored
    act(() => requestOverlayDockOpen('nope'))
    expect(screen.getByTestId('overlay-dock-panel')).toHaveAttribute('aria-label', 'Alpha widget')
  })
})
