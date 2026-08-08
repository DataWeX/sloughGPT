import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, renderHook, act } from '@testing-library/react'
import React from 'react'

import { ConvSidebarProvider, useConvSidebar } from './ConvSidebarContext'

const CONV_KEY = 'sloughgpt:conv-sidebar-collapsed'
const NAV_KEY = 'sloughgpt:nav-sidebar-collapsed'

function renderProvider() {
  const view = render(
    <ConvSidebarProvider>
      <TestConsumer />
    </ConvSidebarProvider>
  )
  return view
}

function TestConsumer() {
  const ctx = useConvSidebar()
  return (
    <div>
      <span data-testid="open">{String(ctx.open)}</span>
      <span data-testid="convCollapsed">{String(ctx.convCollapsed)}</span>
      <span data-testid="navCollapsed">{String(ctx.navCollapsed)}</span>
      <button onClick={ctx.toggle}>toggle-open</button>
      <button onClick={ctx.toggleConv}>toggle-conv</button>
      <button onClick={ctx.toggleNav}>toggle-nav</button>
      <button onClick={() => ctx.setOpen(true)}>set-open-true</button>
      <button onClick={() => ctx.setConvCollapsed(true)}>set-conv-true</button>
      <button onClick={() => ctx.setNavCollapsed(true)}>set-nav-true</button>
    </div>
  )
}

beforeEach(() => {
  localStorage.clear()
})

afterEach(() => cleanup())

describe('ConvSidebarContext', () => {
  it('defaults to open=false, collapsed=false with empty storage', () => {
    renderProvider()
    expect(screen.getByTestId('open').textContent).toBe('false')
    expect(screen.getByTestId('convCollapsed').textContent).toBe('false')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('false')
  })

  it('reads persisted collapsed flags from localStorage', () => {
    localStorage.setItem(CONV_KEY, 'true')
    localStorage.setItem(NAV_KEY, 'true')
    renderProvider()
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('true')
    // open is not persisted — always starts false
    expect(screen.getByTestId('open').textContent).toBe('false')
  })

  it('ignores non-true storage values', () => {
    localStorage.setItem(CONV_KEY, '1')
    renderProvider()
    expect(screen.getByTestId('convCollapsed').textContent).toBe('false')
  })

  it('toggle flips open', () => {
    renderProvider()
    fireEvent.click(screen.getByText('toggle-open'))
    expect(screen.getByTestId('open').textContent).toBe('true')
    fireEvent.click(screen.getByText('toggle-open'))
    expect(screen.getByTestId('open').textContent).toBe('false')
  })

  it('toggleConv/toggleNav flip collapsed flags and persist', () => {
    renderProvider()
    fireEvent.click(screen.getByText('toggle-conv'))
    fireEvent.click(screen.getByText('toggle-nav'))
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('true')
    expect(localStorage.getItem(CONV_KEY)).toBe('true')
    expect(localStorage.getItem(NAV_KEY)).toBe('true')
  })

  it('setOpen/setConvCollapsed/setNavCollapsed apply exact values', () => {
    renderProvider()
    fireEvent.click(screen.getByText('set-open-true'))
    fireEvent.click(screen.getByText('set-conv-true'))
    fireEvent.click(screen.getByText('set-nav-true'))
    expect(screen.getByTestId('open').textContent).toBe('true')
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('true')
    expect(localStorage.getItem(CONV_KEY)).toBe('true')
    expect(localStorage.getItem(NAV_KEY)).toBe('true')
  })

  it('writes false back to storage when toggled off', () => {
    localStorage.setItem(CONV_KEY, 'true')
    renderProvider()
    fireEvent.click(screen.getByText('toggle-conv'))
    expect(localStorage.getItem(CONV_KEY)).toBe('false')
  })

  it('tolerates localStorage throws (private browsing / SSR)', () => {
    const getItem = vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => { throw new Error('denied') })
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('denied') })
    renderProvider()
    expect(screen.getByTestId('convCollapsed').textContent).toBe('false')
    fireEvent.click(screen.getByText('toggle-conv'))
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    getItem.mockRestore()
    setItem.mockRestore()
  })

  it('useConvSidebar throws outside provider', () => {
    expect(() => renderHook(() => useConvSidebar())).toThrow('useConvSidebar must be used within ConvSidebarProvider')
  })

  it('useConvSidebar exposes the full value shape', () => {
    const { result } = renderHook(() => useConvSidebar(), { wrapper: ConvSidebarProvider })
    expect(typeof result.current.toggle).toBe('function')
    expect(typeof result.current.setOpen).toBe('function')
    expect(typeof result.current.toggleConv).toBe('function')
    expect(typeof result.current.toggleNav).toBe('function')
    expect(typeof result.current.setConvCollapsed).toBe('function')
    expect(typeof result.current.setNavCollapsed).toBe('function')
    expect(result.current.open).toBe(false)
  })

  it('updates through the hook API re-render', () => {
    const { result } = renderHook(() => useConvSidebar(), { wrapper: ConvSidebarProvider })
    act(() => result.current.toggle())
    expect(result.current.open).toBe(true)
    act(() => result.current.toggleConv())
    expect(result.current.convCollapsed).toBe(true)
    expect(localStorage.getItem(CONV_KEY)).toBe('true')
  })
})
