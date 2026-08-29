// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, fireEvent, cleanup, renderHook, act } from '@testing-library/react'
import React from 'react'

import { ConvSidebarProvider, useConvSidebar } from './ConvSidebarContext'

const CONV_KEY = 'sloughgpt:conv-sidebar-collapsed'
const NAV_KEY = 'sloughgpt:nav-sidebar-collapsed'

const store = new Map<string, string>()

const { chatDBMock } = vi.hoisted(() => {
  const chatDBMock = {
    getKV: vi.fn(async (key: string) => {
      const entry = store.get(key)
      return entry
    }),
    setKV: vi.fn(async (key: string, value: unknown) => {
      store.set(key, String(value))
    }),
    deleteKV: vi.fn(async (key: string) => {
      store.delete(key)
    }),
  }
  return { chatDBMock }
})

vi.mock('@/lib/db', () => ({
  chatDB: chatDBMock,
}))

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
  store.clear()
  chatDBMock.getKV.mockClear()
  chatDBMock.setKV.mockClear()
  chatDBMock.deleteKV.mockClear()
})

afterEach(() => cleanup())

describe('ConvSidebarContext', () => {
  it('defaults to open=false, collapsed=false with empty storage', () => {
    renderProvider()
    expect(screen.getByTestId('open').textContent).toBe('false')
    expect(screen.getByTestId('convCollapsed').textContent).toBe('false')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('false')
  })

  it('reads persisted collapsed flags from chatDB', async () => {
    store.set(CONV_KEY, 'true')
    store.set(NAV_KEY, 'true')
    renderProvider()
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('open').textContent).toBe('false')
  })

  it('ignores non-true storage values', async () => {
    store.set(CONV_KEY, '1')
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

  it('toggleConv/toggleNav flip collapsed flags and persist', async () => {
    renderProvider()
    fireEvent.click(screen.getByText('toggle-conv'))
    fireEvent.click(screen.getByText('toggle-nav'))
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('true')
    expect(store.get(CONV_KEY)).toBe('true')
    expect(store.get(NAV_KEY)).toBe('true')
  })

  it('setOpen/setConvCollapsed/setNavCollapsed apply exact values', async () => {
    renderProvider()
    fireEvent.click(screen.getByText('set-open-true'))
    fireEvent.click(screen.getByText('set-conv-true'))
    fireEvent.click(screen.getByText('set-nav-true'))
    expect(screen.getByTestId('open').textContent).toBe('true')
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
    expect(screen.getByTestId('navCollapsed').textContent).toBe('true')
    expect(store.get(CONV_KEY)).toBe('true')
    expect(store.get(NAV_KEY)).toBe('true')
  })

  it('writes false back to storage when toggled off', async () => {
    store.set(CONV_KEY, 'true')
    renderProvider()
    fireEvent.click(screen.getByText('toggle-conv'))
    expect(store.get(CONV_KEY)).toBe('false')
  })

  it('tolerates chatDB errors', async () => {
    chatDBMock.getKV.mockRejectedValueOnce(new Error('denied'))
    chatDBMock.setKV.mockRejectedValueOnce(new Error('denied'))
    renderProvider()
    expect(screen.getByTestId('convCollapsed').textContent).toBe('false')
    fireEvent.click(screen.getByText('toggle-conv'))
    expect(screen.getByTestId('convCollapsed').textContent).toBe('true')
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
    expect(store.get(CONV_KEY)).toBe('true')
  })
})
