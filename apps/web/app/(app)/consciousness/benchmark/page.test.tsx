import { render, screen, fireEvent, act } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { LocaleProvider } from '@/hooks/useLocale'
import ConsciousnessBenchmarkPage from './page'

import { createMockController } from '@/lib/__test-helper'
vi.mock('@/lib/consciousness-controller', () => ({
  consciousnessController: createMockController(),
}))
vi.mock('@/hooks/useLocale', () => {
  const mockT = (k: string) => k
  return {
    useLocale: () => ({ t: mockT, locale: 'en', setLocale: vi.fn() }),
    LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
  }
})
vi.mock('@/lib/toast-store', () => ({
  useToastStore: Object.assign(vi.fn((sel: any) => sel({ addToast: vi.fn() })), { getState: () => ({ addToast: vi.fn() }) }),
}))
vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.stubGlobal('fetch', vi.fn())

function renderPage() {
  return render(
    <LocaleProvider>
      <ConsciousnessBenchmarkPage />
    </LocaleProvider>
  )
}

describe('ConsciousnessBenchmarkPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    ;(fetch as any).mockResolvedValue({ ok: true, json: () => Promise.resolve({}) })
    localStorage.clear()
  })

  it('renders page title', () => {
    renderPage()
    expect(screen.getByText('consciousness_benchmark.page_title')).toBeDefined()
  })

  it('shows configuration card', () => {
    renderPage()
    expect(screen.getByText('consciousness_benchmark.config_title')).toBeDefined()
    expect(screen.getByText('consciousness_benchmark.config_desc')).toBeDefined()
  })

  it('shows iterations input with default value', () => {
    const { container } = renderPage()
    const inputs = container.querySelectorAll('input[type="number"]')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
    expect((inputs[0] as HTMLInputElement).value).toBe('50')
  })

  it('shows test type select', () => {
    const { container } = renderPage()
    const select = container.querySelector('select') as HTMLSelectElement
    expect(select).toBeDefined()
    expect(select.value).toBe('reflect')
  })

  it('shows warmup input', () => {
    const { container } = renderPage()
    const inputs = container.querySelectorAll('input[type="number"]')
    expect(inputs.length).toBeGreaterThanOrEqual(2)
    expect((inputs[1] as HTMLInputElement).value).toBe('3')
  })

  it('shows Run Benchmark button', () => {
    renderPage()
    expect(screen.getByText('consciousness_benchmark.run_benchmark')).toBeDefined()
  })

  it('can change iterations value', () => {
    const { container } = renderPage()
    const inputs = container.querySelectorAll('input[type="number"]')
    act(() => {
      fireEvent.change(inputs[0], { target: { value: '100' } })
    })
    expect((inputs[0] as HTMLInputElement).value).toBe('100')
  })

  it('can change test type', () => {
    const { container } = renderPage()
    const select = container.querySelector('select') as HTMLSelectElement
    act(() => {
      fireEvent.change(select, { target: { value: 'process' } })
    })
    expect(select.value).toBe('process')
  })

  it('can change warmup value', () => {
    const { container } = renderPage()
    const inputs = container.querySelectorAll('input[type="number"]')
    act(() => {
      fireEvent.change(inputs[1], { target: { value: '5' } })
    })
    expect((inputs[1] as HTMLInputElement).value).toBe('5')
  })

  it('shows progress text', () => {
    renderPage()
    expect(screen.getByText('consciousness_benchmark.progress')).toBeDefined()
  })
})
