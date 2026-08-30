import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import RootLayout, { metadata, viewport } from './layout'

vi.mock('next/font/local', () => ({
  default: () => ({ variable: '--mock-font' }),
}))

vi.mock('./Providers', () => ({
  Providers: ({ children }: any) => <div data-testid="providers">{children}</div>,
}))

vi.mock('@/components/WebVitals', () => ({ default: () => null }))
vi.mock('@/components/ErrorLifecycle', () => ({ ErrorLifecycle: () => null }))
vi.mock('@/components/ErrorBoundary', () => ({
  ErrorBoundary: ({ children }: any) => <div data-testid="error-boundary">{children}</div>,
}))

afterEach(() => {
  cleanup()
})

describe('RootLayout', () => {
  it('renders children through the provider and error boundary stack', () => {
    render(
      <RootLayout>
        <p>hello</p>
      </RootLayout>,
    )
    expect(screen.getByText('hello')).toBeInTheDocument()
    expect(screen.getByTestId('providers')).toBeInTheDocument()
    expect(screen.getByTestId('error-boundary')).toBeInTheDocument()
  })

  it('applies the font variables to the html element', () => {
    render(
      <RootLayout>
        <span>x</span>
      </RootLayout>,
    )
    const html = document.querySelector('html')
    expect(html).toBeTruthy()
    expect(html?.className).toContain('--mock-font')
    expect(html?.getAttribute('lang')).toBe('en')
  })

  it('injects the theme bootstrap script', () => {
    render(
      <RootLayout>
        <span>x</span>
      </RootLayout>,
    )
    const script = document.querySelector('script')
    expect(script).toBeTruthy()
    expect(script?.textContent ?? '').toContain('theme-')
    expect(script?.textContent ?? '').toContain('localStorage')
  })

  it('exports the platform metadata', () => {
    expect(metadata.title).toBe('Man - AI Platform')
    expect(metadata.icons).toEqual({ icon: '/favicon.svg' })
  })

  it('exports a mobile-safe viewport', () => {
    expect(viewport).toEqual({ width: 'device-width', initialScale: 1, viewportFit: 'cover' })
  })

  it('renders without crashing with no children', () => {
    render(<RootLayout>{null}</RootLayout>)
    expect(document.querySelector('html')).toBeTruthy()
  })
})
