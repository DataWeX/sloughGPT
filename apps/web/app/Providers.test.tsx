import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import { Providers } from './Providers'

const mockSessionProvider = vi.hoisted(() =>
  vi.fn(({ children }: { children: React.ReactNode }) => <div data-testid="session">{children}</div>),
)

vi.mock('next-auth/react', () => ({
  SessionProvider: (props: any) => mockSessionProvider(props),
}))
vi.mock('@/components/ThemeProvider', () => ({
  ThemeProvider: ({ children }: any) => <div data-testid="theme">{children}</div>,
}))
vi.mock('@/contexts/ModelContext', () => ({
  ModelProvider: ({ children }: any) => <div data-testid="model">{children}</div>,
}))
vi.mock('@/hooks/useLocale', () => ({
  LocaleProvider: ({ children }: any) => <div data-testid="locale">{children}</div>,
}))

describe('Providers', () => {
  it('renders children nested inside all providers', () => {
    render(
      <Providers>
        <p>content</p>
      </Providers>,
    )
    expect(screen.getByText('content')).toBeInTheDocument()
    expect(screen.getByTestId('session')).toBeInTheDocument()
    expect(screen.getByTestId('theme')).toBeInTheDocument()
    expect(screen.getByTestId('model')).toBeInTheDocument()
    expect(screen.getByTestId('locale')).toBeInTheDocument()
  })

  it('passes refetchOnWindowFocus false to SessionProvider', () => {
    render(
      <Providers>
        <span>child</span>
      </Providers>,
    )
    expect(mockSessionProvider).toHaveBeenCalledWith(
      expect.objectContaining({ refetchOnWindowFocus: false }),
    )
  })

  it('nests providers in order session > theme > model > locale', () => {
    const { container } = render(
      <Providers>
        <em>leaf</em>
      </Providers>,
    )
    const order = Array.from(container.querySelectorAll('[data-testid]')).map(
      (el) => el.getAttribute('data-testid'),
    )
    expect(order).toEqual(['session', 'theme', 'model', 'locale'])
  })

  it('renders multiple children', () => {
    render(
      <Providers>
        <span>one</span>
        <span>two</span>
      </Providers>,
    )
    expect(screen.getByText('one')).toBeInTheDocument()
    expect(screen.getByText('two')).toBeInTheDocument()
  })
})
