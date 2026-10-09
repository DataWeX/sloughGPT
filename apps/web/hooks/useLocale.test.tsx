import { describe, expect, it, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, renderHook, act, waitFor } from '@testing-library/react'

const mockGetKV = vi.fn()
const mockSetKV = vi.fn()

vi.mock('@/lib/db', () => ({
  chatDB: {
    getKV: (...a: unknown[]) => mockGetKV(...a),
    setKV: (...a: unknown[]) => mockSetKV(...a),
  },
}))

vi.mock('@/lib/dev-log', () => ({
  trackEvent: vi.fn(),
}))

import { LocaleProvider, useLocale, LOCALES } from './useLocale'

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

beforeEach(() => {
  vi.clearAllMocks()
  document.documentElement.lang = ''
  mockGetKV.mockResolvedValue(undefined)
  mockSetKV.mockResolvedValue(undefined)
})

describe('LOCALES', () => {
  it('exports 5 locales with code, name, flag', () => {
    expect(LOCALES).toHaveLength(5)
    LOCALES.forEach(l => {
      expect(l).toHaveProperty('code')
      expect(l).toHaveProperty('name')
      expect(l).toHaveProperty('flag')
    })
  })
})

describe('t function', () => {
  it('returns English translation for en locale', () => {
    let t: ReturnType<typeof useLocale>['t']
    render(
      <LocaleProvider>
        <LocaleReader fn={(x) => { t = x.t }} />
      </LocaleProvider>
    )
    expect(t!('common.save')).toBe('Save')
  })

  it('returns Spanish translation when locale is es', () => {
    let ctx!: ReturnType<typeof useLocale>
    render(
      <LocaleProvider>
        <LocaleReader fn={(x) => { ctx = x }} />
      </LocaleProvider>
    )
    act(() => { ctx.setLocale('es') })
    expect(ctx.t('common.save')).toBe('Guardar')
  })

  it('returns key itself when no translation exists', () => {
    let t: ReturnType<typeof useLocale>['t']
    render(
      <LocaleProvider>
        <LocaleReader fn={(x) => { t = x.t }} />
      </LocaleProvider>
    )
    expect(t!('nonexistent.key')).toBe('nonexistent.key')
  })

  it('interpolates single param into template', () => {
    let t: ReturnType<typeof useLocale>['t']
    render(
      <LocaleProvider>
        <LocaleReader fn={(x) => { t = x.t }} />
      </LocaleProvider>
    )
    expect(t!('home.apiOffline.body', { url: 'http://localhost:8000' })).toBe('Service at http://localhost:8000 is not reachable')
  })

  it('interpolates multiple params', () => {
    let t: ReturnType<typeof useLocale>['t']
    render(
      <LocaleProvider>
        <LocaleReader fn={(x) => { t = x.t }} />
      </LocaleProvider>
    )
    expect(t!('home.apiOffline.body', { url: 'http://example.com:9000' })).toBe('Service at http://example.com:9000 is not reachable')
  })
})

describe('useLocale', () => {
  it('throws when used outside LocaleProvider', () => {
    expect(() => renderHook(() => useLocale())).toThrow(
      'useLocale must be used within a LocaleProvider'
    )
  })

  it('returns context inside provider', () => {
    const { result } = renderHook(() => useLocale(), { wrapper: LocaleProvider })
    expect(result.current.locale).toBe('en')
    expect(result.current.t('common.save')).toBe('Save')
    expect(result.current.locales).toEqual(['en', 'es', 'fr', 'de', 'zh'])
  })
})

describe('LocaleProvider behavior', () => {
  it('renders children', () => {
    render(
      <LocaleProvider>
        <div data-testid="child">hello</div>
      </LocaleProvider>
    )
    expect(screen.getByTestId('child')).toHaveTextContent('hello')
  })

  it('sets document.documentElement.lang to en by default', () => {
    render(
      <LocaleProvider>
        <div />
      </LocaleProvider>
    )
    expect(document.documentElement.lang).toBe('en')
  })

  it('reads saved locale from chatDB', async () => {
    mockGetKV.mockImplementation((key: string) => {
      if (key === 'man_locale') return Promise.resolve('fr')
      return Promise.resolve(undefined)
    })
    render(
      <LocaleProvider>
        <LocaleReader fn={() => {}} />
      </LocaleProvider>
    )
    await waitFor(() => {
      expect(document.documentElement.lang).toBe('fr')
    })
  })

  it('falls back to en for invalid saved locale', async () => {
    mockGetKV.mockImplementation((key: string) => {
      if (key === 'man_locale') return Promise.resolve('invalid_code')
      return Promise.resolve(undefined)
    })
    render(
      <LocaleProvider>
        <LocaleReader fn={() => {}} />
      </LocaleProvider>
    )
    await waitFor(() => {
      expect(document.documentElement.lang).toBe('en')
    })
  })

  it('setLocale updates lang attribute and chatDB', async () => {
    let ctx!: ReturnType<typeof useLocale>
    render(
      <LocaleProvider>
        <LocaleReader fn={(x) => { ctx = x }} />
      </LocaleProvider>
    )
    act(() => { ctx.setLocale('de') })
    expect(document.documentElement.lang).toBe('de')
    expect(mockSetKV).toHaveBeenCalledWith('man_locale', 'de')
  })
})

function LocaleReader({ fn }: { fn: (ctx: ReturnType<typeof useLocale>) => void }) {
  const ctx = useLocale()
  fn(ctx)
  return null
}
