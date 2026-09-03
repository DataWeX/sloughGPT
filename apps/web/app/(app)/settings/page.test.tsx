import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, act } from '@testing-library/react'

const { mockUpdateSettings, mockAddToast, mockSetLocale, mockDeleteKV } = vi.hoisted(() => ({
  mockUpdateSettings: vi.fn(),
  mockAddToast: vi.fn(),
  mockSetLocale: vi.fn(),
  mockDeleteKV: vi.fn().mockResolvedValue(undefined),
}))

vi.mock('@/lib/store', () => ({
  useSettings: () => ({
    apiUrl: 'http://localhost:8000',
    hfToken: '',
    defaultTemp: 0.8,
    defaultMaxTokens: 200,
    defaultTopP: 0.9,
    defaultTopK: 50,
    theme: 'light',
    streaming: true,
    customContext: '',
    collapsibleMessageLength: 500,
  }),
  useUpdateSettings: () => mockUpdateSettings,
  DEFAULT_SETTINGS: {
    apiUrl: 'http://localhost:8000',
    hfToken: '',
    defaultTemp: 0.8,
    defaultMaxTokens: 200,
    defaultTopP: 0.9,
    defaultTopK: 50,
    theme: 'light',
    streaming: true,
    customContext: '',
    collapsibleMessageLength: 500,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => ({ addToast: mockAddToast }),
}))

vi.mock('@/hooks/useLiveStatus', () => ({
  useLiveStatus: () => ({ healthLegacy: null }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ locale: 'en', setLocale: mockSetLocale }),
  LOCALES: [
    { code: 'en', name: 'English', flag: '🇺🇸' },
    { code: 'es', name: 'Español', flag: '🇪🇸' },
  ],
}))

vi.mock('@/lib/system-controller', () => ({
  systemController: {
    getDetailedHealth: vi.fn().mockResolvedValue(null),
    getMetrics: vi.fn().mockResolvedValue(null),
    getDisk: vi.fn().mockResolvedValue(null),
    getInfo: vi.fn().mockResolvedValue(null),
    getProcessGuardStatus: vi.fn().mockResolvedValue({ enabled: false, active: false, model_id: null, health: null }),
    setProcessGuardEnabled: vi.fn().mockResolvedValue({ enabled: true, active: false, model_id: null, health: null }),
  },
}))

vi.mock('@/lib/model-controller', () => ({
  modelController: {
    getHealth: vi.fn().mockResolvedValue({ status: 'healthy', model_loaded: true }),
  },
}))

vi.mock('@/lib/download-utils', () => ({
  downloadJson: vi.fn(),
  importFile: vi.fn().mockResolvedValue(null),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: vi.fn().mockReturnValue('Error'),
}))

vi.mock('@/lib/validation-schemas', () => ({
  settingsSchema: {
    shape: {
      apiUrl: { safeParse: (v: string) => ({ success: true, data: v }) },
    },
  },
}))

vi.mock('@/lib/chat-utils', () => ({
  formatUptime: (s: number) => `${s}s`,
  CURRENT_SESSION_KEY: 'current-session',
}))

vi.mock('@/components/ThemeProvider', () => ({
  useTheme: () => ({
    theme: 'purple',
    mode: 'dark',
    palette: 'noir-violet',
    setTheme: vi.fn(),
    setMode: vi.fn(),
    setPalette: vi.fn(),
  }),
  THEMES: [],
}))

vi.mock('@/components/PageContainer', () => ({
  PageContainer: ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div data-testid="page-container">
      <h1>{title}</h1>
      {children}
    </div>
  ),
}))

vi.mock('@/lib/db', () => ({
  chatDB: {
    deleteKV: mockDeleteKV,
  },
}))

vi.mock('@/lib/theme-storage', () => ({
  PALETTE_IDS: ['noir-violet', 'ocean'],
  PALETTE_LABELS: { 'noir-violet': 'Noir Violet', 'ocean': 'Ocean' },
}))

vi.mock('@/lib/config', () => ({
  PUBLIC_API_URL: 'http://localhost:8000',
}))

import SettingsPage from './page'

describe('SettingsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    localStorage.clear()
  })

  afterEach(async () => {
    await act(async () => {})
  })

  it('renders page header', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Settings').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Appearance section with theme controls', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Appearance').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Language card', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Language').length).toBeGreaterThanOrEqual(1)
  })

  it('renders language buttons', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('English').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Connection section', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Connection').length).toBeGreaterThanOrEqual(1)
  })

  it('renders API URL input with default value', () => {
    render(<SettingsPage />)
    const inputs = screen.getAllByLabelText('Service URL')
    expect(inputs.length).toBeGreaterThanOrEqual(1)
    expect(inputs[0]).toHaveValue('http://localhost:8000')
  })

  it('renders Temperature control', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Temperature').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Memory card', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Memory').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Chat commands card', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Chat commands').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Export settings button', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Export settings').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Import settings button', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Import settings').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Clear chat history button', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Clear').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Reset all settings button', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Reset all settings').length).toBeGreaterThanOrEqual(1)
  })

  it('renders theme toggle buttons', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Light').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('Dark').length).toBeGreaterThanOrEqual(1)
    expect(screen.getAllByText('System').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Streaming switch', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Streaming').length).toBeGreaterThanOrEqual(1)
  })

  it('renders custom context textarea', () => {
    render(<SettingsPage />)
    expect(screen.getAllByLabelText('Custom instructions').length).toBeGreaterThanOrEqual(1)
  })

  it('renders collapsible message length control', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Auto-collapse messages longer than').length).toBeGreaterThanOrEqual(1)
  })

  it('renders Process isolation card', () => {
    render(<SettingsPage />)
    expect(screen.getAllByText('Process isolation').length).toBeGreaterThanOrEqual(1)
  })

  it('shows process guard disabled status', async () => {
    render(<SettingsPage />)
    await screen.findAllByText('Disabled')
  })
})
