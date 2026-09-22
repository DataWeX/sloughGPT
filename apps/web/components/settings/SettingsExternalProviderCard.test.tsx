// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

const { mockGetProviderApi, mockUpdateProviderApi, mockAddToast } = vi.hoisted(() => ({
  mockGetProviderApi: vi.fn(),
  mockUpdateProviderApi: vi.fn(),
  mockAddToast: vi.fn(),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    getProviderApi: mockGetProviderApi,
    updateProviderApi: mockUpdateProviderApi,
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: (sel?: (s: { addToast: typeof mockAddToast }) => unknown) =>
    sel ? sel({ addToast: mockAddToast }) : { addToast: mockAddToast },
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: (err: unknown, fallback = 'error') =>
    err instanceof Error ? err.message : fallback,
}))

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  CardFooter: ({ children }: any) => <div>{children}</div>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Input: (props: any) => <input {...props} />,
  Switch: ({ checked, onCheckedChange, ...props }: any) => (
    <input type="checkbox" checked={checked} onChange={(e) => onCheckedChange?.(e.target.checked)} {...props} />
  ),
  Badge: ({ children, ...props }: any) => <span {...props}>{children}</span>,
  Skeleton: ({ className }: any) => <div className={className} data-testid="skeleton" />,
  Spinner: ({ className }: any) => <div className={className} data-testid="spinner" />,
  Select: ({ children, ...props }: any) => <select {...props}>{children}</select>,
  ActionCard: ({ title, children }: any) => <div data-testid="action-card"><h3>{title}</h3>{children}</div>,
  Tabs: ({ children }: any) => <div>{children}</div>,
  TabsList: ({ children }: any) => <div>{children}</div>,
  TabsTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  TabsContent: ({ children }: any) => <div>{children}</div>,
  Textarea: ({ value, onChange, ...props }: any) => <textarea value={value} onChange={onChange} {...props} />,
  Separator: () => <hr />,
  Tooltip: ({ children }: any) => <>{children}</>,
  TooltipTrigger: ({ children }: any) => <>{children}</>,
  TooltipContent: ({ children }: any) => <>{children}</>,
  Progress: ({ value }: any) => <div data-testid="progress" data-value={value} />,
  Avatar: ({ children }: any) => <div>{children}</div>,
  AvatarFallback: ({ children }: any) => <div>{children}</div>,
  ScrollArea: ({ children }: any) => <div>{children}</div>,
  Table: ({ children }: any) => <table>{children}</table>,
  TableBody: ({ children }: any) => <tbody>{children}</tbody>,
  TableRow: ({ children }: any) => <tr>{children}</tr>,
  TableCell: ({ children }: any) => <td>{children}</td>,
  TableHead: ({ children }: any) => <th>{children}</th>,
  TableHeader: ({ children }: any) => <thead>{children}</thead>,
  Collapsible: ({ children }: any) => <div>{children}</div>,
  CollapsibleTrigger: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  CollapsibleContent: ({ children }: any) => <div>{children}</div>,
  Toggle: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  ToggleGroup: ({ children }: any) => <div>{children}</div>,
  ToggleGroupItem: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Command: ({ children }: any) => <div>{children}</div>,
  CommandInput: ({ ...props }: any) => <input {...props} />,
  CommandList: ({ children }: any) => <div>{children}</div>,
  CommandEmpty: ({ children }: any) => <div>{children}</div>,
  CommandGroup: ({ children }: any) => <div>{children}</div>,
  CommandItem: ({ children, ...props }: any) => <div {...props}>{children}</div>,
}))

import { SettingsExternalProviderCard } from './SettingsExternalProviderCard'

const defaults = {
  enabled: false,
  api_url: '',
  model: 'gpt-4o-mini',
  timeout: 60,
  max_retries: 2,
  api_key_set: false,
}

beforeEach(() => {
  vi.clearAllMocks()
  mockGetProviderApi.mockResolvedValue({ ...defaults })
  mockUpdateProviderApi.mockResolvedValue({ ...defaults, enabled: true, api_url: 'https://openrouter.ai/api/v1', api_key_set: true, registered: true })
})

afterEach(() => cleanup())

describe('SettingsExternalProviderCard', () => {
  it('renders title and load-time fields', async () => {
    render(<SettingsExternalProviderCard version="3.0.0" />)
    expect(screen.getByText('External model provider')).toBeTruthy()
    await waitFor(() => expect(screen.getByLabelText('Endpoint URL')).toBeTruthy())
    expect(screen.getByLabelText('Model id')).toBeTruthy()
    expect(screen.getByLabelText('API key')).toBeTruthy()
    expect(screen.getByLabelText('Use external provider')).toBeTruthy()
  })

  it('loads provider settings on mount', async () => {
    mockGetProviderApi.mockResolvedValue({
      ...defaults,
      enabled: true,
      api_url: 'https://api.openai.com/v1',
      model: 'gpt-4o',
      api_key_set: true,
    })
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(mockGetProviderApi).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(screen.getByLabelText('Endpoint URL')).toHaveValue('https://api.openai.com/v1'))
    expect(screen.getByLabelText('Model id')).toHaveValue('gpt-4o')
    expect(screen.getByLabelText('Use external provider')).toBeChecked()
    expect(screen.getByPlaceholderText('•••••••• (saved)')).toBeTruthy()
    expect(screen.getByText('Clear key')).toBeTruthy()
  })

  it('does not prefill API key from the server', async () => {
    mockGetProviderApi.mockResolvedValue({ ...defaults, api_key_set: true })
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByLabelText('API key')).toHaveValue(''))
  })

  it('saves without api_key when key field left blank and a key is already set', async () => {
    mockGetProviderApi.mockResolvedValue({
      ...defaults,
      api_key_set: true,
      enabled: false,
      api_url: 'https://openrouter.ai/api/v1',
    })
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByLabelText('Endpoint URL')).toBeTruthy())
    fireEvent.click(screen.getByLabelText('Use external provider'))
    fireEvent.click(screen.getByText('Save provider'))
    await waitFor(() => expect(mockUpdateProviderApi).toHaveBeenCalled())
    const body = mockUpdateProviderApi.mock.calls[0][0]
    expect(body.enabled).toBe(true)
    expect(body.api_url).toBe('https://openrouter.ai/api/v1')
    expect(body.model).toBe('gpt-4o-mini')
    expect('api_key' in body).toBe(false)
    expect(mockAddToast).toHaveBeenCalledWith('External provider settings saved', 'success')
  })

  it('includes api_key when user types a new key', async () => {
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByLabelText('Endpoint URL')).toBeTruthy())
    fireEvent.change(screen.getByLabelText('Endpoint URL'), { target: { value: 'https://openrouter.ai/api/v1' } })
    fireEvent.change(screen.getByLabelText('API key'), { target: { value: 'sk-or-test' } })
    fireEvent.click(screen.getByLabelText('Use external provider'))
    fireEvent.click(screen.getByText('Save provider'))
    await waitFor(() => expect(mockUpdateProviderApi).toHaveBeenCalled())
    expect(mockUpdateProviderApi.mock.calls[0][0].api_key).toBe('sk-or-test')
  })

  it('blocks enable without a key when none is saved', async () => {
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByLabelText('Endpoint URL')).toBeTruthy())
    fireEvent.change(screen.getByLabelText('Endpoint URL'), { target: { value: 'https://openrouter.ai/api/v1' } })
    fireEvent.click(screen.getByLabelText('Use external provider'))
    fireEvent.click(screen.getByText('Save provider'))
    expect(mockUpdateProviderApi).not.toHaveBeenCalled()
    expect(mockAddToast).toHaveBeenCalledWith('API key required before enabling the external provider', 'error')
  })

  it('rejects non-http endpoint URL when enabling', async () => {
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByLabelText('Endpoint URL')).toBeTruthy())
    fireEvent.change(screen.getByLabelText('Endpoint URL'), { target: { value: 'not-a-url' } })
    fireEvent.click(screen.getByLabelText('Use external provider'))
    fireEvent.click(screen.getByText('Save provider'))
    expect(mockUpdateProviderApi).not.toHaveBeenCalled()
    expect(screen.getByRole('alert').textContent).toMatch(/http\(s\) URL/)
  })

  it('clears API key via dedicated action', async () => {
    mockGetProviderApi.mockResolvedValue({ ...defaults, api_key_set: true, enabled: true, api_url: 'https://openrouter.ai/api/v1' })
    mockUpdateProviderApi.mockResolvedValue({ ...defaults, api_key_set: false, enabled: true, api_url: 'https://openrouter.ai/api/v1' })
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByText('Clear key')).toBeTruthy())
    fireEvent.click(screen.getByText('Clear key'))
    await waitFor(() => expect(mockUpdateProviderApi).toHaveBeenCalledWith({ api_key: '' }))
    expect(mockAddToast).toHaveBeenCalledWith('API key cleared', 'success')
  })

  it('surfaces load errors', async () => {
    mockGetProviderApi.mockRejectedValue(new Error('boom'))
    render(<SettingsExternalProviderCard />)
    await waitFor(() => expect(screen.getByRole('alert').textContent).toBe('boom'))
  })
})
