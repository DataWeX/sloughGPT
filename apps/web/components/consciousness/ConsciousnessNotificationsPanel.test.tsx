import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { ConsciousnessNotificationsPanel } from './ConsciousnessNotificationsPanel'
import { LocaleProvider } from '@/hooks/useLocale'
import { useConsciousnessNotifications } from '@/hooks/useConsciousnessNotifications'

vi.mock('@/hooks/useConsciousnessNotifications', () => ({
  useConsciousnessNotifications: vi.fn().mockReturnValue({
    notifications: [],
    unreadCount: 0,
    markAsRead: vi.fn(),
    clearAll: vi.fn(),
    addNotification: vi.fn(),
  }),
}))

vi.mock('@/hooks/useLocale', () => ({
  useLocale: () => ({ t: (k: string) => k, locale: 'en', setLocale: vi.fn() }),
  LocaleProvider: ({ children }: { children: React.ReactNode }) => children,
}))

function renderComponent() {
  return render(
    <LocaleProvider>
      <ConsciousnessNotificationsPanel />
    </LocaleProvider>
  )
}

describe('ConsciousnessNotificationsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders toggle button', () => {
    renderComponent()
    expect(screen.getByLabelText('consciousness_notifications.toggle')).toBeTruthy()
  })

  it('panel is hidden initially', () => {
    renderComponent()
    expect(screen.queryByText('consciousness_notifications.title')).toBeNull()
  })

  it('opens panel on click', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_notifications.toggle'))
    expect(screen.getAllByText('consciousness_notifications.title').length).toBeGreaterThan(0)
  })

  it('shows empty state when no notifications', () => {
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_notifications.toggle'))
    expect(screen.getAllByText('consciousness_notifications.empty').length).toBeGreaterThan(0)
  })

  it('shows notifications when available', () => {
    vi.mocked(useConsciousnessNotifications).mockReturnValue({
      notifications: [
        { id: '1', title: 'Test', body: 'Body', type: 'info', timestamp: Date.now(), read: false },
      ],
      unreadCount: 1,
      markAsRead: vi.fn(),
      clearAll: vi.fn(),
      addNotification: vi.fn(),
    })
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_notifications.toggle'))
    expect(screen.getByText('Test')).toBeTruthy()
    expect(screen.getByText('Body')).toBeTruthy()
  })

  it('shows unread count badge', () => {
    vi.mocked(useConsciousnessNotifications).mockReturnValue({
      notifications: [
        { id: '1', title: 'Test', body: 'Body', type: 'info', timestamp: Date.now(), read: false },
      ],
      unreadCount: 1,
      markAsRead: vi.fn(),
      clearAll: vi.fn(),
      addNotification: vi.fn(),
    })
    renderComponent()
    expect(screen.getByText('1')).toBeTruthy()
  })

  it('shows clear all button when notifications exist', () => {
    vi.mocked(useConsciousnessNotifications).mockReturnValue({
      notifications: [
        { id: '1', title: 'Test', body: 'Body', type: 'info', timestamp: Date.now(), read: false },
      ],
      unreadCount: 1,
      markAsRead: vi.fn(),
      clearAll: vi.fn(),
      addNotification: vi.fn(),
    })
    renderComponent()
    fireEvent.click(screen.getByLabelText('consciousness_notifications.toggle'))
    expect(screen.getAllByText('consciousness_notifications.clear_all').length).toBeGreaterThan(0)
  })
})
