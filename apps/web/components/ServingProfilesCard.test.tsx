// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Badge: ({ children, ...props }: any) => <span data-testid="badge" {...props}>{children}</span>,
  Button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
  Card: ({ children, ...props }: any) => <div data-testid="card" {...props}>{children}</div>,
  CardContent: ({ children, ...props }: any) => <div data-testid="card-content" {...props}>{children}</div>,
  CardDescription: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardFooter: ({ children, ...props }: any) => <div data-testid="card-footer" {...props}>{children}</div>,
  CardHeader: ({ children, ...props }: any) => <div data-testid="card-header" {...props}>{children}</div>,
  CardTitle: ({ children, ...props }: any) => <div data-testid="card-title" {...props}>{children}</div>,
  Skeleton: (props: any) => <div data-testid="skeleton" {...props} />,
}))

vi.mock('@/lib/profiles-controller', () => ({
  profilesController: {
    list: vi.fn().mockResolvedValue([
      { id: 'balanced', name: 'Balanced', tier: 'balanced', description: 'Balanced config', device: 'cpu', quantize: false, quant_bits: 0, inference_pool_size: 2, min_ram_gb: 4 },
      { id: 'gpu_perf', name: 'GPU Performance', tier: 'gpu_performance', description: 'GPU config', device: 'cuda', quantize: true, quant_bits: 8, inference_pool_size: 4, min_ram_gb: 16 },
    ]),
    active: vi.fn().mockResolvedValue({ active_profile_id: 'balanced', profile: null }),
    apply: vi.fn().mockResolvedValue({ profile: { name: 'GPU Performance' }, live_settings: {}, requires_restart: [] }),
  },
}))

vi.mock('@/lib/toast-store', () => ({
  useToastStore: () => vi.fn(),
}))

vi.mock('@/lib/error-utils', () => ({
  extractErrorMessage: () => 'error',
}))

vi.mock('@/components/ConfirmDialog', () => ({
  ConfirmDialog: ({ open, onConfirm, ...props }: any) => open ? (
    <div data-testid="confirm-dialog">
      <button data-testid="confirm-apply" onClick={onConfirm}>Confirm</button>
    </div>
  ) : null,
}))

import { ServingProfilesCard } from './ServingProfilesCard'
import { profilesController } from '@/lib/profiles-controller'

afterEach(() => cleanup())

beforeEach(() => {
  vi.clearAllMocks()
  ;(profilesController.list as any).mockResolvedValue([
    { id: 'balanced', name: 'Balanced', tier: 'balanced', description: 'Balanced config', device: 'cpu', quantize: false, quant_bits: 0, inference_pool_size: 2, min_ram_gb: 4 },
    { id: 'gpu_perf', name: 'GPU Performance', tier: 'gpu_performance', description: 'GPU config', device: 'cuda', quantize: true, quant_bits: 8, inference_pool_size: 4, min_ram_gb: 16 },
  ])
  ;(profilesController.active as any).mockResolvedValue({ active_profile_id: 'balanced', profile: null })
})

describe('ServingProfilesCard', () => {
  it('renders the card title', async () => {
    render(<ServingProfilesCard />)
    expect(screen.getByText('Serving Profile')).toBeTruthy()
  })

  it('shows skeletons while loading', () => {
    ;(profilesController.list as any).mockReturnValue(new Promise(() => {}))
    render(<ServingProfilesCard />)
    expect(screen.getAllByTestId('skeleton').length).toBeGreaterThanOrEqual(1)
  })

  it('renders profiles after loading', async () => {
    render(<ServingProfilesCard />)
    await waitFor(() => {
      expect(screen.getByText('Balanced')).toBeTruthy()
      expect(screen.getByText('GPU Performance')).toBeTruthy()
    })
  })

  it('shows Active badge for active profile', async () => {
    render(<ServingProfilesCard />)
    await waitFor(() => {
      expect(screen.getByText('Active')).toBeTruthy()
    })
  })

  it('shows Apply button for inactive profiles', async () => {
    render(<ServingProfilesCard />)
    await waitFor(() => {
      expect(screen.getByText('Apply')).toBeTruthy()
    })
  })

  it('shows profile count in footer', async () => {
    render(<ServingProfilesCard />)
    await waitFor(() => {
      expect(screen.getByText('2 profiles')).toBeTruthy()
    })
  })

  it('shows confirm dialog when apply clicked', async () => {
    render(<ServingProfilesCard />)
    await waitFor(() => {
      expect(screen.getByText('Apply')).toBeTruthy()
    })
    fireEvent.click(screen.getByText('Apply'))
    expect(screen.getByTestId('confirm-dialog')).toBeTruthy()
  })
})
