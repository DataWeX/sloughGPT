import { describe, it, expect, vi } from 'vitest'
import { render, screen } from '@testing-library/react'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn().mockResolvedValue(null),
}))

vi.mock('@/lib/settings-controller', () => ({
  settingsController: {
    getAutoTrainSettingsStatus: vi.fn().mockResolvedValue(null),
    updateAutoTrainSettingsConfig: vi.fn().mockResolvedValue(undefined),
  },
}))

import AutoTrainPage from './page'

describe('AutoTrainPage', () => {
  it('renders the auto-train dashboard', () => {
    render(<AutoTrainPage />)
    expect(screen.getAllByText('Auto-Train').length).toBeGreaterThanOrEqual(1)
    expect(screen.getByText(/Automatic training from conversation pairs/)).toBeDefined()
  })
})