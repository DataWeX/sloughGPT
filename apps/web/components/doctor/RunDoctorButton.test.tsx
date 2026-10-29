// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'

vi.mock('@/lib/http-client', () => ({
  apiGet: vi.fn(),
  apiPost: vi.fn(),
}))
vi.mock('@/lib/toast-store', () => ({
  useToastStore: (selector: (s: { addToast: (m: string, t?: string) => void }) => unknown) =>
    selector({ addToast: vi.fn() }),
}))

import { apiPost } from '@/lib/http-client'
import { RunDoctorButton } from './RunDoctorButton'
import type { DoctorReport } from './types'

const post = vi.mocked(apiPost)

const REPORT = {
  schema_version: 1,
  ts: 1,
  targets: {},
  probes: [],
  findings: [],
  summary: { total: 0, by_severity: { ok: 0, info: 0, warn: 0, critical: 0 } },
  overall: 'ok',
} as unknown as DoctorReport

afterEach(cleanup)

describe('RunDoctorButton', () => {
  beforeEach(() => {
    post.mockReset()
  })

  it('calls POST /doctor/run on click', async () => {
    post.mockResolvedValue({ report: REPORT })
    render(<RunDoctorButton />)
    fireEvent.click(screen.getByTestId('run-doctor-button'))
    await waitFor(() => expect(post).toHaveBeenCalledTimes(1))
    expect(post.mock.calls[0][0]).toBe('/doctor/run')
  })

  it('reports the fresh report to onCompleted', async () => {
    post.mockResolvedValue({ report: REPORT })
    const onCompleted = vi.fn()
    render(<RunDoctorButton onCompleted={onCompleted} />)
    fireEvent.click(screen.getByTestId('run-doctor-button'))
    await waitFor(() => expect(onCompleted).toHaveBeenCalledWith(REPORT))
  })

  it('is disabled and busy while the run is in flight', async () => {
    let resolve: (v: unknown) => void = () => {}
    post.mockImplementation(
      () => new Promise((r) => { resolve = r as (v: unknown) => void }) as never,
    )
    render(<RunDoctorButton />)
    const button = screen.getByTestId('run-doctor-button')
    expect(button).toBeEnabled()
    fireEvent.click(button)
    await waitFor(() => expect(button).toBeDisabled())
    expect(button).toHaveAttribute('aria-busy', 'true')
    expect(button).toHaveTextContent('Running…')
    resolve({ report: REPORT })
    await waitFor(() => expect(button).toBeEnabled())
  })

  it('shows an inline error when the run fails', async () => {
    post.mockRejectedValue(new Error('Connection unavailable — server may be starting up'))
    render(<RunDoctorButton />)
    fireEvent.click(screen.getByTestId('run-doctor-button'))
    const alert = await screen.findByTestId('run-doctor-error')
    expect(alert).toHaveAttribute('role', 'alert')
    expect(alert).toHaveTextContent('Connection unavailable')
    expect(screen.getByTestId('run-doctor-button')).toBeEnabled()
  })

  it('retries are disabled for the run (single POST, no sweep storms)', async () => {
    post.mockResolvedValue({ report: REPORT })
    render(<RunDoctorButton />)
    fireEvent.click(screen.getByTestId('run-doctor-button'))
    await waitFor(() => expect(post).toHaveBeenCalled())
    expect(post.mock.calls[0][2]).toMatchObject({ skipCircuitBreaker: true })
  })
})
