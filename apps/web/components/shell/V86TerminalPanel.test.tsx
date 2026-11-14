import { describe, it, expect, vi, afterEach, beforeEach } from 'vitest'
import { render, screen, cleanup, fireEvent, waitFor } from '@testing-library/react'
import { V86TerminalPanel } from './V86TerminalPanel'

const { useV86Mock, probeImageMock } = vi.hoisted(() => ({
  useV86Mock: vi.fn(),
  probeImageMock: vi.fn(),
}))

vi.mock('@/hooks/useV86', () => ({
  useV86: useV86Mock,
  probeImage: probeImageMock,
  IMAGE_URLS: {
    hda: ['/buildroot/buildroot.img'],
    kernel: ['/buildroot/bzimage68.bin', 'https://i.copy.sh/buildroot-bzimage68.bin'],
    iso: ['/buildroot/buildroot.iso'],
  },
}))

beforeEach(() => {
  useV86Mock.mockReturnValue({
    isBooted: false,
    isSuspended: false,
    error: null,
    init: vi.fn(),
    reboot: vi.fn(),
    reset: vi.fn(),
    suspend: vi.fn(),
    resume: vi.fn(),
  })
  probeImageMock.mockResolvedValue({ available: true, size: 1 })
})

afterEach(() => cleanup())

describe('V86TerminalPanel', () => {
  it('shows booting indicator when not booted', () => {
    const { container } = render(<V86TerminalPanel />)
    const dot = container.querySelector('.animate-pulse')
    expect(dot).not.toBeNull()
  })
  it('renders screen container', () => {
    const { container } = render(<V86TerminalPanel />)
    expect(container.querySelector('[data-testid="v86-screen"]')).not.toBeNull()
  })
  it('applies className', () => {
    const { container } = render(<V86TerminalPanel className="h-96" />)
    const el = container.querySelector('[class*="h-96"]')
    expect(el).not.toBeNull()
  })

  it('renders the boot media selector with all kinds, default auto', async () => {
    render(<V86TerminalPanel />)
    const select = await screen.findByLabelText('Boot media')
    expect(select).toBeTruthy()
    expect((select as HTMLSelectElement).value).toBe('auto')
    const options = Array.from((select as HTMLSelectElement).options)
    expect(options.map((o) => o.value)).toEqual(['auto', 'hda', 'kernel', 'iso'])
    await waitFor(() => expect(probeImageMock).toHaveBeenCalled())
  })

  it('disables kinds whose images probe unavailable', async () => {
    probeImageMock.mockImplementation((url: string) =>
      Promise.resolve({ available: !url.includes('.iso'), size: 1 }),
    )
    render(<V86TerminalPanel />)
    const select = (await screen.findByLabelText('Boot media')) as HTMLSelectElement
    await waitFor(() => {
      const iso = Array.from(select.options).find((o) => o.value === 'iso')
      expect(iso?.disabled).toBe(true)
    })
    const hda = Array.from(select.options).find((o) => o.value === 'hda')
    expect(hda?.disabled).toBe(false)
  })

  it('passes an explicit bootMedia and reboots when media changes', async () => {
    const reboot = vi.fn()
    useV86Mock.mockReturnValue({
      isBooted: true,
      error: null,
      init: vi.fn(),
      reboot,
      reset: vi.fn(),
    })
    render(<V86TerminalPanel />)
    const select = (await screen.findByLabelText('Boot media')) as HTMLSelectElement
    expect(useV86Mock).toHaveBeenLastCalledWith(expect.objectContaining({ bootMedia: undefined }))
    fireEvent.change(select, { target: { value: 'iso' } })
    await waitFor(() => {
      expect(useV86Mock).toHaveBeenLastCalledWith(expect.objectContaining({ bootMedia: 'iso' }))
      expect(reboot).toHaveBeenCalledTimes(1)
    })
  })

  it('does not reboot when the auto default resolves', async () => {
    render(<V86TerminalPanel />)
    const select = (await screen.findByLabelText('Boot media')) as HTMLSelectElement
    expect(useV86Mock).toHaveBeenLastCalledWith(expect.objectContaining({ bootMedia: undefined }))
    expect(select.value).toBe('auto')
  })

  it('renders the desktop gallery selector with the 8 vendored states', async () => {
    render(<V86TerminalPanel />)
    const select = (await screen.findByLabelText('Desktop states')) as HTMLSelectElement
    const values = Array.from(select.options).map((o) => o.value)
    expect(values).toHaveLength(9) // none + 8 desktops
    expect(values[0]).toBe('')
    expect(values).toContain('windows98')
    expect(values).toContain('reactos')
    expect(values).toContain('9front')
  })

  it('selecting a gallery preset reboots into that state', async () => {
    const reboot = vi.fn()
    useV86Mock.mockReturnValue({
      isBooted: true,
      isSuspended: false,
      error: null,
      init: vi.fn(),
      reboot,
      reset: vi.fn(),
      suspend: vi.fn(),
      resume: vi.fn(),
    })
    render(<V86TerminalPanel />)
    const select = (await screen.findByLabelText('Desktop states')) as HTMLSelectElement
    fireEvent.change(select, { target: { value: 'windows98' } })
    await waitFor(() => {
      expect(useV86Mock).toHaveBeenLastCalledWith(
        expect.objectContaining({ statePresetId: 'windows98' }),
      )
      expect(reboot).toHaveBeenCalledTimes(1)
    })
  })

  it('shows a power-on button while suspended and resumes on press', async () => {
    const resume = vi.fn()
    useV86Mock.mockReturnValue({
      isBooted: true,
      isSuspended: true,
      error: null,
      init: vi.fn(),
      reboot: vi.fn(),
      reset: vi.fn(),
      suspend: vi.fn(),
      resume,
    })
    render(<V86TerminalPanel />)
    const button = await screen.findByRole('button', { name: /power on/i })
    expect(screen.getByText('VM Suspended')).toBeTruthy()
    fireEvent.click(button)
    expect(resume).toHaveBeenCalledTimes(1)
  })

  it('hides the standby overlay while running', async () => {
    useV86Mock.mockReturnValue({
      isBooted: true,
      isSuspended: false,
      error: null,
      init: vi.fn(),
      reboot: vi.fn(),
      reset: vi.fn(),
      suspend: vi.fn(),
      resume: vi.fn(),
    })
    render(<V86TerminalPanel />)
    await screen.findByLabelText('Desktop states')
    expect(screen.queryByRole('button', { name: /power on/i })).toBeNull()
  })
})
