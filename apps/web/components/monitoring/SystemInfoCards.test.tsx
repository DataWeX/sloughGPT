import { describe, it, expect, afterEach, vi } from 'vitest'
import { render, screen, cleanup, fireEvent } from '@testing-library/react'
import React from 'react'
import { GpuCard, DiskCard, ServerInfoCard, BatteryCard } from './SystemInfoCards'
import type { BatteryInfo } from '@/lib/system-controller'

const gpu = {
  backend: 'mps',
  device_type: 'gpu',
  vram_gb: 8,
  tier: 'high',
  memory_hint: '{"cuda_available":true}',
}
const disk = { total_gb: 256, used_gb: 128, free_gb: 128, percent: 50 }
const battery: BatteryInfo = {
  status: {
    level: 72,
    is_charging: true,
    is_plugged: true,
    health: 'Good',
    capacity: -1,
    voltage_mv: 7400,
    current_ma: 1200,
    time_to_full_min: 45,
    time_to_empty_min: null,
    source: 'sysfs',
    name: 'BAT0',
    level_band: 'ok',
    updated_at: 0,
  },
  control: { supported: true, writable: true, path: '/sys/x', current_limit: 80, reason: 'ready' },
  advice: {
    limit: 80,
    action: 'cap_at_80',
    reason: 'At 72% and still charging — cap long-term charge to 80%.',
  },
}
const info = {
  platform: 'darwin',
  platform_release: '24.0',
  platform_version: '24.0.0',
  architecture: 'arm64',
  cpu_count: 8,
  processor: 'Apple M1',
}

describe('GpuCard', () => {
  afterEach(cleanup)

  it('renders nothing when gpu is undefined', () => {
    const { container } = render(<GpuCard />)
    expect(container.innerHTML).toBe('')
  })

  it('renders GPU backend and device', () => {
    render(<GpuCard gpu={gpu} />)
    expect(screen.getByText('mps')).toBeDefined()
    expect(screen.getByText('gpu')).toBeDefined()
  })

  it('renders VRAM and tier', () => {
    render(<GpuCard gpu={gpu} />)
    expect(screen.getByText('8 GB')).toBeDefined()
    expect(screen.getByText('high')).toBeDefined()
  })

  it('renders parsed hints from memory_hint JSON', () => {
    render(<GpuCard gpu={gpu} />)
    expect(screen.getByText('cuda available')).toBeDefined()
    expect(screen.getByText('Yes')).toBeDefined()
  })
})

describe('DiskCard', () => {
  afterEach(cleanup)

  it('renders nothing when disk is undefined', () => {
    const { container } = render(<DiskCard />)
    expect(container.innerHTML).toBe('')
  })

  it('renders used and total GB', () => {
    render(<DiskCard disk={disk} />)
    expect(screen.getByText('128.0 GB used')).toBeDefined()
    expect(screen.getByText('256.0 GB total')).toBeDefined()
  })

  it('renders free GB', () => {
    render(<DiskCard disk={disk} />)
    expect(screen.getByText('128.0 GB free')).toBeDefined()
  })

  it('renders percentage', () => {
    render(<DiskCard disk={disk} />)
    expect(screen.getByText('50%')).toBeDefined()
  })
})

describe('ServerInfoCard', () => {
  afterEach(cleanup)

  it('renders nothing when info is undefined', () => {
    const { container } = render(<ServerInfoCard />)
    expect(container.innerHTML).toBe('')
  })

  it('renders platform info', () => {
    render(<ServerInfoCard info={info} />)
    expect(screen.getByText(/darwin/)).toBeDefined()
    expect(screen.getByText(/24.0/)).toBeDefined()
  })

  it('renders architecture', () => {
    render(<ServerInfoCard info={info} />)
    expect(screen.getByText('arm64')).toBeDefined()
  })

  it('renders CPU count', () => {
    render(<ServerInfoCard info={info} />)
    expect(screen.getByText('8')).toBeDefined()
  })
})

describe('BatteryCard', () => {
  afterEach(cleanup)

  it('renders nothing when battery is undefined', () => {
    const { container } = render(<BatteryCard />)
    expect(container.innerHTML).toBe('')
  })

  it('renders level, name, and ETA', () => {
    render(<BatteryCard battery={battery} />)
    expect(screen.getByText('72% · BAT0')).toBeDefined()
    expect(screen.getByText('Full in 45m')).toBeDefined()
  })

  it('shows the applied cap', () => {
    render(<BatteryCard battery={battery} />)
    expect(screen.getByText('Capped at 80%')).toBeDefined()
  })

  it('offers to lift the cap and reports the requested percent', () => {
    const onSetLimit = vi.fn()
    render(<BatteryCard battery={battery} onSetLimit={onSetLimit} />)
    const button = screen.getByRole('button', { name: 'Lift cap' })
    fireEvent.click(button)
    expect(onSetLimit).toHaveBeenCalledWith(100)
  })

  it('offers to cap when no cap is applied', () => {
    const uncapped: BatteryInfo = {
      ...battery,
      control: { ...battery.control, current_limit: 100 },
    }
    const onSetLimit = vi.fn()
    render(<BatteryCard battery={uncapped} onSetLimit={onSetLimit} />)
    fireEvent.click(screen.getByRole('button', { name: 'Cap 80%' }))
    expect(onSetLimit).toHaveBeenCalledWith(80)
  })

  it('hides the control button when the kernel cannot write the threshold', () => {
    const readonly: BatteryInfo = {
      ...battery,
      control: { ...battery.control, writable: false, reason: 'threshold is read-only here' },
    }
    render(<BatteryCard battery={readonly} onSetLimit={vi.fn()} />)
    expect(screen.getByText('Capped at 80%')).toBeDefined()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('summarises unsupported control instead of dumping the reason', () => {
    const unsupported: BatteryInfo = {
      ...battery,
      control: {
        supported: false,
        writable: false,
        path: null,
        current_limit: null,
        reason: 'kernel does not expose charge_control_end_threshold',
      },
    }
    render(<BatteryCard battery={unsupported} onSetLimit={vi.fn()} />)
    expect(screen.getByText('Charge cap unavailable')).toBeDefined()
    expect(screen.queryByRole('button')).toBeNull()
  })

  it('summarises the advice in one line with the full reason in the title', () => {
    render(<BatteryCard battery={battery} />)
    const advice = screen.getByTestId('battery-advice')
    expect(advice.textContent).toBe('Cap charge at 80%')
    expect(advice.getAttribute('title')).toBe(battery.advice.reason)
  })

  it('marks a simulated source', () => {
    const sim: BatteryInfo = { ...battery, status: { ...battery.status, source: 'simulated' } }
    render(<BatteryCard battery={sim} />)
    expect(screen.getByText('simulated')).toBeDefined()
  })
})
