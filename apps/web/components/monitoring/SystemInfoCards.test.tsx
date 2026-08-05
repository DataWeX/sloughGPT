import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { GpuCard, DiskCard, ServerInfoCard } from './SystemInfoCards'

const gpu = { backend: 'mps', device_type: 'gpu', vram_gb: 8, tier: 'high', memory_hint: '{"cuda_available":true}' }
const disk = { total_gb: 256, used_gb: 128, free_gb: 128, percent: 50 }
const info = { platform: 'darwin', platform_release: '24.0', platform_version: '24.0.0', architecture: 'arm64', cpu_count: 8, processor: 'Apple M1' }

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
