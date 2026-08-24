import { describe, it, expect, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'
import { ProcessCard } from './ProcessCard'

const baseSystem = { cpu_percent: 12.5, memory_percent: 40.2, memory_available_mb: 8192 }

describe('ProcessCard', () => {
  afterEach(cleanup)

  it('renders empty state when detailed is null', () => {
    render(<ProcessCard detailed={null} />)
    expect(screen.getAllByText('Process info unavailable').length).toBeGreaterThanOrEqual(1)
  })

  it('renders empty state when system is undefined', () => {
    render(<ProcessCard detailed={{}} />)
    expect(screen.getAllByText('Process info unavailable').length).toBeGreaterThanOrEqual(1)
  })

  it('renders CPU and Memory from system block', () => {
    render(<ProcessCard detailed={{ system: baseSystem }} />)
    expect(screen.getByText('12.5%')).toBeDefined()
    expect(screen.getByText('40.2%')).toBeDefined()
  })

  it('hides process rows when no process data is present', () => {
    render(<ProcessCard detailed={{ system: baseSystem }} />)
    expect(screen.queryByText('Open files')).toBeNull()
    expect(screen.queryByText('Threads')).toBeNull()
    expect(screen.queryByText('GC Gen 0')).toBeNull()
  })

  it('renders open files and threads when present', () => {
    render(
      <ProcessCard detailed={{ system: { ...baseSystem, open_files: 42, threads: 8 } }} />,
    )
    expect(screen.getByText('42')).toBeDefined()
    expect(screen.getByText('8')).toBeDefined()
  })

  it('renders GC generation counts when present', () => {
    render(
      <ProcessCard detailed={{ system: { ...baseSystem, gc_gen0: 10, gc_gen1: 2, gc_gen2: 1 } }} />,
    )
    expect(screen.getByText('10')).toBeDefined()
    expect(screen.getByText('2')).toBeDefined()
    expect(screen.getByText('1')).toBeDefined()
  })

  it('renders GPU backend and VRAM when gpu is present', () => {
    render(
      <ProcessCard
        detailed={{ system: baseSystem, gpu: { backend: 'cpu', device_type: 'cpu', vram_gb: 0, tier: 'low' } }}
      />,
    )
    expect(screen.getByText('cpu')).toBeDefined()
    expect(screen.getByText('GPU')).toBeDefined()
  })
})
