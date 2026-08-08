import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, fireEvent, cleanup } from '@testing-library/react'
import React from 'react'
import { AlertPanel } from './AlertPanel'

function MockNotification(this: any, title: string) {
  this.title = title
}
MockNotification.permission = 'default'
MockNotification.requestPermission = vi.fn()

function renderPanel(props: Partial<Parameters<typeof AlertPanel>[0]> = {}) {
  const base = {
    cpuThreshold: 80,
    memThreshold: 75,
    onCpuThresholdChange: vi.fn(),
    onMemThresholdChange: vi.fn(),
    alerts: [] as Array<{ time: string; type: string; value: number }>,
  }
  return render(<AlertPanel {...base} {...props} />)
}

describe('AlertPanel', () => {
  afterEach(() => {
    cleanup()
    delete (window as any).Notification
  })

  it('renders cpu and mem thresholds with values', () => {
    renderPanel()
    expect(screen.getByText('CPU')).toBeDefined()
    expect(screen.getByText('MEM')).toBeDefined()
    expect(screen.getByText('80%')).toBeDefined()
    expect(screen.getByText('75%')).toBeDefined()
  })

  it('calls onCpuThresholdChange when CPU slider changes', () => {
    const onCpuThresholdChange = vi.fn()
    renderPanel({ onCpuThresholdChange })
    const slider = screen.getAllByRole('slider')[0]
    fireEvent.change(slider, { target: { value: '90' } })
    expect(onCpuThresholdChange).toHaveBeenCalledWith(90)
  })

  it('calls onMemThresholdChange when MEM slider changes', () => {
    const onMemThresholdChange = vi.fn()
    renderPanel({ onMemThresholdChange })
    const slider = screen.getAllByRole('slider')[1]
    fireEvent.change(slider, { target: { value: '60' } })
    expect(onMemThresholdChange).toHaveBeenCalledWith(60)
  })

  it('shows no recent alerts section when there are none', () => {
    renderPanel({ alerts: [] })
    expect(screen.queryByText('Recent alerts')).toBeNull()
  })

  it('renders up to 5 recent alerts', () => {
    const alerts = Array.from({ length: 7 }, (_, i) => ({ time: `t${i}`, type: 'CPU', value: 90 + i }))
    renderPanel({ alerts })
    expect(screen.getByText('Recent alerts')).toBeDefined()
    expect(screen.getByText('CPU 94%')).toBeDefined()
    expect(screen.queryByText('CPU 95%')).toBeNull()
  })

  it('shows notifications toggle when Notification API exists', () => {
    ;(window as any).Notification = MockNotification
    renderPanel()
    expect(screen.getByText('Enable notifications')).toBeDefined()
  })

  it('requests permission when not granted', () => {
    ;(window as any).Notification = MockNotification
    renderPanel()
    fireEvent.click(screen.getByText('Enable notifications'))
    expect(MockNotification.requestPermission).toHaveBeenCalled()
  })

  it('shows "Notifications on" when permission granted', () => {
    MockNotification.permission = 'granted'
    ;(window as any).Notification = MockNotification
    renderPanel()
    expect(screen.getByText('Notifications on')).toBeDefined()
  })
})
