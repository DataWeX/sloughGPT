// @vitest-environment jsdom
import { describe, it, expect, vi, afterEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import React from 'react'

vi.mock('@sloughgpt/strui', () => ({
  cn: (...args: any[]) => args.filter(Boolean).join(' '),
  Card: ({ children, ...props }: any) => <div {...props}>{children}</div>,
  CardHeader: ({ children }: any) => <div>{children}</div>,
  CardTitle: ({ children }: any) => <div>{children}</div>,
  CardDescription: ({ children }: any) => <div>{children}</div>,
  CardContent: ({ children }: any) => <div>{children}</div>,
  Input: (props: any) => <input {...props} />,
  Switch: ({ checked, onCheckedChange, ...props }: any) => (
    <input type="checkbox" checked={checked} onChange={e => onCheckedChange(e.target.checked)} {...props} />
  ),
}))

import { SettingsTrainingCard } from './SettingsTrainingCard'

afterEach(() => cleanup())

describe('SettingsTrainingCard', () => {
  it('renders title', () => {
    render(<SettingsTrainingCard
      preferredModel="" preferredMethod="" maxCheckpoints={10} autoTrainThreshold={0.8}
      autoTrain={false} enableTracking={true}
      onPreferredModelChange={() => {}} onPreferredMethodChange={() => {}}
      onMaxCheckpointsChange={() => {}} onAutoTrainThresholdChange={() => {}}
      onAutoTrainChange={() => {}} onEnableTrackingChange={() => {}}
    />)
    expect(screen.getByText('Training')).toBeTruthy()
  })

  it('renders preferred model input', () => {
    render(<SettingsTrainingCard
      preferredModel="gpt2" preferredMethod="" maxCheckpoints={10} autoTrainThreshold={0.8}
      autoTrain={false} enableTracking={true}
      onPreferredModelChange={() => {}} onPreferredMethodChange={() => {}}
      onMaxCheckpointsChange={() => {}} onAutoTrainThresholdChange={() => {}}
      onAutoTrainChange={() => {}} onEnableTrackingChange={() => {}}
    />)
    expect(screen.getByDisplayValue('gpt2')).toBeTruthy()
  })

  it('renders auto-train toggle', () => {
    render(<SettingsTrainingCard
      preferredModel="" preferredMethod="" maxCheckpoints={10} autoTrainThreshold={0.8}
      autoTrain={true} enableTracking={true}
      onPreferredModelChange={() => {}} onPreferredMethodChange={() => {}}
      onMaxCheckpointsChange={() => {}} onAutoTrainThresholdChange={() => {}}
      onAutoTrainChange={() => {}} onEnableTrackingChange={() => {}}
    />)
    expect(screen.getByText('Auto-train')).toBeTruthy()
  })
})
