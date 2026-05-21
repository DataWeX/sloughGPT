import { describe, expect, it } from 'vitest'

import { trainingJobStatusToStrui } from './training-status'

describe('trainingJobStatusToStrui', () => {
  it('maps API statuses to strui JobStatus states', () => {
    expect(trainingJobStatusToStrui('pending')).toBe('queued')
    expect(trainingJobStatusToStrui('running')).toBe('running')
    expect(trainingJobStatusToStrui('completed')).toBe('success')
    expect(trainingJobStatusToStrui('failed')).toBe('error')
    expect(trainingJobStatusToStrui('cancelled')).toBe('cancelled')
  })

  it('falls back to idle for unexpected runtime values', () => {
    expect(
      trainingJobStatusToStrui('not-a-real-status' as unknown as import('./training-controller').TrainingJob['status']),
    ).toBe('idle')
  })
})
