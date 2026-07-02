import type { TrainingJob } from '@/lib/training-controller'

/** Job status states. */
export type JobStatusState = 'idle' | 'queued' | 'running' | 'success' | 'error' | 'cancelled'

/** Map API training job status to JobStatusState. */
export function trainingJobStatusToStrui(status: TrainingJob['status']): JobStatusState {
  switch (status) {
    case 'pending':
      return 'queued'
    case 'running':
      return 'running'
    case 'completed':
      return 'success'
    case 'failed':
      return 'error'
    case 'cancelled':
      return 'cancelled'
    default:
      return 'idle'
  }
}
