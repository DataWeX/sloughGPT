import type { JobStatusState } from '@sloughgpt/strui'

import type { TrainingJob } from '@/lib/training-controller'

/** Map API training job status to strui `JobStatus` states. */
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
