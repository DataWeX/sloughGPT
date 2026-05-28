/** Prefab API hooks — compose useQuery/useMutation with controller methods.

  Usage:
    const models = useModels()
    const souls = useSouls()
    const { mutate: loadModel } = useLoadModel()
    const { mutate: switchSoul } = useSwitchSoul()
    const { data: checkpoints } = useCheckpoints(soulName)
    const { data: status } = useLearnerStatus()
    const { mutate: trainLearner } = useTrainLearner()
    const { data: health } = useHealthStatus()
*/

'use client'

import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { sessionController } from '@/lib/session-controller'
import { useQuery, useMutation, createQueryHook, createMutationHook, useInvalidate } from '@/lib/query'

// ─── Health ─────────────────────────────────────────────────────────────────

export function useHealthStatus(refreshInterval?: number) {
  return     useQuery('health', () => fetch('/health').then(r => r.json()).catch(() => null), {
    staleTime: refreshInterval ?? 5000,
  })
}

// ─── Models ─────────────────────────────────────────────────────────────────

export function useModels() {
  return useQuery('models', () => modelController.list(), { staleTime: 10000 })
}

export function useLoadModel() {
  const invalidate = useInvalidate()
  return useMutation(
    (modelId: string) => modelController.load(modelId),
    {
      onSuccess: () => {
        invalidate('models')
        invalidate('health')
      },
    },
  )
}

// ─── Souls & Checkpoints ────────────────────────────────────────────────────

export function useSouls() {
  return useQuery('souls', () => soulsController.list(), { staleTime: 30000 })
}

export function useCurrentSoul() {
  return useQuery('current-soul', () => soulsController.getCurrent(), { staleTime: 10000 })
}

export function useCheckpoints() {
  return useQuery('checkpoints', () => soulsController.listCheckpoints(), { staleTime: 15000 })
}

export function useSwitchSoul() {
  const invalidate = useInvalidate()
  return useMutation(
    (params: { name: string; checkpointName?: string }) =>
      soulsController.switch(params.name, params.checkpointName),
    {
      onSuccess: () => {
        invalidate('souls')
        invalidate('current-soul')
        invalidate('checkpoints')
      },
    },
  )
}

// ─── Sessions ───────────────────────────────────────────────────────────────

export function useSessions() {
  return useQuery('sessions', () => sessionController.list(), { staleTime: 5000 })
}

export function useCreateSession() {
  const invalidate = useInvalidate()
  return useMutation((title: string) => sessionController.create(title), {
    onSuccess: () => invalidate('sessions'),
  })
}

export function useDeleteSession() {
  const invalidate = useInvalidate()
  return useMutation((id: string) => sessionController.delete(id), {
    onSuccess: () => invalidate('sessions'),
  })
}

// ─── Learner ────────────────────────────────────────────────────────────────

export function useLearnerStatus() {
  return useQuery('learner-status', () =>
    fetch('/learn/status').then(r => r.json()),
    { staleTime: 5000 },
  )
}

export function useTrainLearner() {
  const invalidate = useInvalidate()
  return useMutation(() =>
    fetch('/learn/train', { method: 'POST' }).then(r => r.json()),
    { onSuccess: () => invalidate('learner-status') },
  )
}

export function useLearnerSearch() {
  const invalidate = useInvalidate()
  return useMutation(
    (params: { query: string; max_results?: number }) =>
      fetch('/learn/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: params.query, max_results: params.max_results ?? 5 }) }).then(r => r.json()),
    {
      onSuccess: () => {
        invalidate('learner-status')
        invalidate('knowledge-topics')
      },
    },
  )
}

export function useKnowledgeTopics() {
  return useQuery('knowledge-topics', () =>
    fetch('/learn/knowledge').then(r => r.json()),
    { staleTime: 30000 },
  )
}

export function useKnowledgeQuery(topic: string) {
  return useQuery(['knowledge', topic], () =>
    fetch(`/learn/knowledge?topic=${encodeURIComponent(topic)}`).then(r => r.json()),
    { enabled: !!topic },
  )
}

// ─── Generic one-shot fetcher ───────────────────────────────────────────────

export function useFetch<T>(url: string, options?: { staleTime?: number; enabled?: boolean }) {
  return useQuery<T>(url, () => fetch(url).then(r => r.json()) as Promise<T>, options)
}
