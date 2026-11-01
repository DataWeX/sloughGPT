'use client'

import { modelController } from '@/lib/model-controller'
import { soulsController } from '@/lib/souls-controller'
import { useQuery, useMutation, useInvalidate } from '@/lib/query'

export function useModels() {
  return useQuery('models', () => modelController.list(), { staleTime: 10000 })
}

export function useLoadModel() {
  const invalidate = useInvalidate()
  return useMutation(
    (modelId: string) => modelController.load(modelId),
    { onSuccess: () => invalidate('models') },
  )
}

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
