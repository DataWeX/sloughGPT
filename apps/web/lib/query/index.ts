/**
 * Request Controller — composable, hook-based HTTP state management.
 *
 * Design:
 * - ``useQuery(key, fetcher)`` — fetch + cache + auto-refetch
 * - ``useMutation(fn)`` — mutate + invalidate related queries
 * - ``useInvalidate()`` — manual cache invalidation
 * - ``createQueryHook`` / ``createMutationHook`` — prefab factories
 *
 * The underlying store (Zustand) provides:
 * - Request deduplication (same key = single in-flight promise)
 * - Configurable stale time and garbage collection
 * - Subscriber tracking (auto-cleanup when no components watch a key)
 *
 * Usage:
 * ```tsx
 * import { useQuery, useMutation, useInvalidate } from '@/lib/query'
 *
 * function ModelsPage() {
 *   const { data: models, isLoading } = useQuery('models', () => api.getModels())
 *   const { mutate: deleteModel } = useMutation(id => api.deleteModel(id), {
 *     invalidateKeys: ['models'],
 *   })
 *   // ...
 * }
 *
 * // Prefab hook for reuse:
 * const useModels = createQueryHook('models', api.getModels)
 * const { data } = useModels({ type: 'local' })
 * ```
 */

export {
  useQuery,
  useMutation,
  useInvalidate,
  useIsFetching,
  createQueryHook,
  createMutationHook,
} from './hooks'
export { invalidateQuery, fetchQuery, getQueryState } from './client'
export type {
  QueryKey,
  QueryStatus,
  QueryOptions,
  QueryResult,
  MutationOptions,
  MutationResult,
} from './types'
