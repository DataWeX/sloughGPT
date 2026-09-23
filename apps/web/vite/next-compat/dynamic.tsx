import { lazy, Suspense, type ComponentType, type ReactNode } from 'react'

type DynamicOptions = {
  loading?: ComponentType<Record<string, unknown>>
  ssr?: boolean
  chunks?: string[]
}

type LoaderResult =
  { default: ComponentType<Record<string, unknown>> } | ComponentType<Record<string, unknown>>

/** next/dynamic → React.lazy + Suspense (ssr:false is a no-op in the browser). */
export default function dynamic(
  loader: () => Promise<LoaderResult>,
  options: DynamicOptions = {},
): ComponentType<Record<string, unknown>> {
  const Lazy = lazy(async () => {
    const mod = (await loader()) as LoaderResult
    if (typeof mod === 'function') return { default: mod }
    if (mod && typeof mod === 'object' && 'default' in mod && typeof mod.default === 'function') {
      return { default: mod.default }
    }
    // Named-export style: treat the module itself as needing .default fallback
    const named = mod as unknown as Record<string, unknown>
    const firstFn = Object.values(named).find((v) => typeof v === 'function') as
      ComponentType<Record<string, unknown>> | undefined
    if (firstFn) return { default: firstFn }
    throw new Error('next/dynamic compat: loader did not yield a component')
  })
  const Loading = options.loading
  return function DynamicClient(props: Record<string, unknown>) {
    return (
      <Suspense fallback={Loading ? <Loading {...props} /> : (null as unknown as ReactNode)}>
        <Lazy {...props} />
      </Suspense>
    )
  }
}
