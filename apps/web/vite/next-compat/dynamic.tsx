import { lazy, Suspense, type ComponentType, type ReactNode } from 'react'

type DynamicOptions = {
  loading?: ComponentType<Record<string, unknown>>
  ssr?: boolean
  chunks?: string[]
}

/* eslint-disable @typescript-eslint/no-explicit-any */
type AnyComponent = ComponentType<any>

type LoaderResult = AnyComponent | { default: AnyComponent } | Record<string, unknown>

/**
 * next/dynamic → React.lazy + Suspense (ssr:false is a no-op in the browser).
 * Loader may resolve to a component, a module with default, or a named-export module.
 */
export default function dynamic<P = any>(
  loader: () => Promise<LoaderResult>,
  options: DynamicOptions = {},
): ComponentType<P> {
  const Lazy = lazy(async () => {
    const mod = (await loader()) as LoaderResult
    if (typeof mod === 'function') return { default: mod as AnyComponent }
    if (
      mod &&
      typeof mod === 'object' &&
      'default' in mod &&
      typeof (mod as { default?: unknown }).default === 'function'
    ) {
      return { default: (mod as { default: AnyComponent }).default }
    }
    const named = mod as unknown as Record<string, unknown>
    const firstFn = Object.values(named).find((v) => typeof v === 'function') as
      AnyComponent | undefined
    if (firstFn) return { default: firstFn }
    throw new Error('next/dynamic compat: loader did not yield a component')
  })
  const Loading = options.loading
  return function DynamicClient(props: P) {
    return (
      <Suspense
        fallback={
          Loading ? (
            <Loading {...(props as Record<string, unknown>)} />
          ) : (
            (null as unknown as ReactNode)
          )
        }
      >
        <Lazy {...(props as Record<string, unknown>)} />
      </Suspense>
    )
  } as ComponentType<P>
}
