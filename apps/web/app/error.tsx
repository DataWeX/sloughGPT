'use client'

export default function RootError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-4 p-8">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-destructive/10">
        <span className="text-xl font-bold text-destructive">!</span>
      </div>
      <h1 className="text-lg font-semibold">Application error</h1>
      <p className="text-sm text-muted-foreground text-center max-w-sm">
        {error.message || 'The application encountered an unexpected error.'}
      </p>
      <button
        onClick={reset}
        className="inline-flex items-center justify-center rounded-md bg-primary text-primary-foreground px-4 h-9 text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        Reload
      </button>
    </div>
  )
}
