'use client'

// Shared label/value row for the VM result and training cards. Extracted from
// page.tsx by the page-decomposition refactor.

export function StatusRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between text-xs">
      <span className="text-muted-foreground">{label}</span>
      <span className="font-mono">{value}</span>
    </div>
  )
}
