export function traitLabel(key: string): string {
  return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

export function traitColor(value: number): string {
  if (value >= 0.8) return 'text-success'
  if (value >= 0.6) return 'text-primary'
  if (value >= 0.4) return 'text-warning'
  return 'text-muted-foreground'
}

export function sourceDir(path: string): string {
  if (!path) return ''
  const parts = path.replace(/\\/g, '/').split('/')
  const modelsIdx = parts.indexOf('models')
  if (modelsIdx >= 0 && modelsIdx + 1 < parts.length) return parts[modelsIdx + 1]
  return parts[parts.length - 2] || ''
}

export function verdictBadge(verdict: string): { className: string; label: string } {
  switch (verdict) {
    case 'improved': return { className: 'bg-success/10 text-success', label: 'Improved' }
    case 'degraded': return { className: 'bg-destructive/10 text-destructive', label: 'Degraded' }
    case 'neutral': return { className: 'bg-muted text-muted-foreground', label: 'Neutral' }
    default: return { className: 'bg-muted text-muted-foreground', label: verdict }
  }
}
