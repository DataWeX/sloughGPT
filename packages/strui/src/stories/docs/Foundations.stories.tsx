import type { Meta, StoryObj } from '@storybook/react'

const meta = {
  title: 'Docs/Foundations',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Color roles, typography, and motion tokens for the pastel lattice shell. Toggle **Surface** (toolbar) for light/dark.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta

export default meta

const swatches = [
  { name: 'background', className: 'bg-background', fg: 'text-foreground' },
  { name: 'foreground', className: 'bg-foreground', fg: 'text-background' },
  { name: 'primary', className: 'bg-primary', fg: 'text-primary-foreground' },
  { name: 'secondary', className: 'bg-secondary', fg: 'text-secondary-foreground' },
  { name: 'muted', className: 'bg-muted', fg: 'text-muted-foreground' },
  { name: 'accent', className: 'bg-accent', fg: 'text-accent-foreground' },
  { name: 'destructive', className: 'bg-destructive', fg: 'text-destructive-foreground' },
  { name: 'success', className: 'bg-success', fg: 'text-primary-foreground' },
  { name: 'warning', className: 'bg-warning', fg: 'text-secondary-foreground' },
  { name: 'border', className: 'bg-border', fg: 'text-foreground' },
  { name: 'card', className: 'bg-card', fg: 'text-card-foreground' },
] as const

export const ColorRoles: StoryObj = {
  render: () => (
    <div className="str-safe-x mx-auto max-w-5xl space-y-10 px-4 py-12 md:px-8">
      <header className="space-y-2 border-b border-border pb-8">
        <p className="text-xs font-bold uppercase tracking-[0.2em] text-primary">Foundations</p>
        <h1 className="sl-h1">Color & surface</h1>
        <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
          Semantic tokens map to Tailwind utilities (<code className="font-mono text-xs">bg-primary</code>, etc.).
          Sharp corners and soft contrast keep the lattice readable in both schemes.
        </p>
      </header>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {swatches.map((s) => (
          <div
            key={s.name}
            className={`flex min-h-[4.5rem] items-end justify-between border border-border p-4 ${s.className}`}
          >
            <span className={`font-mono text-xs font-medium uppercase tracking-wider ${s.fg}`}>{s.name}</span>
          </div>
        ))}
      </div>
    </div>
  ),
}

export const Typography: StoryObj = {
  render: () => (
    <div className="str-safe-x mx-auto max-w-3xl space-y-10 px-4 py-12 md:px-8">
      <header className="space-y-2 border-b border-border pb-8">
        <p className="text-xs font-bold uppercase tracking-[0.2em] text-primary">Foundations</p>
        <h1 className="sl-h1">Typography</h1>
        <p className="text-sm text-muted-foreground">
          Outfit for UI copy; JetBrains Mono for code and metrics. Utilities: <code className="font-mono text-xs">sl-h1</code>,{' '}
          <code className="font-mono text-xs">sl-h2</code>, <code className="font-mono text-xs">sl-muted</code>,{' '}
          <code className="font-mono text-xs">sl-label</code>.
        </p>
      </header>
      <div className="space-y-8">
        <div>
          <p className="sl-label">Heading 1 — sl-h1</p>
          <p className="sl-h1">The quick brown fox jumps over the lazy dog</p>
        </div>
        <div>
          <p className="sl-label">Heading 2 — sl-h2</p>
          <p className="sl-h2">Section title for cards and panels</p>
        </div>
        <div>
          <p className="sl-label">Body</p>
          <p className="text-sm leading-relaxed text-foreground">
            Body text uses a comfortable line height for long-form settings and chat metadata. Muted helper copy sits one step
            back on the lattice.
          </p>
          <p className="sl-muted mt-3">Muted — secondary lines, hints, timestamps.</p>
        </div>
        <div>
          <p className="sl-label">Code</p>
          <pre className="sl-code overflow-x-auto p-4 text-left text-xs sm:text-sm">
            <code>{`POST /chat/stream\nmodel_id: "sloughgpt"`}</code>
          </pre>
        </div>
      </div>
    </div>
  ),
}
