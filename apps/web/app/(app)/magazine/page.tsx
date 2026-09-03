'use client'

import { useState } from 'react'
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  Button,
  Badge,
  Input,
  Separator,
  Switch,
  Checkbox,
  Progress,
  cn,
} from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'

/* ──────────────────────────────────────────────────────────────────
 * Color token definitions — light & dark mode RGB triples
 * ────────────────────────────────────────────────────────────────── */

const lightColors = [
  { name: '--background', rgb: '248 246 252', label: 'Background', desc: 'Page background' },
  { name: '--foreground', rgb: '25 22 36', label: 'Foreground', desc: 'Primary text' },
  { name: '--card', rgb: '255 255 255', label: 'Card', desc: 'Card/panel backgrounds' },
  { name: '--card-foreground', rgb: '25 22 36', label: 'Card FG', desc: 'Text on cards' },
  { name: '--primary', rgb: '124 82 196', label: 'Primary', desc: 'Buttons, links, active' },
  { name: '--primary-foreground', rgb: '250 248 255', label: 'Primary FG', desc: 'Text on primary' },
  { name: '--secondary', rgb: '237 232 248', label: 'Secondary', desc: 'Secondary backgrounds' },
  { name: '--secondary-foreground', rgb: '42 37 55', label: 'Secondary FG', desc: 'Text on secondary' },
  { name: '--muted', rgb: '244 242 248', label: 'Muted', desc: 'Subtle backgrounds' },
  { name: '--muted-foreground', rgb: '130 122 150', label: 'Muted FG', desc: 'Captions, secondary' },
  { name: '--accent', rgb: '236 145 95', label: 'Accent', desc: 'Highlights, warnings' },
  { name: '--accent-foreground', rgb: '250 248 255', label: 'Accent FG', desc: 'Text on accent' },
  { name: '--border', rgb: '228 224 242', label: 'Border', desc: 'Borders, dividers' },
  { name: '--input', rgb: '228 224 242', label: 'Input', desc: 'Input borders' },
  { name: '--ring', rgb: '124 82 196', label: 'Ring', desc: 'Focus rings' },
  { name: '--success', rgb: '52 176 125', label: 'Success', desc: 'Success states' },
  { name: '--warning', rgb: '236 168 60', label: 'Warning', desc: 'Warning states' },
  { name: '--destructive', rgb: '220 80 90', label: 'Destructive', desc: 'Errors, destructive' },
] as const

const darkColors = [
  { name: '--background', rgb: '17 15 24', label: 'Background', desc: 'Deep charcoal-violet' },
  { name: '--foreground', rgb: '238 234 248', label: 'Foreground', desc: 'Primary text' },
  { name: '--card', rgb: '28 25 38', label: 'Card', desc: 'Card/panel backgrounds' },
  { name: '--card-foreground', rgb: '238 234 248', label: 'Card FG', desc: 'Text on cards' },
  { name: '--primary', rgb: '192 170 244', label: 'Primary', desc: 'Lilac primary' },
  { name: '--primary-foreground', rgb: '25 22 36', label: 'Primary FG', desc: 'Text on primary' },
  { name: '--secondary', rgb: '50 44 68', label: 'Secondary', desc: 'Secondary backgrounds' },
  { name: '--secondary-foreground', rgb: '238 234 248', label: 'Secondary FG', desc: 'Text on secondary' },
  { name: '--muted', rgb: '38 34 52', label: 'Muted', desc: 'Subtle backgrounds' },
  { name: '--muted-foreground', rgb: '150 140 172', label: 'Muted FG', desc: 'Captions, secondary' },
  { name: '--accent', rgb: '240 176 130', label: 'Accent', desc: 'Warm peach accent' },
  { name: '--accent-foreground', rgb: '25 22 36', label: 'Accent FG', desc: 'Text on accent' },
  { name: '--border', rgb: '52 46 72', label: 'Border', desc: 'Borders, dividers' },
  { name: '--input', rgb: '52 46 72', label: 'Input', desc: 'Input borders' },
  { name: '--ring', rgb: '192 170 244', label: 'Ring', desc: 'Focus rings' },
  { name: '--success', rgb: '72 192 140', label: 'Success', desc: 'Success states' },
  { name: '--warning', rgb: '240 192 80', label: 'Warning', desc: 'Warning states' },
  { name: '--destructive', rgb: '235 100 110', label: 'Destructive', desc: 'Errors, destructive' },
] as const

const accentThemes = [
  { name: 'blue', label: 'Blue', rgb: '90 130 220' },
  { name: 'purple', label: 'Purple', rgb: '155 108 214' },
  { name: 'pink', label: 'Pink', rgb: '218 130 170' },
  { name: 'red', label: 'Red', rgb: '230 120 130' },
  { name: 'orange', label: 'Orange', rgb: '236 155 90' },
  { name: 'green', label: 'Green', rgb: '72 178 130' },
  { name: 'teal', label: 'Teal', rgb: '72 166 200' },
] as const

const typographyScale = [
  { role: 'Page Title', class: 'text-2xl md:text-3xl font-semibold', sample: 'Noir Violet', font: 'font-[family-name:var(--font-rubik)]', weight: '600' },
  { role: 'Section Title', class: 'text-base font-medium', sample: 'Design System', font: 'font-[family-name:var(--font-rubik)]', weight: '500' },
  { role: 'Body', class: 'text-sm', sample: 'Warm, sophisticated, technical.', font: 'font-[family-name:var(--font-rubik)]', weight: '400' },
  { role: 'Caption', class: 'text-xs text-muted-foreground', sample: 'Last updated 2 minutes ago', font: 'font-[family-name:var(--font-rubik)]', weight: '400' },
  { role: 'Label', class: 'text-xs font-medium uppercase tracking-wider', sample: 'STATUS', font: 'font-[family-name:var(--font-rubik)]', weight: '500' },
  { role: 'Badge', class: 'text-[10px] font-medium', sample: 'Active', font: 'font-[family-name:var(--font-rubik)]', weight: '500' },
  { role: 'Mono', class: 'font-mono text-xs', sample: 'rgb(124, 82, 196)', font: 'font-[family-name:var(--font-jetbrains-mono)]', weight: '400' },
] as const

const spacingScale = [
  { token: 'gap-1', px: '4px', class: 'gap-1' },
  { token: 'gap-2', px: '8px', class: 'gap-2' },
  { token: 'gap-3', px: '12px', class: 'gap-3' },
  { token: 'gap-4', px: '16px', class: 'gap-4' },
  { token: 'gap-6', px: '24px', class: 'gap-6' },
]

const radiusScale = [
  { token: 'rounded-none', px: '0px' },
  { token: 'rounded-sm', px: '2px' },
  { token: 'rounded', px: '4px' },
  { token: 'rounded-md', px: '6px' },
  { token: 'rounded-lg', px: '10px' },
  { token: 'rounded-xl', px: '14px' },
]

/* ──────────────────────────────────────────────────────────────────
 * Swatch — small reusable color swatch component
 * ────────────────────────────────────────────────────────────────── */

function Swatch({
  rgb,
  label,
  description,
  size = 'md',
}: {
  rgb: string
  label: string
  description?: string
  size?: 'sm' | 'md' | 'lg'
}) {
  const sizeClasses = {
    sm: 'h-8 w-8 rounded',
    md: 'h-12 w-12 rounded-md',
    lg: 'h-16 w-16 rounded-lg',
  }
  return (
    <div className="flex flex-col items-center gap-1.5">
      <div
        className={cn(sizeClasses[size], 'border border-border/40 shadow-sm transition-transform hover:scale-110')}
        style={{ backgroundColor: `rgb(${rgb})` }}
        role="img"
        aria-label={`${label}: rgb(${rgb})`}
      />
      <div className="text-center">
        <div className="text-[10px] font-medium">{label}</div>
        <div className="font-mono text-[10px] text-muted-foreground">{rgb}</div>
        {description && <div className="text-[10px] text-muted-foreground">{description}</div>}
      </div>
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────────
 * SectionHeader — editorial section divider
 * ────────────────────────────────────────────────────────────────── */

function SectionHeader({ number, title }: { number: string; title: string }) {
  return (
    <div className="flex items-baseline gap-3 pt-2">
      <span className="font-mono text-xs text-muted-foreground">{number}</span>
      <h2 className="text-base font-medium">{title}</h2>
      <div className="h-px flex-1 bg-border" />
    </div>
  )
}

/* ──────────────────────────────────────────────────────────────────
 * MagazinePage
 * ────────────────────────────────────────────────────────────────── */

export default function MagazinePage() {
  const [interactiveHover, setInteractiveHover] = useState(false)
  const [switchOn, setSwitchOn] = useState(false)
  const [checkboxOn, setCheckboxOn] = useState(false)

  return (
    <PageContainer
      title={
        <span className="sl-h1">
          <span className="text-primary">Noir Violet</span> Brand Guide
        </span>
      }
      subtitle="A design system reference for sloughGPT"
    >
      <style>{`
        @media print {
          .sl-page { overflow: visible !important; height: auto !important; }
          .sl-app-shell { height: auto !important; overflow: visible !important; }
          .sl-app-body { overflow: visible !important; }
          .sl-app-main { overflow: visible !important; }
          .sl-app-content { overflow: visible !important; }
          .no-print { display: none !important; }
          * { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
          @page { margin: 1.5cm; size: A4; }
          .magazine-card { break-inside: avoid; page-break-inside: avoid; }
        }
      `}</style>

      {/* ─── 1. Hero / Cover ──────────────────────────────────── */}
      <Card className="magazine-card overflow-hidden">
        <div className="relative flex flex-col items-center justify-center px-6 py-12 text-center sm:py-16">
          {/* Background accent glow */}
          <div
            className="pointer-events-none absolute inset-0 opacity-[0.07]"
            style={{
              background:
                'radial-gradient(ellipse at 50% 30%, rgb(var(--primary)) 0%, transparent 70%)',
            }}
          />
          <div className="relative z-10 space-y-4">
            {/* Primary color swatch */}
            <div className="flex justify-center">
              <div
                className="h-20 w-20 rounded-xl shadow-lg transition-transform hover:scale-105"
                style={{ backgroundColor: 'rgb(124, 82, 196)' }}
                role="img"
                aria-label="Primary violet color swatch"
              />
            </div>
            <div>
              <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">
                <span style={{ color: 'rgb(124, 82, 196)' }}>Noir</span>{' '}
                <span style={{ color: 'rgb(var(--foreground))' }}>Violet</span>
              </h1>
              <p className="mt-2 text-sm" style={{ color: 'rgb(var(--muted-foreground))' }}>
                Warm, sophisticated, technical.
              </p>
            </div>
            <div className="flex items-center justify-center gap-2">
              <Badge variant="default">v1.0</Badge>
              <Badge variant="secondary">Design System</Badge>
              <Badge
                variant="outline"
                style={{
                  borderColor: 'rgb(236, 145, 95)',
                  color: 'rgb(236, 145, 95)',
                }}
              >
                Accent: Terracotta
              </Badge>
            </div>
            <p className="mx-auto max-w-md text-xs" style={{ color: 'rgb(var(--muted-foreground))' }}>
              A calm confidence. Not sterile, not playful. A tool that respects its user.
            </p>
          </div>
        </div>
      </Card>

      {/* ─── 2. Color Palette ─────────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="02" title="Color Palette" />

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Light Mode</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4 sm:grid-cols-6 md:grid-cols-9">
              {lightColors.map((c) => (
                <Swatch key={c.name} rgb={c.rgb} label={c.label} description={c.desc} size="md" />
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Dark Mode</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-4 gap-4 sm:grid-cols-6 md:grid-cols-9">
              {darkColors.map((c) => (
                <Swatch key={c.name} rgb={c.rgb} label={c.label} description={c.desc} size="md" />
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* ─── 3. Typography Scale ──────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="03" title="Typography Scale" />

        <Card className="magazine-card">
          <CardContent className="pt-4">
            <div className="space-y-4">
              {typographyScale.map((t) => (
                <div key={t.role} className="flex flex-col gap-1 border-b border-border/40 pb-3 last:border-0 last:pb-0">
                  <div className="flex items-baseline justify-between">
                    <span className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
                      {t.role}
                    </span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      {t.class}
                    </span>
                  </div>
                  <p className={cn(t.class, t.font)}>
                    {t.sample}
                  </p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Font Families</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-border/40 pb-2">
                <div>
                  <div className="text-sm font-medium" style={{ fontFamily: 'var(--font-rubik), system-ui, sans-serif' }}>
                    Rubik
                  </div>
                  <div className="text-xs text-muted-foreground">Body text &mdash; system-ui fallback</div>
                </div>
                <Badge variant="secondary">--font-rubik</Badge>
              </div>
              <div className="flex items-center justify-between border-b border-border/40 pb-2">
                <div>
                  <div className="text-sm font-medium" style={{ fontFamily: 'var(--font-lato), system-ui, sans-serif' }}>
                    Lato
                  </div>
                  <div className="text-xs text-muted-foreground">Numeric data &mdash; system-ui fallback</div>
                </div>
                <Badge variant="secondary">--font-lato</Badge>
              </div>
              <div className="flex items-center justify-between">
                <div>
                  <div className="font-mono text-sm font-medium">
                    JetBrains Mono
                  </div>
                  <div className="text-xs text-muted-foreground">Code &mdash; ui-monospace fallback</div>
                </div>
                <Badge variant="secondary">--font-jetbrains-mono</Badge>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* ─── 4. Component Gallery ─────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="04" title="Component Gallery" />

        {/* Buttons */}
        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Buttons</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm">Primary</Button>
                <Button size="sm" variant="secondary">Secondary</Button>
                <Button size="sm" variant="destructive">Destructive</Button>
                <Button size="sm" variant="outline">Outline</Button>
                <Button size="sm" variant="ghost">Ghost</Button>
                <Button size="sm" variant="link">Link</Button>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" disabled>Disabled</Button>
                <Button size="sm" variant="outline" disabled>Disabled</Button>
              </div>
              <p className="text-xs text-muted-foreground">
                All buttons use <code className="font-mono text-[10px] bg-muted px-1 rounded">h-9</code> default,{' '}
                <code className="font-mono text-[10px] bg-muted px-1 rounded">h-11</code> for primary CTAs.
                Touch target minimum: 24x24px.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Badges */}
        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Badges</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="default">Default</Badge>
              <Badge variant="secondary">Secondary</Badge>
              <Badge variant="destructive">Destructive</Badge>
              <Badge variant="outline">Outline</Badge>
            </div>
          </CardContent>
        </Card>

        {/* Form Elements */}
        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Form Elements</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Input
                  </label>
                  <Input placeholder="Enter text..." aria-label="Example input" />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    Input (disabled)
                  </label>
                  <Input placeholder="Disabled input" disabled aria-label="Disabled input" />
                </div>
              </div>
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <Switch
                    checked={switchOn}
                    onCheckedChange={setSwitchOn}
                    aria-label="Example switch"
                  />
                  <span className="text-sm">Switch</span>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    checked={checkboxOn}
                    onCheckedChange={(v) => setCheckboxOn(v === true)}
                    aria-label="Example checkbox"
                  />
                  <span className="text-sm">Checkbox</span>
                </div>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Progress
                </label>
                <Progress value={64} className="h-2" aria-label="Progress: 64%" />
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Cards */}
        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Cards</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-3">
              <div className="rounded-md border border-border p-4 shadow-sm">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Shadow SM</div>
                <div className="mt-1 text-sm">Subtle elevation for cards at rest.</div>
              </div>
              <div className="rounded-md border border-border p-4 shadow-md">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Shadow MD</div>
                <div className="mt-1 text-sm">Hover states, dropdowns.</div>
              </div>
              <div className="rounded-md border border-border p-4 shadow-lg">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Shadow LG</div>
                <div className="mt-1 text-sm">Modals, popovers.</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* ─── 5. Spacing & Radius ──────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="05" title="Spacing & Radius" />

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Spacing Scale</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {spacingScale.map((s) => (
                <div key={s.token} className="flex items-center gap-3">
                  <span className="w-16 shrink-0 font-mono text-[10px] text-muted-foreground">{s.token}</span>
                  <div className={cn('flex', s.class)}>
                    <div
                      className="h-6 rounded-sm"
                      style={{ backgroundColor: 'rgb(var(--primary))', width: s.px, opacity: 0.7 }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground">{s.px}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Border Radius</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-4">
              {radiusScale.map((r) => (
                <div key={r.token} className="flex flex-col items-center gap-1.5">
                  <div
                    className="h-12 w-12 border border-border"
                    style={{
                      borderRadius: r.px,
                      backgroundColor: 'rgb(var(--primary))',
                      opacity: 0.15,
                    }}
                  />
                  <div className="text-center">
                    <div className="font-mono text-[10px] text-muted-foreground">{r.token}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{r.px}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Shadow Depth</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-end gap-6">
              {[
                { token: 'shadow-sm', label: 'SM' },
                { token: 'shadow-md', label: 'MD' },
                { token: 'shadow-lg', label: 'LG' },
                { token: 'shadow-xl', label: 'XL' },
              ].map((s) => (
                <div key={s.token} className="flex flex-col items-center gap-2">
                  <div
                    className={cn('h-16 w-16 rounded-lg border border-border', s.token)}
                    style={{ backgroundColor: 'rgb(var(--card))' }}
                  />
                  <div className="text-center">
                    <div className="text-[10px] font-medium">{s.label}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{s.token}</div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* ─── 6. Interactive States ────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="06" title="Interactive States" />

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Button States</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap items-center gap-3">
              <Button size="sm">Default</Button>
              <Button size="sm" className="bg-primary/90">Hover</Button>
              <Button size="sm" className="ring-2 ring-ring ring-offset-2">Focus</Button>
              <Button size="sm" className="active:scale-[0.98]">Active</Button>
              <Button size="sm" disabled className="opacity-40 pointer-events-none">Disabled</Button>
            </div>
            <p className="mt-3 text-xs text-muted-foreground">
              Focus: <code className="font-mono text-[10px] bg-muted px-1 rounded">ring-2 ring-ring ring-offset-2</code>.
              Active: <code className="font-mono text-[10px] bg-muted px-1 rounded">scale-[0.98]</code> press feedback.
              Disabled: <code className="font-mono text-[10px] bg-muted px-1 rounded">opacity-40 pointer-events-none</code>.
            </p>
          </CardContent>
        </Card>

        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Hover Preview</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2">
              <button
                type="button"
                className={cn(
                  'rounded-md border p-4 text-left text-sm transition-all duration-200',
                  'hover:border-primary/50 hover:shadow-md hover:-translate-y-0.5',
                  'focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                  interactiveHover ? 'border-primary/50 shadow-md -translate-y-0.5' : 'border-border'
                )}
                onMouseEnter={() => setInteractiveHover(true)}
                onMouseLeave={() => setInteractiveHover(false)}
                aria-label="Hoverable card demo"
              >
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Hover Me</div>
                <div className="mt-1">Hover card with elevation change.</div>
              </button>
              <div className="rounded-md border border-border p-4">
                <div className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Focus Ring</div>
                <div className="mt-2">
                  <Input
                    placeholder="Tab here to see focus ring..."
                    aria-label="Focus ring demo"
                    className="focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                  />
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* ─── 7. Dark Mode Preview ─────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="07" title="Dark Mode Preview" />

        <div
          className="magazine-card rounded-lg border p-6"
          style={{
            backgroundColor: 'rgb(17, 15, 24)',
            borderColor: 'rgb(52, 46, 72)',
          }}
        >
          <div className="space-y-4">
            <div className="flex items-baseline justify-between">
              <h3 className="text-base font-medium" style={{ color: 'rgb(238, 234, 248)' }}>
                Dark Mode Palette
              </h3>
              <span className="font-mono text-[10px]" style={{ color: 'rgb(150, 140, 172)' }}>
                html.dark
              </span>
            </div>
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
              {[
                { label: 'Background', rgb: '17 15 24' },
                { label: 'Card', rgb: '28 25 38' },
                { label: 'Primary', rgb: '192 170 244' },
                { label: 'Accent', rgb: '240 176 130' },
                { label: 'Success', rgb: '72 192 140' },
                { label: 'Border', rgb: '52 46 72' },
              ].map((c) => (
                <Swatch key={c.label} rgb={c.rgb} label={c.label} size="sm" />
              ))}
            </div>
            {/* Sample dark UI elements */}
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                className="rounded-md px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  backgroundColor: 'rgb(192, 170, 244)',
                  color: 'rgb(25, 22, 36)',
                }}
              >
                Primary Button
              </button>
              <button
                type="button"
                className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors"
                style={{
                  borderColor: 'rgb(52, 46, 72)',
                  color: 'rgb(238, 234, 248)',
                }}
              >
                Outline Button
              </button>
              <span
                className="rounded-md px-2 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: 'rgb(72, 192, 140)',
                  color: 'rgb(25, 22, 36)',
                }}
              >
                Success Badge
              </span>
              <span
                className="rounded-md px-2 py-0.5 text-[10px] font-medium"
                style={{
                  backgroundColor: 'rgb(240, 176, 130)',
                  color: 'rgb(25, 22, 36)',
                }}
              >
                Accent Badge
              </span>
            </div>
            <p className="text-xs" style={{ color: 'rgb(150, 140, 172)' }}>
              Body text in muted foreground. Warm, deep charcoal-violet with vibrant lilac accents.
            </p>
          </div>
        </div>
      </div>

      <Separator />

      {/* ─── 8. Accent Themes ─────────────────────────────────── */}
      <div className="space-y-4">
        <SectionHeader number="08" title="Accent Themes" />

        <Card className="magazine-card">
          <CardContent className="pt-4">
            <p className="mb-4 text-xs text-muted-foreground">
              Seven accent themes override <code className="font-mono text-[10px] bg-muted px-1 rounded">--primary</code> and{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">--ring</code> via CSS class on{' '}
              <code className="font-mono text-[10px] bg-muted px-1 rounded">&lt;html&gt;</code>.
            </p>
            <div className="grid grid-cols-4 gap-4 sm:grid-cols-7">
              {accentThemes.map((t) => (
                <Swatch key={t.name} rgb={t.rgb} label={t.label} size="lg" />
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Theme preview row */}
        <Card className="magazine-card">
          <CardHeader>
            <CardTitle className="text-base">Theme in Context</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid gap-3 sm:grid-cols-2">
              {accentThemes.map((t) => (
                <div
                  key={t.name}
                  className="flex items-center gap-3 rounded-md border border-border p-3"
                >
                  <div
                    className="h-8 w-8 shrink-0 rounded-md shadow-sm"
                    style={{ backgroundColor: `rgb(${t.rgb})` }}
                  />
                  <div className="min-w-0">
                    <div className="text-xs font-medium">{t.label}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">
                      html.theme-{t.name}
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="ml-auto shrink-0"
                    style={{
                      borderColor: `rgb(${t.rgb})`,
                      color: `rgb(${t.rgb})`,
                    }}
                  >
                    Apply
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <Separator />

      {/* ─── Footer ───────────────────────────────────────────── */}
      <div className="flex items-center justify-between pb-4">
        <div className="text-xs text-muted-foreground">
          sloughGPT &middot; Noir Violet Design System
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Locked</Badge>
          <span className="font-mono text-[10px] text-muted-foreground">v1.0</span>
        </div>
      </div>
    </PageContainer>
  )
}
