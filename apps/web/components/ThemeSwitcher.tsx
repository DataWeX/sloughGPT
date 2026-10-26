'use client'

import type { KeyboardEvent } from 'react'

import { IconMoon, IconSun } from '@/components/icons/NavIcons'

import { cn } from '@sloughgpt/strui'
import { useTheme, THEMES } from './ThemeProvider'
import { PALETTE_IDS, PALETTE_LABELS, PALETTE_COLORS } from '@/lib/theme-storage'

/**
 * Theme switcher — the sidebar footer compartment.
 *
 * Composition, not monolith: one component reads `useTheme` (the single
 * source of truth) and hands props down to three dumb, controlled
 * sub-components. No local copy of truth anywhere — selection state lives
 * in the provider, this file only renders it and emits events.
 *
 * Every radiogroup follows the ARIA radiogroup pattern: roving arrow-key
 * focus, selection follows focus, checked state gated on `mounted` so the
 * server markup stays mode-neutral (hydration invariant — see the test).
 */

/** Roving focus + selection-follows-focus for a radiogroup. */
function radioKeyDown(
  e: KeyboardEvent<HTMLElement>,
  index: number,
  count: number,
  select: (i: number) => void,
) {
  let next: number | null = null
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown') next = (index + 1) % count
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') next = (index - 1 + count) % count
  else if (e.key === 'Home') next = 0
  else if (e.key === 'End') next = count - 1
  if (next === null) return
  e.preventDefault()
  select(next)
  // currentTarget is the radiogroup itself — rove within it, not above it.
  const radios = e.currentTarget.querySelectorAll<HTMLElement>('[role="radio"]')
  radios?.[next]?.focus()
}

/** Checked-state ring shared by swatches (tokens only). */
const CHECKED_RING = 'ring-2 ring-primary ring-offset-2 ring-offset-card'
const FOCUS_RING =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card'
/** All selection/label text transitions colour smoothly (the "transitions thereof"). */
const TRANSITION = 'transition-all duration-200 ease-smooth'

export function ThemeSwitcher({ collapsed = false }: { collapsed?: boolean }) {
  const { theme, mode, palette, mounted, setTheme, setMode, setPalette } = useTheme()
  const current = THEMES.find((t) => t.id === theme) ?? THEMES[1]

  // Collapsed icon rail: the full composition cannot fit — show the compact
  // mode toggle only (the old footer behaviour), everything else on expand.
  if (collapsed) {
    // Label/icon gate on `mounted` exactly like the pre-v2 toggle: the
    // server string never depends on stored mode (hydration invariant).
    const target = mounted && mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode'
    return (
      <div
        role="group"
        aria-label="Theme settings"
        data-testid="theme-switcher"
        className="flex justify-center"
      >
        <button
          type="button"
          onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}
          aria-label={target}
          title={target}
          className={cn(
            'flex h-9 w-9 cursor-pointer items-center justify-center rounded-full border border-border/70 bg-muted/60 text-muted-foreground hover:text-foreground',
            TRANSITION,
            FOCUS_RING,
          )}
        >
          {mounted && mode === 'light' ? (
            <IconSun aria-hidden="true" className="h-4 w-4" />
          ) : (
            <IconMoon aria-hidden="true" className="h-4 w-4" />
          )}
        </button>
      </div>
    )
  }

  return (
    <div
      role="group"
      aria-label="Theme settings"
      data-testid="theme-switcher"
      className="flex w-full flex-col gap-2.5"
    >
      {/* Caption — what is currently selected (mounted-gated: pre-mount the
          server string must match, so it renders empty until hydration). */}
      <div className="flex items-baseline justify-between gap-2 px-0.5">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          Theme
        </span>
        <span
          data-testid="theme-caption"
          className="truncate text-[11px] font-medium text-foreground/80"
        >
          {mounted ? `${current.name} · ${PALETTE_LABELS[palette]}` : ''}
        </span>
      </div>

      <ModeSegmented mode={mode} mounted={mounted} onChange={setMode} />
      <ColorSwatches theme={theme} mounted={mounted} onChange={setTheme} />
      <PalettePicker palette={palette} mounted={mounted} onChange={setPalette} />
    </div>
  )
}

/** Segmented Dark | Light control with a sliding pill. */
function ModeSegmented({
  mode,
  mounted,
  onChange,
}: {
  mode: 'dark' | 'light'
  mounted: boolean
  onChange: (m: 'dark' | 'light') => void
}) {
  const modes = ['dark', 'light'] as const
  return (
    <div
      role="radiogroup"
      aria-label="Mode"
      className="relative flex rounded-full border border-border/70 bg-muted/60 p-1"
      onKeyDown={(e) =>
        radioKeyDown(e, modes.indexOf(mode), modes.length, (i) => onChange(modes[i]))
      }
    >
      {/* Sliding pill — exactly half the padding box per segment. */}
      <span
        aria-hidden
        className={cn(
          'pointer-events-none absolute bottom-1 left-1 top-1 w-[calc(50%-0.25rem)] rounded-full bg-card shadow-sm',
          TRANSITION,
          mounted && mode === 'light' && 'translate-x-full',
        )}
      />
      {modes.map((m) => (
        <button
          key={m}
          type="button"
          role="radio"
          aria-checked={mounted && mode === m}
          aria-label={`Switch to ${m} mode`}
          onClick={() => onChange(m)}
          className={cn(
            'z-10 flex flex-1 cursor-pointer items-center justify-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium',
            TRANSITION,
            FOCUS_RING,
            mounted && mode === m
              ? 'text-foreground'
              : 'text-muted-foreground hover:text-foreground',
          )}
        >
          {m === 'dark' ? (
            <IconMoon aria-hidden="true" className="h-3.5 w-3.5" />
          ) : (
            <IconSun aria-hidden="true" className="h-3.5 w-3.5" />
          )}
          {m === 'dark' ? 'Dark' : 'Light'}
        </button>
      ))}
    </div>
  )
}

/** Seven colour swatches as a radiogroup — ring + scale on the checked one. */
function ColorSwatches({
  theme,
  mounted,
  onChange,
}: {
  theme: string
  mounted: boolean
  onChange: (id: (typeof THEMES)[number]['id']) => void
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Color theme"
      className="flex flex-wrap items-center justify-center gap-2"
      onKeyDown={(e) =>
        radioKeyDown(
          e,
          THEMES.findIndex((t) => t.id === theme),
          THEMES.length,
          (i) => onChange(THEMES[i].id),
        )
      }
    >
      {THEMES.map((t) => (
        <button
          key={t.id}
          type="button"
          role="radio"
          aria-checked={mounted && theme === t.id}
          onClick={() => onChange(t.id)}
          title={`${t.name} — ${t.aura}`}
          style={{ backgroundColor: t.color }}
          className={cn(
            'h-4 w-4 cursor-pointer rounded-full border border-border/40',
            TRANSITION,
            FOCUS_RING,
            mounted && theme === t.id
              ? cn(CHECKED_RING, 'scale-110')
              : 'opacity-80 hover:scale-110 hover:opacity-100',
          )}
        >
          <span className="sr-only">{t.name}</span>
        </button>
      ))}
    </div>
  )
}

/** Palette chips — swatch dot + label (the caption lives in the chip). */
function PalettePicker({
  palette,
  mounted,
  onChange,
}: {
  palette: string
  mounted: boolean
  onChange: (id: (typeof PALETTE_IDS)[number]) => void
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Palette"
      className="flex items-center justify-center gap-2"
      onKeyDown={(e) =>
        radioKeyDown(
          e,
          PALETTE_IDS.indexOf(palette as (typeof PALETTE_IDS)[number]),
          PALETTE_IDS.length,
          (i) => onChange(PALETTE_IDS[i]),
        )
      }
    >
      {PALETTE_IDS.map((id) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={mounted && palette === id}
          onClick={() => onChange(id)}
          title={PALETTE_LABELS[id]}
          className={cn(
            'flex cursor-pointer items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium',
            TRANSITION,
            FOCUS_RING,
            mounted && palette === id
              ? 'border-primary/70 bg-primary/10 text-foreground ring-1 ring-primary/40'
              : 'border-border/70 text-muted-foreground hover:border-border hover:bg-muted/60 hover:text-foreground',
          )}
        >
          <span
            aria-hidden
            className="h-2.5 w-2.5 rounded-full border border-border/40"
            style={{ backgroundColor: PALETTE_COLORS[id] }}
          />
          {PALETTE_LABELS[id]}
        </button>
      ))}
    </div>
  )
}
