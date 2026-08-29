'use client'

import { IconMoon, IconSun } from '@/components/icons/NavIcons'

import { cn, Button } from '@sloughgpt/strui'
import { useTheme, THEMES } from './ThemeProvider'
import { PALETTE_IDS, PALETTE_LABELS, PALETTE_COLORS } from '@/lib/theme-storage'

export function ThemeSwitcher() {
  const { theme, mode, palette, setTheme, setMode, setPalette } = useTheme()

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="flex items-center justify-center gap-3" role="group" aria-label="Theme settings">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setMode(mode === 'dark' ? 'light' : 'dark')}
          aria-label={mode === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {mode === 'dark' ? <IconMoon aria-hidden="true" /> : <IconSun aria-hidden="true" />}
        </Button>

        <span className="text-xs text-muted-foreground/50 font-medium">or</span>

        <div className="flex items-center gap-1" role="radiogroup" aria-label="Color theme">
          {THEMES.map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setTheme(t.id)}
              className={cn('h-3.5 w-3.5 rounded-none transition-all duration-200 ease-smooth', theme === t.id
                  ? 'ring-2 ring-primary ring-offset-2 ring-offset-card scale-110 shadow-sm'
                  : 'opacity-80 hover:scale-105 hover:opacity-100')}
              style={{ backgroundColor: t.color }}
              role="radio"
              aria-checked={theme === t.id}
              aria-label={t.name}
              title={t.name}
            />
          ))}
        </div>
      </div>

      <div className="flex items-center justify-center gap-2" role="radiogroup" aria-label="Palette">
        {PALETTE_IDS.map((id) => (
          <button
            key={id}
            type="button"
            onClick={() => setPalette(id)}
            className={cn('h-3.5 w-3.5 rounded-full transition-all duration-200 ease-smooth', palette === id
                ? 'ring-2 ring-primary ring-offset-2 ring-offset-card scale-110 shadow-sm'
                : 'opacity-80 hover:scale-105 hover:opacity-100')}
            style={{ backgroundColor: PALETTE_COLORS[id] }}
            role="radio"
            aria-checked={palette === id}
            aria-label={PALETTE_LABELS[id]}
            title={PALETTE_LABELS[id]}
          />
        ))}
      </div>
    </div>
  )
}
