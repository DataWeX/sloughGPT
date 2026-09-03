'use client'

import { createContext, useContext, useState, useEffect, useLayoutEffect, ReactNode } from 'react'
import { syncHtmlTheme } from '@/lib/sync-html-theme'
import { trackEvent } from '@/lib/dev-log'
import {
  isStoredThemeId,
  isStoredPaletteId,
  MODE_STORAGE_KEY,
  THEME_STORAGE_KEY,
  PALETTE_STORAGE_KEY,
  type StoredThemeId,
  type ThemeMode,
  type StoredPaletteId,
} from '@/lib/theme-storage'

interface ThemeContextType {
  theme: StoredThemeId
  mode: ThemeMode
  palette: StoredPaletteId
  setTheme: (theme: StoredThemeId) => void
  setMode: (mode: ThemeMode) => void
  setPalette: (palette: StoredPaletteId) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

function getInitialTheme(): StoredThemeId {
  if (typeof window === 'undefined') return 'purple'
  const v = localStorage.getItem(THEME_STORAGE_KEY)
  return isStoredThemeId(v) ? v : 'purple'
}

function getInitialMode(): ThemeMode {
  if (typeof window === 'undefined') return 'dark'
  const v = localStorage.getItem(MODE_STORAGE_KEY)
  if (v === 'light' || v === 'dark') return v
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

function getInitialPalette(): StoredPaletteId {
  if (typeof window === 'undefined') return 'noir-violet'
  const v = localStorage.getItem(PALETTE_STORAGE_KEY)
  return isStoredPaletteId(v) ? v : 'noir-violet'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, _setTheme] = useState<StoredThemeId>(getInitialTheme)
  const [mode, _setMode] = useState<ThemeMode>(getInitialMode)
  const [palette, _setPalette] = useState<StoredPaletteId>(getInitialPalette)
  const [mounted, setMounted] = useState(false)

  const setTheme = (next: StoredThemeId) => {
    trackEvent('theme_changed', { from: theme, to: next })
    _setTheme(next)
  }

  const setMode = (next: ThemeMode) => {
    trackEvent('mode_changed', { from: mode, to: next })
    _setMode(next)
  }

  const setPalette = (next: StoredPaletteId) => {
    trackEvent('palette_changed', { from: palette, to: next })
    _setPalette(next)
  }

  useLayoutEffect(() => {
    // Theme is already loaded from localStorage synchronously.
    // Sync to HTML element and mark as mounted.
    syncHtmlTheme(mode, theme, palette)
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return
    syncHtmlTheme(mode, theme, palette)
    localStorage.setItem(THEME_STORAGE_KEY, theme)
    localStorage.setItem(MODE_STORAGE_KEY, mode)
    localStorage.setItem(PALETTE_STORAGE_KEY, palette)
  }, [theme, mode, palette, mounted])

  return (
    <ThemeContext.Provider value={{ theme, mode, palette, setTheme, setMode, setPalette }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

/** Accent presets — ids kept for localStorage; hues match ``globals.css`` theme-* */
export const THEMES: { id: StoredThemeId; name: string; color: string }[] = [
  { id: 'blue', name: 'Periwinkle', color: '#5a82dc' },
  { id: 'purple', name: 'Lilac', color: '#9b6cd6' },
  { id: 'pink', name: 'Rose', color: '#da82aa' },
  { id: 'red', name: 'Coral', color: '#e67882' },
  { id: 'orange', name: 'Peach', color: '#ec9b5a' },
  { id: 'green', name: 'Mint', color: '#48b282' },
  { id: 'teal', name: 'Dew', color: '#48a6c8' },
]
