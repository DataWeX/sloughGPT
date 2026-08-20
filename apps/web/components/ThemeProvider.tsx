'use client'

import { createContext, useContext, useState, useEffect, useLayoutEffect, ReactNode } from 'react'
import { syncHtmlTheme } from '@/lib/sync-html-theme'
import { chatDB } from '@/lib/db'
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

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<StoredThemeId>('purple')
  const [mode, setMode] = useState<ThemeMode>('dark')
  const [palette, setPalette] = useState<StoredPaletteId>('noir-violet')
  const [mounted, setMounted] = useState(false)

  useLayoutEffect(() => {
    Promise.all([
      chatDB.getKV<string>(THEME_STORAGE_KEY),
      chatDB.getKV<string>(MODE_STORAGE_KEY),
      chatDB.getKV<string>(PALETTE_STORAGE_KEY),
    ]).then(([savedTheme, savedMode, savedPalette]) => {
      const t = isStoredThemeId(savedTheme) ? savedTheme : 'purple'
      const p = isStoredPaletteId(savedPalette) ? savedPalette : 'noir-violet'
      let m: ThemeMode
      if (savedMode === 'light' || savedMode === 'dark') {
        m = savedMode
      } else if (typeof window.matchMedia === 'function' && window.matchMedia('(prefers-color-scheme: light)').matches) {
        m = 'light'
      } else {
        m = 'dark'
      }
      setTheme(t)
      setMode(m)
      setPalette(p)
      syncHtmlTheme(m, t, p)
      setMounted(true)
    })
  }, [])

  useEffect(() => {
    if (!mounted) return
    syncHtmlTheme(mode, theme, palette)
    chatDB.setKV(THEME_STORAGE_KEY, theme).catch(() => {})
    chatDB.setKV(MODE_STORAGE_KEY, mode).catch(() => {})
    chatDB.setKV(PALETTE_STORAGE_KEY, palette).catch(() => {})
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
