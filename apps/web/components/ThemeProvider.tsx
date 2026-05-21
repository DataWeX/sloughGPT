'use client'

import { createContext, useContext, useState, useEffect, useLayoutEffect, ReactNode } from 'react'

import { syncHtmlTheme } from '@/lib/sync-html-theme'
import {
  isStoredThemeId,
  MODE_STORAGE_KEY,
  THEME_STORAGE_KEY,
  type StoredThemeId,
  type ThemeMode,
} from '@/lib/theme-storage'

interface ThemeContextType {
  theme: StoredThemeId
  mode: ThemeMode
  setTheme: (theme: StoredThemeId) => void
  setMode: (mode: ThemeMode) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setTheme] = useState<StoredThemeId>('purple')
  const [mode, setMode] = useState<ThemeMode>('dark')
  const [mounted, setMounted] = useState(false)

  useLayoutEffect(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY)
    const savedMode = localStorage.getItem(MODE_STORAGE_KEY) as ThemeMode
    const t = isStoredThemeId(savedTheme) ? savedTheme : 'purple'
    const m = savedMode === 'light' || savedMode === 'dark' ? savedMode : 'dark'
    setTheme(t)
    setMode(m)
    syncHtmlTheme(m, t)
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return
    syncHtmlTheme(mode, theme)
    localStorage.setItem(THEME_STORAGE_KEY, theme)
    localStorage.setItem(MODE_STORAGE_KEY, mode)
  }, [theme, mode, mounted])

  return (
    <ThemeContext.Provider value={{ theme, mode, setTheme, setMode }}>
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
  { id: 'blue', name: 'Periwinkle', color: '#8b7bc4' },
  { id: 'purple', name: 'Lilac', color: '#a67fd4' },
  { id: 'pink', name: 'Rose', color: '#d894b4' },
  { id: 'red', name: 'Coral', color: '#e88890' },
  { id: 'orange', name: 'Peach', color: '#e8a86c' },
  { id: 'green', name: 'Mint', color: '#6bb89a' },
  { id: 'teal', name: 'Dew', color: '#6cabcc' },
]

/** Maps theme ID to CSS variable value for --primary */
export const THEME_PRIMARY: Record<StoredThemeId, string> = {
  blue: '139 123 196',
  purple: '166 127 212',
  pink: '216 148 180',
  red: '232 136 144',
  orange: '232 168 108',
  green: '107 184 154',
  teal: '108 171 204',
}
