'use client'

import { useLocale, LOCALES } from '@/hooks/useLocale'
import { CustomDropdown } from './CustomDropdown'

export function LanguageSwitcher() {
  const { locale, setLocale } = useLocale()

  const currentLang = LOCALES.find(l => l.code === locale) || LOCALES[0]

  return (
    <CustomDropdown
      grid
      trigger={
        <button className="flex items-center justify-center gap-1 w-full px-2 py-1.5 hover:bg-accent/50 rounded-md transition-colors cursor-pointer text-xs">
          <span>{currentLang.flag}</span>
          <span className="uppercase">{currentLang.code}</span>
        </button>
      }
      items={LOCALES.map((l) => ({
        label: `${l.flag} ${l.code.toUpperCase()}`,
        onClick: () => setLocale(l.code as typeof locale),
      }))}
    />
  )
}