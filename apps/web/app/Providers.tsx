'use client'

import { SessionProvider } from 'next-auth/react'
import { ThemeProvider } from '@/components/ThemeProvider'
import { ModelProvider } from '@/contexts/ModelContext'
import { LocaleProvider } from '@/hooks/useLocale'
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider refetchOnWindowFocus={false}>
      <ThemeProvider>
        <ModelProvider>
          <LocaleProvider>
            {children}
          </LocaleProvider>
        </ModelProvider>
      </ThemeProvider>
    </SessionProvider>
  )
}
