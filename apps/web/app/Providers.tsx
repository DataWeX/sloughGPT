'use client'

import { SessionProvider } from 'next-auth/react'
import { ThemeProvider } from '@/components/ThemeProvider'
import { BootstrapProvider } from '@/contexts/BootstrapContext'
import { ModelProvider } from '@/contexts/ModelContext'
import { LocaleProvider } from '@/hooks/useLocale'
export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider refetchOnWindowFocus={false}>
      <ThemeProvider>
        <BootstrapProvider>
          <ModelProvider>
            <LocaleProvider>{children}</LocaleProvider>
          </ModelProvider>
        </BootstrapProvider>
      </ThemeProvider>
    </SessionProvider>
  )
}
