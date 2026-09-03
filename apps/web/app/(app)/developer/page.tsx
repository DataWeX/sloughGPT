'use client'
export const dynamic = 'force-dynamic'

import { useState } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardHeader, CardTitle, Button, cn } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import Link from 'next/link'

type DevTab = 'files' | 'voice' | 'shell'

const TABS: { id: DevTab; label: string; href: string; description: string }[] = [
  { id: 'files', label: 'Files', href: '/files', description: 'Upload, manage & ingest documents' },
  { id: 'voice', label: 'Voice', href: '/voice', description: 'Text-to-speech via browser or AI model' },
  { id: 'shell', label: 'Shell', href: '/shell', description: 'Backend terminal & browser VM' },
]

export default function DeveloperPage() {
  const [tab, setTab] = useState<DevTab>('files')
  const active = TABS.find(t => t.id === tab)!

  return (
    <PageContainer title="Developer" subtitle="Files, voice & shell tools">
      <div className="flex gap-1 border-b border-border/30 mb-4" role="tablist" aria-label="Developer tools">
        {TABS.map(t => (
          <button
            type="button"
            role="tab"
            key={t.id}
            aria-selected={tab === t.id}
            aria-label={`${t.label} tab`}
            onClick={() => setTab(t.id)}
            className={cn(
              'px-3 py-1.5 text-xs font-medium rounded-t transition-colors',
              tab === t.id ? 'bg-primary/10 text-primary border-b-2 border-primary' : 'text-muted-foreground hover:text-foreground',
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">{active.label}</CardTitle>
          <Button asChild size="sm" variant="ghost">
            <Link href={active.href} aria-label={`Open ${active.label} page`}>
              Open full page
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">{active.description}</p>
        </CardContent>
      </Card>
    </PageContainer>
  )
}
