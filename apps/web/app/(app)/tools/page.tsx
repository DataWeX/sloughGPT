'use client'

import { useEffect, useState, type ComponentType } from 'react'
import Link from 'next/link'
import { PageContainer } from '@/components/PageContainer'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import {
  IconDocument,
  IconChat,
  IconSparkle,
  IconBrain,
  IconChart,
  IconSearch,
} from '@/components/icons/NavIcons'
import { listTools, type ToolProfile } from '@/lib/tools-controller'

type IconComponent = ComponentType<{ className?: string }>

const ICONS: Record<string, IconComponent> = {
  document: IconDocument,
  chat: IconChat,
  sparkle: IconSparkle,
  brain: IconBrain,
  chart: IconChart,
  search: IconSearch,
}

export default function ToolsPage() {
  const [tools, setTools] = useState<ToolProfile[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    listTools().then((result) => {
      if (!active) return
      setTools(result)
      setLoading(false)
    })
    return () => {
      active = false
    }
  }, [])

  return (
    <PageContainer title="Tools">
      <p className="text-sm text-muted-foreground mb-4">
        Quick access to every everyday tool, straight from the engine.
      </p>

      {loading && <p className="text-sm text-muted-foreground">Loading tools...</p>}

      {!loading && tools.length === 0 && (
        <p className="text-sm text-muted-foreground">No tools yet — start the backend to see them.</p>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {tools.map((tool) => {
          const Icon = ICONS[tool.icon] ?? IconSparkle
          return (
            <Link key={tool.id} href={`/${tool.id}`} className="block">
              <Card className="h-full transition-all hover:border-primary">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <Icon className="h-4 w-4" />
                    <CardTitle className="text-sm">{tool.name}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription>{tool.description}</CardDescription>
                </CardContent>
              </Card>
            </Link>
          )
        })}
      </div>
    </PageContainer>
  )
}