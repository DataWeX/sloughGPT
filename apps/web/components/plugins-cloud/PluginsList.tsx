'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle, Badge, Skeleton } from '@sloughgpt/strui'

export interface PluginInfo {
  name: string
  version: string
  description: string
  author: string
  enabled: boolean
}

interface PluginsListProps {
  plugins?: PluginInfo[]
  loading?: boolean
}

export function PluginsList({ plugins = [], loading = false }: PluginsListProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Installed Plugins</CardTitle>
        <CardDescription>Manage loaded plugins</CardDescription>
      </CardHeader>
      <CardContent>
        {loading ? (
          <Skeleton className="h-20 w-full" />
        ) : plugins.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            No plugins installed. Add plugins to ~/.config/sloughgpt/plugins/
          </p>
        ) : (
          <div className="space-y-2">
            {plugins.map((plugin) => (
              <div
                key={plugin.name}
                className="flex items-center justify-between p-2 border border-border/50 rounded"
              >
                <div>
                  <span className="font-medium text-sm">{plugin.name}</span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    v{plugin.version}
                  </span>
                  <span className="ml-2 text-xs text-muted-foreground">
                    by {plugin.author}
                  </span>
                </div>
                <Badge variant={plugin.enabled ? 'default' : 'secondary'}>
                  {plugin.enabled ? 'Enabled' : 'Disabled'}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
