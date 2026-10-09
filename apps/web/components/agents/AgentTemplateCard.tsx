'use client'

import { Card, CardHeader, CardTitle, CardContent, Button, Badge, cn } from '@sloughgpt/strui'
import { Sparkles } from 'lucide-react'

interface AgentTemplate {
  name: string
  desc: string
  instructions: string
  tools: string[]
}

interface AgentTemplateCardProps {
  templates: AgentTemplate[]
  onSelect: (template: AgentTemplate) => void
}

export function AgentTemplateCard({ templates, onSelect }: AgentTemplateCardProps) {
  return (
    <Card data-testid="agent-template-card">
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Sparkles className="h-4 w-4" />
          Agent Templates
        </CardTitle>
      </CardHeader>
      <CardContent>
        {templates.length === 0 ? (
          <div className="text-sm text-muted-foreground">No templates available.</div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2">
            {templates.map(template => (
              <div
                key={template.name}
                className="flex flex-col gap-2 p-3 rounded border border-border hover:border-primary/30 transition-colors"
              >
                <div className="text-xs font-medium">{template.name}</div>
                <div className="text-[11px] text-muted-foreground line-clamp-2">{template.desc}</div>
                {template.tools.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {template.tools.map(tool => (
                      <Badge key={tool} variant="secondary" className="text-[9px]">{tool}</Badge>
                    ))}
                  </div>
                )}
                <Button
                  size="sm"
                  variant="outline"
                  className="mt-auto h-7 text-[10px]"
                  onClick={() => onSelect(template)}
                >
                  Select
                </Button>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
