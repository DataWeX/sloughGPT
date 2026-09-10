'use client'

import { useState } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, cn } from '@sloughgpt/strui'

interface Feature {
  id: string
  icon: string
  title: string
  description: string
  link?: string
  color: string
}

const FEATURES: Feature[] = [
  { id: 'chat', icon: '💬', title: 'Smart Chat', description: 'Conversational AI that remembers context and learns from you.', link: '/chat', color: 'bg-primary/10 text-primary' },
  { id: 'knowledge', icon: '🧠', title: 'Knowledge Base', description: 'Store facts, preferences, and information your AI will remember.', link: '/kb', color: 'bg-accent/10 text-accent' },
  { id: 'companion', icon: '🤝', title: 'Companion', description: 'Customize your AI personality — warmth, curiosity, humor, and more.', link: '/companion', color: 'bg-warning/10 text-warning' },
  { id: 'voice', icon: '🎙️', title: 'Voice', description: 'Speak to your AI and hear responses with natural voice synthesis.', link: '/voice', color: 'bg-success/10 text-success' },
  { id: 'images', icon: '🎨', title: 'Image Generation', description: 'Create images from text descriptions with multiple styles.', link: '/images', color: 'bg-destructive/10 text-destructive' },
  { id: 'files', icon: '📁', title: 'File Management', description: 'Upload, organize, and share documents with your AI.', link: '/files', color: 'bg-muted text-muted-foreground' },
  { id: 'analytics', icon: '📊', title: 'Analytics', description: 'Track your usage, conversation quality, and AI performance.', link: '/analytics', color: 'bg-primary/10 text-primary' },
  { id: 'training', icon: '🏋️', title: 'Auto-Train', description: 'The AI improves automatically from your feedback and conversations.', link: '/auto-train', color: 'bg-accent/10 text-accent' },
]

interface OnboardingFeatureCardProps {
  onNavigate?: (link: string) => void
}

export function OnboardingFeatureCard({ onNavigate }: OnboardingFeatureCardProps) {
  const [expanded, setExpanded] = useState<string | null>(null)

  return (
    <Card data-testid="onboarding-feature">
      <CardHeader>
        <CardTitle className="text-base">Explore Features</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {FEATURES.map(f => (
            <button
              key={f.id}
              className={cn(
                'text-left p-2.5 rounded-lg border transition-all',
                expanded === f.id ? 'border-primary/50 bg-primary/5' : 'border-border hover:border-primary/30'
              )}
              onClick={() => setExpanded(expanded === f.id ? null : f.id)}
              data-testid={`feature-${f.id}`}
            >
              <div className="text-lg mb-1">{f.icon}</div>
              <div className="text-xs font-medium">{f.title}</div>
              {expanded === f.id && (
                <div className="mt-1.5">
                  <div className="text-[10px] text-muted-foreground">{f.description}</div>
                  {f.link && (
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-[9px] mt-1 h-5"
                      onClick={(e) => { e.stopPropagation(); onNavigate?.(f.link!) }}
                    >
                      Open →
                    </Button>
                  )}
                </div>
              )}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
