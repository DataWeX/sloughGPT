'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid, Skeleton } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { PageContainer } from '@/components/PageContainer'
import { companionController, type CompanionTraits, type CompanionPreset } from '@/lib/companion-controller'
import { CompanionInsightsCard } from '@/components/companion/CompanionInsightsCard'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'

const TRAIT_LABELS: Record<string, { label: string; color: string }> = {
  warmth: { label: 'Warmth', color: 'bg-orange-500/15 text-orange-600 dark:text-orange-400' },
  curiosity: { label: 'Curiosity', color: 'bg-blue-500/15 text-blue-600 dark:text-blue-400' },
  creativity: { label: 'Creativity', color: 'bg-purple-500/15 text-purple-600 dark:text-purple-400' },
  confidence: { label: 'Confidence', color: 'bg-green-500/15 text-green-600 dark:text-green-400' },
  humor: { label: 'Humor', color: 'bg-pink-500/15 text-pink-600 dark:text-pink-400' },
}

export default function CompanionPage() {
  const [traits, setTraits] = useState<CompanionTraits | null>(null)
  const [presets, setPresets] = useState<CompanionPreset[]>([])
  const [systemPrompt, setSystemPrompt] = useState('')
  const [chatInput, setChatInput] = useState('')
  const [chatMessages, setChatMessages] = useState<Array<{ role: 'user' | 'assistant'; content: string }>>([])
  const [chatLoading, setChatLoading] = useState(false)
  const [saving, setSaving] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const addToast = useToastStore(s => s.addToast)

  useEffect(() => {
    let ignore = false
    Promise.all([
      companionController.getInfo(),
      companionController.listPresets(),
      companionController.getPrompt(),
    ]).then(([info, presetRes, promptRes]) => {
      if (ignore) return
      setTraits(info.traits)
      setPresets(presetRes.presets)
      setSystemPrompt(promptRes.system_prompt)
    }).catch((err) => {
      if (!ignore) setError(extractErrorMessage(err, 'Could not load companion data'))
    }).finally(() => { if (!ignore) setLoading(false) })
    return () => { ignore = true }
  }, [])

  const handleTraitChange = (key: string, value: number) => {
    if (!traits) return
    setTraits({ ...traits, [key]: value })
  }

  const handleSave = async () => {
    if (!traits) return
    setSaving(true)
    try {
      const res = await companionController.setPersonality(traits)
      setTraits(res.traits)
      const promptRes = await companionController.getPrompt()
      setSystemPrompt(promptRes.system_prompt)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not save personality'), 'error')
    } finally {
      setSaving(false)
    }
  }

  const handlePreset = async (presetId: string) => {
    try {
      const res = await companionController.setPreset(presetId)
      setTraits(res.traits)
      const promptRes = await companionController.getPrompt()
      setSystemPrompt(promptRes.system_prompt)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not apply preset'), 'error')
    }
  }

  const handleReset = async () => {
    try {
      const res = await companionController.reset()
      setTraits(res.traits)
      const promptRes = await companionController.getPrompt()
      setSystemPrompt(promptRes.system_prompt)
    } catch (err) {
      addToast(extractErrorMessage(err, 'Could not reset companion'), 'error')
    }
  }

  const handleChat = async () => {
    if (!chatInput.trim() || chatLoading) return
    const userMsg = chatInput.trim()
    setChatMessages(prev => [...prev, { role: 'user', content: userMsg }])
    setChatLoading(true)
    setChatInput('')
    try {
      const res = await companionController.chat(userMsg)
      setChatMessages(prev => [...prev, { role: 'assistant', content: res.response }])
    } catch (err) {
      setChatMessages(prev => [...prev, { role: 'assistant', content: `[Error: ${extractErrorMessage(err, 'could not reach model')}]` }])
    } finally {
      setChatLoading(false)
    }
  }

  const handleClearChat = () => {
    setChatMessages([])
  }

  if (loading) {
    return (
      <PageContainer title="Companion" subtitle="AI personality management" loadingCards={3}>
        <KpiGrid>
          <StatCard label="Active Preset" value={<Skeleton className="h-5 w-16" />} />
          <StatCard label="Warmth" value={<Skeleton className="h-5 w-8" />} />
          <StatCard label="Curiosity" value={<Skeleton className="h-5 w-8" />} />
          <StatCard label="Creativity" value={<Skeleton className="h-5 w-8" />} />
        </KpiGrid>
        <Card><CardContent><div className="h-48 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
      </PageContainer>
    )
  }

  if (error) {
    return (
      <PageContainer title="Companion" subtitle="AI personality management" error={error} onRetry={() => window.location.reload()}>
        <></>
      </PageContainer>
    )
  }

  const avgTrait = traits ? Math.round(Object.values(traits).reduce((a, b) => a + b, 0) / Object.values(traits).length) : 0

  return (
    <PageContainer title="Companion" subtitle="AI personality management">
      <KpiGrid>
        <StatCard label="Active Preset" value={presets.length > 0 ? presets[0].name : 'Custom'} />
        <StatCard label="Warmth" value={traits ? String(traits.warmth ?? 0) : <Skeleton className="h-5 w-8" />} />
        <StatCard label="Curiosity" value={traits ? String(traits.curiosity ?? 0) : <Skeleton className="h-5 w-8" />} />
        <StatCard label="Avg Trait" value={String(avgTrait)} />
      </KpiGrid>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Presets</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {presets.map(p => (
              <button
                key={p.id}
                type="button"
                onClick={() => handlePreset(p.id)}
                className="rounded-md bg-muted px-3 py-1.5 text-xs font-medium text-muted-foreground hover:bg-muted/80 transition-colors"
                title={p.description}
              >
                {p.name}
              </button>
            ))}
          </div>
        </CardContent>
      </Card>

      <CompanionInsightsCard traits={traits} presets={presets} />

      {traits && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle className="text-base">Personality Traits</CardTitle>
            <div className="flex gap-1.5">
              <Button size="sm" variant="ghost" onClick={handleReset} aria-label="Reset">
                <IconRefresh className="h-4 w-4" />
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Radar chart */}
            {(() => {
              const traitKeys = Object.keys(TRAIT_LABELS)
              const n = traitKeys.length
              const cx = 100, cy = 100, r = 70
              const angleStep = (2 * Math.PI) / n
              const points = traitKeys.map((key, i) => {
                const val = (traits[key as keyof CompanionTraits] as number) ?? 0.5
                const angle = angleStep * i - Math.PI / 2
                return { x: cx + r * val * Math.cos(angle), y: cy + r * val * Math.sin(angle) }
              })
              const polygonPoints = points.map(p => `${p.x},${p.y}`).join(' ')
              return (
                <div className="flex justify-center">
                  <svg viewBox="0 0 200 200" className="w-48 h-48">
                    {/* Grid rings */}
                    {[0.25, 0.5, 0.75, 1].map(scale => (
                      <polygon
                        key={scale}
                        points={traitKeys.map((_, i) => {
                          const angle = angleStep * i - Math.PI / 2
                          return `${cx + r * scale * Math.cos(angle)},${cy + r * scale * Math.sin(angle)}`
                        }).join(' ')}
                        fill="none"
                        stroke="currentColor"
                        className="text-border/40"
                        strokeWidth="0.5"
                      />
                    ))}
                    {/* Axis lines */}
                    {traitKeys.map((_, i) => {
                      const angle = angleStep * i - Math.PI / 2
                      return (
                        <line
                          key={i}
                          x1={cx}
                          y1={cy}
                          x2={cx + r * Math.cos(angle)}
                          y2={cy + r * Math.sin(angle)}
                          stroke="currentColor"
                          className="text-border/30"
                          strokeWidth="0.5"
                        />
                      )
                    })}
                    {/* Trait polygon */}
                    <polygon
                      points={polygonPoints}
                      fill="rgb(var(--primary))"
                      fillOpacity="0.15"
                      stroke="rgb(var(--primary))"
                      strokeWidth="1.5"
                    />
                    {/* Data points + labels */}
                    {traitKeys.map((key, i) => {
                      const angle = angleStep * i - Math.PI / 2
                      const labelR = r + 18
                      return (
                        <g key={key}>
                          <circle cx={points[i].x} cy={points[i].y} r="3" fill="rgb(var(--primary))" />
                          <text
                            x={cx + labelR * Math.cos(angle)}
                            y={cy + labelR * Math.sin(angle)}
                            textAnchor="middle"
                            dominantBaseline="middle"
                            className="fill-muted-foreground"
                            fontSize="8"
                          >
                            {TRAIT_LABELS[key].label}
                          </text>
                        </g>
                      )
                    })}
                  </svg>
                </div>
              )
            })()}
            <div className="grid gap-3">
              {Object.entries(TRAIT_LABELS).map(([key, { label, color }]) => (
                <div key={key} className="flex items-center gap-3">
                  <span className={`text-xs font-medium px-1.5 py-0.5 rounded ${color} w-20 text-center`}>
                    {label}
                  </span>
                   <input
                     type="range"
                     min={0}
                     max={1}
                     step={0.05}
                value={(traits[key as keyof CompanionTraits] as number) ?? 0.5}
                onChange={e => handleTraitChange(key, parseFloat(e.target.value))}
                aria-label={`${label} trait level`}
                className="flex-1 h-1.5 bg-muted rounded-full appearance-none cursor-pointer accent-primary"
              />
              <span className="text-xs font-mono text-muted-foreground w-8 text-right">
                {((traits[key as keyof CompanionTraits] as number) ?? 0.5).toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
            <Button size="sm" onClick={handleSave} disabled={saving}>
              {saving ? 'Saving...' : 'Save Personality'}
            </Button>
          </CardContent>
        </Card>
      )}

      {systemPrompt && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">System Prompt</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs text-muted-foreground bg-muted/30 rounded-md p-3 whitespace-pre-wrap font-mono max-h-40 overflow-y-auto">
              {systemPrompt}
            </pre>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="text-base">Test Chat</CardTitle>
          {chatMessages.length > 0 && (
            <Button size="sm" variant="ghost" onClick={handleClearChat} aria-label="Clear chat history">
              Clear
            </Button>
          )}
        </CardHeader>
        <CardContent className="space-y-3">
          {chatMessages.length > 0 && (
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {chatMessages.map((msg, i) => (
                <div
                  key={i}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg px-3 py-2 text-sm ${
                      msg.role === 'user'
                        ? 'bg-primary/10 text-foreground'
                        : 'bg-muted/50 text-foreground'
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}
              {chatLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted/50 rounded-lg px-3 py-2 text-sm text-muted-foreground">
                    Thinking...
                  </div>
                </div>
              )}
            </div>
          )}
          {chatMessages.length === 0 && !chatLoading && (
            <p className="text-xs text-muted-foreground text-center py-4">Say something to your companion...</p>
          )}
          <div ref={chatEndRef} />
          <div className="flex gap-2">
            <Input
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleChat()}
              placeholder="Type a message..."
              disabled={chatLoading}
              aria-label="Chat message"
            />
            <Button size="sm" onClick={handleChat} disabled={chatLoading || !chatInput.trim()}>
              {chatLoading ? '...' : 'Send'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </PageContainer>
  )
}
