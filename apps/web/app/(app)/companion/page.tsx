'use client'

import { useState, useEffect, useRef } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Button, Input, StatCard, KpiGrid } from '@sloughgpt/strui'
import { IconRefresh } from '@sloughgpt/strui'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { companionController, type CompanionTraits, type CompanionPreset } from '@/lib/companion-controller'
import { CompanionInsightsCard } from '@/components/companion/CompanionInsightsCard'
import { useToastStore } from '@/lib/toast-store'

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
  const [chatResponse, setChatResponse] = useState('')
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
    }).catch(() => {
      if (!ignore) setError('Failed to load companion data')
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
    } catch {
      addToast('Failed to save personality', 'error')
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
    } catch {
      addToast('Failed to apply preset', 'error')
    }
  }

  const handleReset = async () => {
    try {
      const res = await companionController.reset()
      setTraits(res.traits)
      const promptRes = await companionController.getPrompt()
      setSystemPrompt(promptRes.system_prompt)
    } catch {
      addToast('Failed to reset companion', 'error')
    }
  }

  const handleChat = async () => {
    if (!chatInput.trim() || chatLoading) return
    setChatLoading(true)
    try {
      const res = await companionController.chat(chatInput)
      setChatResponse(res.response)
    } catch {
      setChatResponse('[Error: could not reach model]')
    } finally {
      setChatLoading(false)
      setChatInput('')
    }
  }

  if (loading) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Companion" subtitle="AI personality management" />} />
        <div className="space-y-4">
          <KpiGrid>
            <StatCard label="Active Preset" value="..." />
            <StatCard label="Warmth" value="..." />
            <StatCard label="Curiosity" value="..." />
            <StatCard label="Creativity" value="..." />
          </KpiGrid>
          <Card><CardContent><div className="h-48 animate-pulse bg-muted/50 rounded" /></CardContent></Card>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="sl-page mx-auto max-w-4xl">
        <AppRouteHeader left={<AppRouteHeaderLead title="Companion" subtitle="AI personality management" />} />
        <div className="space-y-4">
          <Card>
            <CardContent className="text-center py-8">
              <p className="text-sm text-destructive mb-2">{error}</p>
              <Button size="sm" variant="ghost" onClick={() => window.location.reload()}>Retry</Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const avgTrait = traits ? Math.round(Object.values(traits).reduce((a, b) => a + b, 0) / Object.values(traits).length) : 0

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Companion" subtitle="AI personality management" />} />
      <div className="space-y-4">
        <KpiGrid>
          <StatCard label="Active Preset" value={presets.length > 0 ? presets[0].name : 'Custom'} />
          <StatCard label="Warmth" value={traits ? String(traits.warmth ?? 0) : '...'} />
          <StatCard label="Curiosity" value={traits ? String(traits.curiosity ?? 0) : '...'} />
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

        <CompanionInsightsCard traits={traits as unknown as Record<string, number>} presets={presets} />

        {traits && (
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <CardTitle className="text-base">Personality Traits</CardTitle>
              <div className="flex gap-1.5">
                <Button size="sm" variant="ghost" onClick={handleReset}>
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
          <CardHeader>
            <CardTitle className="text-base">Test Chat</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {chatResponse && (
              <div className="rounded-md bg-muted/30 p-3 text-sm">
                {chatResponse}
              </div>
            )}
            <div className="flex gap-2">
              <Input
                value={chatInput}
                onChange={e => setChatInput(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handleChat()}
                placeholder="Say something to your companion..."
                disabled={chatLoading}
              />
              <Button size="sm" onClick={handleChat} disabled={chatLoading || !chatInput.trim()}>
                {chatLoading ? '...' : 'Send'}
              </Button>
            </div>
            <div ref={chatEndRef} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
