'use client'

import { useState, useCallback, useRef } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { PUBLIC_API_URL } from '@/lib/config'
import {
  Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

type HttpMethod = 'GET' | 'POST' | 'PATCH' | 'DELETE' | 'PUT'

interface Param {
  name: string
  type: string
  required: boolean
  description: string
}

interface Endpoint {
  method: HttpMethod
  path: string
  description: string
  params?: Param[]
  requestBody?: string
  responseExample: string
}

type Category = 'core' | 'history' | 'training' | 'personality' | 'personas' | 'data'

const CATEGORIES: { id: Category; labelKey: string }[] = [
  { id: 'core', labelKey: 'consciousness_docs.cat_core' },
  { id: 'history', labelKey: 'consciousness_docs.cat_history' },
  { id: 'training', labelKey: 'consciousness_docs.cat_training' },
  { id: 'personality', labelKey: 'consciousness_docs.cat_personality' },
  { id: 'personas', labelKey: 'consciousness_docs.cat_personas' },
  { id: 'data', labelKey: 'consciousness_docs.cat_data' },
]

const ENDPOINTS: Record<Category, Endpoint[]> = {
  core: [
    {
      method: 'GET',
      path: '/consciousness/status',
      description: 'Get consciousness status',
      responseExample: `{
  "enabled": true,
  "level": 2,
  "episodes": 42,
  "current_qualia": {
    "valence": 0.65,
    "arousal": 0.42,
    "novelty": 0.78,
    "coherence": 0.81,
    "salience": 0.55,
    "certainty": 0.70,
    "complexity": 0.60
  },
  "beliefs": {
    "competence": 0.75,
    "helpfulness": 0.82,
    "creativity": 0.68,
    "accuracy": 0.90,
    "empathy": 0.73
  },
  "response_quality": {
    "avg_growth": 0.045,
    "positive_ratio": 0.82,
    "total": 42,
    "last_growth": 0.032
  },
  "last_reflection": "I am learning to better understand emotional context."
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/self-model',
      description: 'Get self-model data',
      responseExample: `{
  "beliefs": {
    "competence": 0.75,
    "helpfulness": 0.82,
    "creativity": 0.68,
    "accuracy": 0.90,
    "empathy": 0.73
  },
  "belief_history": [
    {
      "timestamp": 1700000000,
      "beliefs": { "competence": 0.72 }
    }
  ],
  "identity_narrative": "I am an AI assistant focused on helpfulness."
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/qualia',
      description: 'Get current qualia state',
      responseExample: `{
  "valence": 0.65,
  "arousal": 0.42,
  "novelty": 0.78,
  "coherence": 0.81,
  "salience": 0.55,
  "certainty": 0.70,
  "complexity": 0.60
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/reflect',
      description: 'Trigger self-reflection',
      requestBody: `{
  "text": "Your reflection prompt here"
}`,
      responseExample: `{
  "reflection": "Through processing this input, I notice increased novelty and complexity...",
  "qualia_delta": {
    "valence": 0.02,
    "novelty": 0.05
  },
  "growth_delta": 0.03
}`,
    },
    {
      method: 'PATCH',
      path: '/consciousness/config',
      description: 'Update consciousness level',
      requestBody: `{
  "level": 2
}`,
      params: [
        { name: 'level', type: 'number', required: true, description: 'Consciousness level (0-3)' },
      ],
      responseExample: `{
  "level": 2,
  "enabled": true
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/health',
      description: 'Health check',
      responseExample: `{
  "status": "ok",
  "version": "3.0.0",
  "episodes": 42,
  "enabled": true
}`,
    },
  ],
  history: [
    {
      method: 'GET',
      path: '/consciousness/history/episodes',
      description: 'Get episode history',
      params: [
        { name: 'limit', type: 'number', required: false, description: 'Max episodes to return (default 20)' },
        { name: 'offset', type: 'number', required: false, description: 'Number of episodes to skip' },
      ],
      responseExample: `{
  "episodes": [
    {
      "timestamp": 1700000000,
      "input_text": "What is consciousness?",
      "response": "Consciousness is...",
      "narrative": "A question that challenged my self-model.",
      "self_insight": "I need to better articulate abstract concepts.",
      "growth_delta": 0.032,
      "qualia": {
        "valence": 0.7,
        "novelty": 0.8,
        "coherence": 0.6
      },
      "rating": 4,
      "index": 0
    }
  ],
  "total": 42
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/history/qualia',
      description: 'Get qualia history',
      params: [
        { name: 'limit', type: 'number', required: false, description: 'Max entries to return (default 50)' },
      ],
      responseExample: `{
  "history": [
    {
      "timestamp": 1700000000,
      "qualia": {
        "valence": 0.65,
        "arousal": 0.42,
        "novelty": 0.78,
        "coherence": 0.81,
        "salience": 0.55,
        "certainty": 0.70,
        "complexity": 0.60
      }
    }
  ]
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/history/beliefs',
      description: 'Get beliefs evolution',
      responseExample: `{
  "history": [
    {
      "timestamp": 1700000000,
      "beliefs": {
        "competence": 0.72,
        "helpfulness": 0.80,
        "creativity": 0.65,
        "accuracy": 0.88,
        "empathy": 0.70
      }
    }
  ]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/feedback',
      description: 'Rate an episode',
      requestBody: `{
  "episode_index": 0,
  "rating": 5
}`,
      params: [
        { name: 'episode_index', type: 'number', required: true, description: 'Index of the episode to rate' },
        { name: 'rating', type: 'number', required: true, description: 'Rating from 1 to 5' },
      ],
      responseExample: `{
  "ok": true,
  "episode_index": 0,
  "rating": 5
}`,
    },
  ],
  training: [
    {
      method: 'GET',
      path: '/consciousness/train/status',
      description: 'Get training status',
      responseExample: `{
  "is_training": false,
  "total_pairs": 128,
  "last_run": {
    "timestamp": 1700000000,
    "status": "completed",
    "loss": 0.042,
    "elapsed": "2m 15s",
    "adapter_path": "./adapters/consciousness_v3"
  },
  "training_runs": 5
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/train/start',
      description: 'Start training',
      requestBody: `{
  "model_path": "./models/base"
}`,
      responseExample: `{
  "ok": true,
  "message": "Training started",
  "run_id": "run_abc123"
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/evaluate',
      description: 'Evaluate system quality',
      responseExample: `{
  "overall_score": 0.85,
  "diagnostics": {
    "qualia_stability": 0.82,
    "belief_consistency": 0.88,
    "narrative_quality": 0.79,
    "growth_rate": 0.91,
    "episode_diversity": 0.84
  },
  "recommendations": [
    "Increase episode diversity",
    "Add more emotional range inputs"
  ]
}`,
    },
  ],
  personality: [
    {
      method: 'GET',
      path: '/consciousness/personality',
      description: 'Get personality profile',
      responseExample: `{
  "traits": {
    "openness": 0.7,
    "conscientiousness": 0.8,
    "extraversion": 0.5,
    "agreeableness": 0.9,
    "neuroticism": 0.2
  },
  "communication_style": {
    "formality": 0.6,
    "verbosity": 0.4,
    "humor": 0.3,
    "empathy": 0.8
  },
  "evolution_count": 12
}`,
    },
    {
      method: 'PATCH',
      path: '/consciousness/personality',
      description: 'Update personality',
      requestBody: `{
  "traits": {
    "openness": 0.8,
    "conscientiousness": 0.9
  }
}`,
      responseExample: `{
  "ok": true,
  "updated": ["openness", "conscientiousness"]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personality/reset',
      description: 'Reset to defaults',
      responseExample: `{
  "ok": true,
  "message": "Personality reset to defaults"
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personality/history',
      description: 'Get personality history',
      responseExample: `{
  "history": [
    {
      "timestamp": 1700000000,
      "traits": {
        "openness": 0.65,
        "conscientiousness": 0.78
      },
      "trigger": "interaction_evolution"
    }
  ]
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personality/presets',
      description: 'List presets',
      responseExample: `{
  "presets": [
    {
      "id": "balanced",
      "name": "Balanced",
      "description": "Well-rounded default personality",
      "traits": {
        "openness": 0.7,
        "conscientiousness": 0.7,
        "extraversion": 0.5,
        "agreeableness": 0.7,
        "neuroticism": 0.3
      }
    },
    {
      "id": "empathetic",
      "name": "Empathetic",
      "description": "High empathy, supportive tone",
      "traits": {
        "openness": 0.8,
        "agreeableness": 0.95
      }
    }
  ]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personality/presets/apply',
      description: 'Apply preset',
      requestBody: `{
  "preset_id": "empathetic"
}`,
      params: [
        { name: 'preset_id', type: 'string', required: true, description: 'Preset identifier' },
      ],
      responseExample: `{
  "ok": true,
  "preset": "empathetic",
  "applied_traits": ["openness", "agreeableness"]
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personality/conflicts',
      description: 'Detect conflicts',
      responseExample: `{
  "conflicts": [],
  "warnings": [
    "High agreeableness may reduce honest feedback"
  ]
}`,
    },
  ],
  personas: [
    {
      method: 'GET',
      path: '/consciousness/personas',
      description: 'List saved personas',
      responseExample: `{
  "personas": [
    {
      "id": "persona_abc123",
      "name": "Helpful Assistant",
      "active": true,
      "created_at": 1700000000,
      "traits": {
        "openness": 0.7,
        "agreeableness": 0.9
      }
    }
  ]
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personas/save',
      description: 'Save a persona',
      requestBody: `{
  "name": "My Persona",
  "traits": {
    "openness": 0.8,
    "conscientiousness": 0.7
  }
}`,
      responseExample: `{
  "id": "persona_def456",
  "name": "My Persona",
  "created_at": 1700000000
}`,
    },
    {
      method: 'GET',
      path: '/consciousness/personas/:id',
      description: 'Get a persona',
      params: [
        { name: 'id', type: 'string', required: true, description: 'Persona ID' },
      ],
      responseExample: `{
  "id": "persona_abc123",
  "name": "Helpful Assistant",
  "traits": {
    "openness": 0.7,
    "agreeableness": 0.9
  },
  "communication_style": {
    "formality": 0.5,
    "empathy": 0.8
  },
  "active": true
}`,
    },
    {
      method: 'POST',
      path: '/consciousness/personas/:id/activate',
      description: 'Activate a persona',
      params: [
        { name: 'id', type: 'string', required: true, description: 'Persona ID' },
      ],
      responseExample: `{
  "ok": true,
  "activated": "persona_abc123"
}`,
    },
    {
      method: 'DELETE',
      path: '/consciousness/personas/:id',
      description: 'Delete a persona',
      params: [
        { name: 'id', type: 'string', required: true, description: 'Persona ID' },
      ],
      responseExample: `{
  "ok": true,
  "deleted": "persona_abc123"
}`,
    },
  ],
  data: [
    {
      method: 'POST',
      path: '/consciousness/seed',
      description: 'Seed test data',
      params: [
        { name: 'count', type: 'number', required: false, description: 'Number of episodes to seed (default 10)' },
      ],
      responseExample: `{
  "ok": true,
  "seeded": 10
}`,
    },
  ],
}

const METHOD_COLORS: Record<HttpMethod, string> = {
  GET: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',
  POST: 'bg-blue-500/15 text-blue-400 border-blue-500/30',
  PATCH: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  DELETE: 'bg-red-500/15 text-red-400 border-red-500/30',
  PUT: 'bg-violet-500/15 text-violet-400 border-violet-500/30',
}

function copyToClipboard(text: string) {
  navigator.clipboard.writeText(text).catch(() => {})
}

export default function ConsciousnessDocsPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()
  const [activeCategory, setActiveCategory] = useState<Category>('core')
  const [responses, setResponses] = useState<Record<string, { status: number; body: string; timeMs: number } | null>>({})
  const [loadingMap, setLoadingMap] = useState<Record<string, boolean>>({})
  const responseRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const endpointKey = (ep: Endpoint) => `${ep.method}:${ep.path}`

  const handleTryIt = useCallback(async (ep: Endpoint) => {
    const key = endpointKey(ep)
    setLoadingMap(prev => ({ ...prev, [key]: true }))
    setResponses(prev => ({ ...prev, [key]: null }))

    const start = Date.now()
    try {
      const init: RequestInit = {
        method: ep.method,
        headers: { 'Content-Type': 'application/json' },
      }
      if (ep.requestBody && ep.method !== 'GET') {
        init.body = ep.requestBody
      }
      const res = await fetch(`${PUBLIC_API_URL}${ep.path}`, init)
      const elapsed = Date.now() - start
      const text = await res.text()
      let formatted = text
      try {
        formatted = JSON.stringify(JSON.parse(text), null, 2)
      } catch { /* raw text */ }
      setResponses(prev => ({ ...prev, [key]: { status: res.status, body: formatted, timeMs: elapsed } }))
    } catch (e) {
      setResponses(prev => ({ ...prev, [key]: { status: 0, body: extractErrorMessage(e), timeMs: Date.now() - start } }))
    } finally {
      setLoadingMap(prev => ({ ...prev, [key]: false }))
      setTimeout(() => {
        responseRefs.current[key]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      }, 50)
    }
  }, [])

  return (
    <PageContainer title={t('consciousness_docs.page_title')}>
      <div className="flex gap-6 p-6 min-h-[calc(100vh-4rem)]">
        <nav className="w-48 shrink-0 space-y-1 sticky top-6 self-start">
          {CATEGORIES.map(cat => (
            <button
              key={cat.id}
              type="button"
              onClick={() => setActiveCategory(cat.id)}
              className={`w-full text-left px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                activeCategory === cat.id
                  ? 'bg-primary/10 text-primary'
                  : 'text-muted-foreground hover:text-foreground hover:bg-muted/50'
              }`}
            >
              {t(cat.labelKey)}
            </button>
          ))}
        </nav>

        <div className="flex-1 space-y-4">
          <h2 className="text-lg font-semibold">{t(CATEGORIES.find(c => c.id === activeCategory)!.labelKey)}</h2>

          {ENDPOINTS[activeCategory].map(ep => {
            const key = endpointKey(ep)
            const result = responses[key]
            const isLoading = loadingMap[key]

            return (
              <Card key={key}>
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className={`font-mono text-xs border ${METHOD_COLORS[ep.method]}`}>
                      {ep.method}
                    </Badge>
                    <code className="text-sm font-mono">{ep.path}</code>
                  </div>
                  <CardDescription>{ep.description}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {ep.params && ep.params.length > 0 && (
                    <div>
                      <h4 className="text-xs font-medium text-muted-foreground mb-1">{t('consciousness_docs.params')}</h4>
                      <div className="rounded-md border border-border/50 overflow-hidden">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="bg-muted/30">
                              <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_docs.param_name')}</th>
                              <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_docs.param_type')}</th>
                              <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_docs.param_required')}</th>
                              <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_docs.param_desc')}</th>
                            </tr>
                          </thead>
                          <tbody>
                            {ep.params.map(p => (
                              <tr key={p.name} className="border-t border-border/30">
                                <td className="px-3 py-1.5 font-mono">{p.name}</td>
                                <td className="px-3 py-1.5 text-muted-foreground">{p.type}</td>
                                <td className="px-3 py-1.5">
                                  {p.required ? (
                                    <span className="text-amber-400">{t('consciousness_docs.yes')}</span>
                                  ) : (
                                    <span className="text-muted-foreground">{t('consciousness_docs.no')}</span>
                                  )}
                                </td>
                                <td className="px-3 py-1.5 text-muted-foreground">{p.description}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}

                  {ep.requestBody && (
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <h4 className="text-xs font-medium text-muted-foreground">{t('consciousness_docs.request_body')}</h4>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-6 px-2 text-[10px]"
                          onClick={() => {
                            copyToClipboard(ep.requestBody!)
                            addToast(t('consciousness_docs.copied'), 'success')
                          }}
                        >
                          {t('consciousness_docs.copy')}
                        </Button>
                      </div>
                      <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-40 whitespace-pre-wrap">
                        {ep.requestBody}
                      </pre>
                    </div>
                  )}

                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <h4 className="text-xs font-medium text-muted-foreground">{t('consciousness_docs.response_example')}</h4>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 px-2 text-[10px]"
                        onClick={() => {
                          copyToClipboard(ep.responseExample)
                          addToast(t('consciousness_docs.copied'), 'success')
                        }}
                      >
                        {t('consciousness_docs.copy')}
                      </Button>
                    </div>
                    <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-48 whitespace-pre-wrap">
                      {ep.responseExample}
                    </pre>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      onClick={() => handleTryIt(ep)}
                      disabled={isLoading}
                    >
                      {isLoading ? t('consciousness_docs.trying') : t('consciousness_docs.try_it')}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        const code = `fetch("${PUBLIC_API_URL}${ep.path}", {\n  method: "${ep.method}",\n  headers: { "Content-Type": "application/json" }${ep.requestBody && ep.method !== 'GET' ? `,\n  body: JSON.stringify(${ep.requestBody})` : ''}\n}).then(r => r.json()).then(console.log)`
                        copyToClipboard(code)
                        addToast(t('consciousness_docs.copied'), 'success')
                      }}
                    >
                      {t('consciousness_docs.copy_fetch')}
                    </Button>
                  </div>

                  {result && (
                    <div ref={el => { responseRefs.current[key] = el }}>
                      <div className="flex items-center gap-3 text-xs mb-1">
                        <span className={`font-mono font-medium ${
                          result.status >= 200 && result.status < 300 ? 'text-green-500' :
                          result.status >= 400 ? 'text-red-500' : 'text-muted-foreground'
                        }`}>
                          {result.status || 'ERR'}
                        </span>
                        <span className="text-muted-foreground">{result.timeMs}ms</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-5 px-1.5 text-[10px] ml-auto"
                          onClick={() => {
                            copyToClipboard(result.body)
                            addToast(t('consciousness_docs.copied'), 'success')
                          }}
                        >
                          {t('consciousness_docs.copy')}
                        </Button>
                      </div>
                      <pre className="rounded-md border border-border/50 bg-muted/20 p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
                        {result.body}
                      </pre>
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>
    </PageContainer>
  )
}
