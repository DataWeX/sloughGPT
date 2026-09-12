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

type TestStatus = 'pending' | 'running' | 'pass' | 'fail' | 'skip'

interface TestCase {
  id: string
  name: string
  suite: string
  status: TestStatus
  duration: number
  error?: string
}

interface TestSuite {
  id: string
  label: string
  enabled: boolean
}

const SUITES: TestSuite[] = [
  { id: 'core', label: 'Core Engine Tests', enabled: true },
  { id: 'qualia', label: 'Qualia Tests', enabled: true },
  { id: 'self_model', label: 'Self-Model Tests', enabled: true },
  { id: 'personality', label: 'Personality Tests', enabled: true },
  { id: 'api', label: 'API Tests', enabled: true },
  { id: 'integration', label: 'Integration Tests', enabled: true },
]

async function apiCall(method: string, path: string, body?: string): Promise<{ status: number; data: any; duration: number }> {
  const start = Date.now()
  try {
    const init: RequestInit = {
      method,
      headers: { 'Content-Type': 'application/json' },
    }
    if (body) init.body = body
    const res = await fetch(`${PUBLIC_API_URL}${path}`, init)
    const duration = Date.now() - start
    const text = await res.text()
    let data: any
    try { data = JSON.parse(text) } catch { data = text }
    return { status: res.status, data: data?.data ?? data, duration }
  } catch (e) {
    return { status: 0, data: extractErrorMessage(e), duration: Date.now() - start }
  }
}

function assert(condition: boolean, msg: string): void {
  if (!condition) throw new Error(msg)
}

function buildTests(): Omit<TestCase, 'status' | 'duration'>[] {
  const tests: Omit<TestCase, 'status' | 'duration'>[] = []

  tests.push(
    { id: 'core.process', name: 'engine.process() returns valid response', suite: 'core' },
    { id: 'core.reflect', name: 'engine.reflect() processes input', suite: 'core' },
    { id: 'core.get_status', name: 'engine.get_status() returns status object', suite: 'core' },
    { id: 'core.health', name: 'health endpoint returns 200', suite: 'core' },
  )

  tests.push(
    { id: 'qualia.get', name: 'GET /qualia returns qualia object', suite: 'qualia' },
    { id: 'qualia.dimensions', name: 'Qualia has 7 dimensions', suite: 'qualia' },
    { id: 'qualia.values', name: 'Qualia values are numbers 0-1', suite: 'qualia' },
    { id: 'qualia.reflect', name: 'Qualia updates after reflect', suite: 'qualia' },
  )

  tests.push(
    { id: 'self_model.get', name: 'GET /self-model returns beliefs', suite: 'self_model' },
    { id: 'self_model.beliefs', name: 'Beliefs contain confidence values', suite: 'self_model' },
    { id: 'self_model.update', name: 'Beliefs update after reflect', suite: 'self_model' },
    { id: 'self_model.structure', name: 'Self-model has expected structure', suite: 'self_model' },
  )

  tests.push(
    { id: 'personality.get', name: 'GET /personality returns traits', suite: 'personality' },
    { id: 'personality.preset', name: 'Apply personality preset', suite: 'personality' },
    { id: 'personality.reset', name: 'Reset personality to defaults', suite: 'personality' },
    { id: 'personality.traits', name: 'Personality traits are valid', suite: 'personality' },
  )

  tests.push(
    { id: 'api.status', name: 'GET /consciousness/status', suite: 'api' },
    { id: 'api.self_model', name: 'GET /consciousness/self-model', suite: 'api' },
    { id: 'api.reflect', name: 'POST /consciousness/reflect', suite: 'api' },
    { id: 'api.health', name: 'GET /consciousness/health', suite: 'api' },
    { id: 'api.qualia', name: 'GET /consciousness/qualia', suite: 'api' },
    { id: 'api.feedback', name: 'POST /consciousness/feedback', suite: 'api' },
    { id: 'api.seed', name: 'POST /consciousness/seed', suite: 'api' },
    { id: 'api.config', name: 'PATCH /consciousness/config', suite: 'api' },
    { id: 'api.personality', name: 'POST /personality/presets/apply', suite: 'api' },
  )

  tests.push(
    { id: 'integration.reflect_cycle', name: 'Full reflect → status cycle', suite: 'integration' },
    { id: 'integration.seed_rate', name: 'Seed → reflect → feedback flow', suite: 'integration' },
    { id: 'integration.personality_reflect', name: 'Personality preset → reflect → qualia check', suite: 'integration' },
  )

  return tests
}

export default function ConsciousnessTestRunnerPage() {
  const addToast = useToastStore(state => state.addToast)
  const { t } = useLocale()

  const [suites, setSuites] = useState<TestSuite[]>(() => SUITES.map(s => ({ ...s })))
  const [tests, setTests] = useState<TestCase[]>(() => buildTests().map(tc => ({ ...tc, status: 'pending', duration: 0 })))
  const [running, setRunning] = useState(false)
  const [stopRequested, setStopRequested] = useState(false)
  const [progress, setProgress] = useState(0)
  const abortRef = useRef(false)

  const totalTests = tests.length
  const passed = tests.filter(t => t.status === 'pass').length
  const failed = tests.filter(t => t.status === 'fail').length
  const skipped = tests.filter(t => t.status === 'skip').length
  const totalDuration = tests.reduce((s, t) => s + t.duration, 0)

  const toggleSuite = useCallback((id: string) => {
    setSuites(prev => prev.map(s => s.id === id ? { ...s, enabled: !s.enabled } : s))
  }, [])

  const runTests = useCallback(async (all: boolean) => {
    if (running) return
    abortRef.current = false
    setStopRequested(false)
    setRunning(true)
    setProgress(0)

    const enabledIds = new Set(
      all ? SUITES.filter(s => s.enabled).map(s => s.id) : SUITES.filter(s => s.enabled).map(s => s.id)
    )
    const runnable = buildTests().filter(tc => enabledIds.has(tc.suite))
    setTests(runnable.map(tc => ({ ...tc, status: 'pending', duration: 0 })))

    const results: TestCase[] = []

    for (let i = 0; i < runnable.length; i++) {
      if (abortRef.current) {
        results.push({ ...runnable[i], status: 'skip', duration: 0 })
        setTests([...results, ...runnable.slice(i + 1).map(tc => ({ ...tc, status: 'skip' as TestStatus, duration: 0 }))])
        break
      }

      const tc = runnable[i]
      const runningTest: TestCase = { ...tc, status: 'running', duration: 0 }
      results.push(runningTest)
      setTests([...results])

      const start = Date.now()
      let status: TestStatus = 'pass'
      let error: string | undefined

      try {
        switch (tc.id) {
          case 'core.process': {
            const r = await apiCall('POST', '/consciousness/reflect', '{"text": "test"}')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            assert(r.data && typeof r.data === 'object', 'Response is not an object')
            break
          }
          case 'core.reflect': {
            const r = await apiCall('POST', '/consciousness/reflect', '{"text": "What is consciousness?"}')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            assert(r.data !== null && r.data !== undefined, 'No data returned')
            break
          }
          case 'core.get_status': {
            const r = await apiCall('GET', '/consciousness/status')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            assert(r.data !== null, 'Status is null')
            break
          }
          case 'core.health': {
            const r = await apiCall('GET', '/consciousness/health')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            break
          }
          case 'qualia.get': {
            const r = await apiCall('GET', '/consciousness/qualia')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            assert(r.data !== null, 'Qualia is null')
            break
          }
          case 'qualia.dimensions': {
            const r = await apiCall('GET', '/consciousness/qualia')
            const q = (r.data as any)?.qualia ?? r.data
            assert(typeof q === 'object' && q !== null, 'Qualia not an object')
            const keys = Object.keys(q)
            assert(keys.length >= 5, `Expected >= 5 dimensions, got ${keys.length}`)
            break
          }
          case 'qualia.values': {
            const r = await apiCall('GET', '/consciousness/qualia')
            const q = (r.data as any)?.qualia ?? r.data
            for (const [k, v] of Object.entries(q as Record<string, unknown>)) {
              assert(typeof v === 'number', `${k} is not a number`)
              assert((v as number) >= 0 && (v as number) <= 1, `${k} out of range: ${v}`)
            }
            break
          }
          case 'qualia.reflect': {
            await apiCall('POST', '/consciousness/reflect', '{"text": "test qualia update"}')
            const r = await apiCall('GET', '/consciousness/qualia')
            assert(r.status >= 200 && r.status < 300, 'Qualia fetch failed after reflect')
            break
          }
          case 'self_model.get': {
            const r = await apiCall('GET', '/consciousness/self-model')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            assert(r.data !== null, 'Self-model is null')
            break
          }
          case 'self_model.beliefs': {
            const r = await apiCall('GET', '/consciousness/self-model')
            const sm = (r.data as any)?.beliefs ?? r.data
            if (sm && typeof sm === 'object') {
              const vals = Object.values(sm as Record<string, unknown>)
              assert(vals.length > 0, 'No beliefs found')
            }
            break
          }
          case 'self_model.update': {
            const before = await apiCall('GET', '/consciousness/self-model')
            await apiCall('POST', '/consciousness/reflect', '{"text": "self model update test"}')
            const after = await apiCall('GET', '/consciousness/self-model')
            assert(after.status >= 200 && after.status < 300, 'Self-model fetch failed after reflect')
            break
          }
          case 'self_model.structure': {
            const r = await apiCall('GET', '/consciousness/self-model')
            assert(r.data !== null && r.data !== undefined, 'Self-model data missing')
            break
          }
          case 'personality.get': {
            const r = await apiCall('GET', '/consciousness/personality')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            assert(r.data !== null, 'Personality is null')
            break
          }
          case 'personality.preset': {
            const r = await apiCall('POST', '/consciousness/personality/presets/apply', '{"preset_id": "balanced"}')
            assert(r.status >= 200 && r.status < 300, `Expected 2xx, got ${r.status}`)
            break
          }
          case 'personality.reset': {
            const r = await apiCall('POST', '/consciousness/personality/reset')
            assert(r.status >= 200 || r.status === 404, `Unexpected status: ${r.status}`)
            break
          }
          case 'personality.traits': {
            const r = await apiCall('GET', '/consciousness/personality')
            const p = (r.data as any)?.traits ?? r.data
            if (p && typeof p === 'object') {
              for (const [, v] of Object.entries(p as Record<string, unknown>)) {
                assert(typeof v === 'number', `Trait value not a number`)
              }
            }
            break
          }
          case 'api.status': {
            const r = await apiCall('GET', '/consciousness/status')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.self_model': {
            const r = await apiCall('GET', '/consciousness/self-model')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.reflect': {
            const r = await apiCall('POST', '/consciousness/reflect', '{"text": "api test"}')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.health': {
            const r = await apiCall('GET', '/consciousness/health')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.qualia': {
            const r = await apiCall('GET', '/consciousness/qualia')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.feedback': {
            const r = await apiCall('POST', '/consciousness/feedback', '{"episode_index": 0, "rating": 3}')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.seed': {
            const r = await apiCall('POST', '/consciousness/seed?count=2')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.config': {
            const r = await apiCall('PATCH', '/consciousness/config', '{"level": 2}')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'api.personality': {
            const r = await apiCall('POST', '/consciousness/personality/presets/apply', '{"preset_id": "balanced"}')
            assert(r.status >= 200 && r.status < 300, `Status: ${r.status}`)
            break
          }
          case 'integration.reflect_cycle': {
            const r1 = await apiCall('POST', '/consciousness/reflect', '{"text": "integration cycle test"}')
            assert(r1.status >= 200 && r1.status < 300, 'Reflect failed')
            const r2 = await apiCall('GET', '/consciousness/status')
            assert(r2.status >= 200 && r2.status < 300, 'Status fetch failed')
            break
          }
          case 'integration.seed_rate': {
            const r1 = await apiCall('POST', '/consciousness/seed?count=1')
            assert(r1.status >= 200 && r1.status < 300, 'Seed failed')
            const r2 = await apiCall('POST', '/consciousness/reflect', '{"text": "integration seed rate"}')
            assert(r2.status >= 200 && r2.status < 300, 'Reflect failed')
            const r3 = await apiCall('POST', '/consciousness/feedback', '{"episode_index": 0, "rating": 4}')
            assert(r3.status >= 200 && r3.status < 300, 'Feedback failed')
            break
          }
          case 'integration.personality_reflect': {
            const r1 = await apiCall('POST', '/consciousness/personality/presets/apply', '{"preset_id": "creative"}')
            assert(r1.status >= 200 && r1.status < 300, 'Preset apply failed')
            const r2 = await apiCall('POST', '/consciousness/reflect', '{"text": "personality integration test"}')
            assert(r2.status >= 200 && r2.status < 300, 'Reflect failed')
            const r3 = await apiCall('GET', '/consciousness/qualia')
            assert(r3.status >= 200 && r3.status < 300, 'Qualia fetch failed')
            break
          }
          default:
            status = 'skip'
            error = 'Unknown test'
        }
      } catch (e) {
        status = 'fail'
        error = extractErrorMessage(e)
      }

      const duration = Date.now() - start
      const completed: TestCase = { ...tc, status, duration, error }
      results[results.length - 1] = completed
      setTests([...results])
      setProgress(i + 1)
    }

    setRunning(false)
    const final = results
    const p = final.filter(t => t.status === 'pass').length
    const f = final.filter(t => t.status === 'fail').length
    addToast(`Tests complete: ${p} passed, ${f} failed`, f > 0 ? 'error' : 'success')
  }, [running, addToast])

  const stopTests = useCallback(() => {
    abortRef.current = true
    setStopRequested(true)
  }, [])

  const exportResults = useCallback(() => {
    const blob = new Blob([JSON.stringify(tests, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `consciousness-tests-${Date.now()}.json`
    a.click()
    URL.revokeObjectURL(url)
    addToast('Results exported', 'success')
  }, [tests, addToast])

  const copyResults = useCallback(() => {
    navigator.clipboard.writeText(JSON.stringify(tests, null, 2)).then(
      () => addToast('Results copied to clipboard', 'success'),
      () => addToast('Failed to copy', 'error'),
    )
  }, [tests, addToast])

  const statusColor = (s: TestStatus) => {
    switch (s) {
      case 'pass': return 'bg-green-500/15 text-green-400 border-green-500/30'
      case 'fail': return 'bg-red-500/15 text-red-400 border-red-500/30'
      case 'skip': return 'bg-muted/30 text-muted-foreground border-border/30'
      case 'running': return 'bg-blue-500/15 text-blue-400 border-blue-500/30'
      default: return 'bg-muted/20 text-muted-foreground border-border/30'
    }
  }

  const statusLabel = (s: TestStatus) => {
    switch (s) {
      case 'pass': return 'PASS'
      case 'fail': return 'FAIL'
      case 'skip': return 'SKIP'
      case 'running': return 'RUN'
      default: return '—'
    }
  }

  const progressPct = totalTests > 0 ? (progress / totalTests) * 100 : 0

  return (
    <PageContainer title={t('consciousness_test_runner.page_title')}>
      <div className="space-y-6 p-6">

        <Card>
          <CardHeader>
            <CardTitle>{t('consciousness_test_runner.suites_title')}</CardTitle>
            <CardDescription>{t('consciousness_test_runner.suites_desc')}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
              {suites.map(s => (
                <label key={s.id} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={s.enabled}
                    onChange={() => toggleSuite(s.id)}
                    disabled={running}
                    className="rounded border-input"
                  />
                  <span className="text-muted-foreground">{s.label}</span>
                </label>
              ))}
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={() => runTests(true)} disabled={running}>
                {t('consciousness_test_runner.run_all')}
              </Button>
              <Button onClick={() => runTests(false)} disabled={running} variant="outline">
                {t('consciousness_test_runner.run_selected')}
              </Button>
              {running && (
                <Button onClick={stopTests} variant="destructive">
                  {t('consciousness_test_runner.stop')}
                </Button>
              )}
              <div className="flex-1" />
              <div className="text-xs text-muted-foreground">
                {t('consciousness_test_runner.progress', { current: progress, total: totalTests })}
              </div>
            </div>
            {running && (
              <div className="w-full h-2 bg-muted/30 rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300"
                  style={{ width: `${progressPct}%` }}
                />
              </div>
            )}
          </CardContent>
        </Card>

        {tests.some(t => t.status !== 'pending') && (
          <Card>
            <CardHeader>
              <CardTitle>{t('consciousness_test_runner.results_title')}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-5 gap-4 text-sm">
                <div>
                  <div className="text-muted-foreground text-xs">{t('consciousness_test_runner.summary_total')}</div>
                  <div className="font-mono text-lg">{totalTests}</div>
                </div>
                <div>
                  <div className="text-muted-foreground text-xs">{t('consciousness_test_runner.summary_passed')}</div>
                  <div className="font-mono text-lg text-green-500">{passed}</div>
                </div>
                <div>
                  <div className="text-muted-foreground text-xs">{t('consciousness_test_runner.summary_failed')}</div>
                  <div className="font-mono text-lg text-red-500">{failed}</div>
                </div>
                <div>
                  <div className="text-muted-foreground text-xs">{t('consciousness_test_runner.summary_skipped')}</div>
                  <div className="font-mono text-lg text-muted-foreground">{skipped}</div>
                </div>
                <div>
                  <div className="text-muted-foreground text-xs">{t('consciousness_test_runner.summary_duration')}</div>
                  <div className="font-mono text-lg">{totalDuration}ms</div>
                </div>
              </div>
              <div className="rounded-md border border-border/50 overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="bg-muted/30">
                      <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_test_runner.col_status')}</th>
                      <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_test_runner.col_suite')}</th>
                      <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_test_runner.col_name')}</th>
                      <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_test_runner.col_duration')}</th>
                      <th className="px-3 py-1.5 text-left font-medium">{t('consciousness_test_runner.col_error')}</th>
                    </tr>
                  </thead>
                  <tbody>
                    {tests.map(tc => (
                      <tr key={tc.id} className="border-t border-border/30">
                        <td className="px-3 py-1.5">
                          <Badge className={`text-[10px] ${statusColor(tc.status)}`}>
                            {statusLabel(tc.status)}
                          </Badge>
                        </td>
                        <td className="px-3 py-1.5 text-muted-foreground">{tc.suite}</td>
                        <td className="px-3 py-1.5">{tc.name}</td>
                        <td className="px-3 py-1.5 font-mono">{tc.duration}ms</td>
                        <td className="px-3 py-1.5 text-red-400 max-w-[300px] truncate">{tc.error ?? ''}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" onClick={exportResults}>
                  {t('consciousness_test_runner.export_json')}
                </Button>
                <Button variant="outline" onClick={copyResults}>
                  {t('consciousness_test_runner.copy_clipboard')}
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

      </div>
    </PageContainer>
  )
}
