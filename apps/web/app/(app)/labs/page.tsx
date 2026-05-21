'use client'

import { useState, useEffect, useRef } from 'react'
import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { Toggle } from '@/components/ui'
import { FoldSection } from '@/components/strui'
import { cn } from '@/lib/cn'
import { modelController } from '@/lib/model-controller'
import { soulsController, type Checkpoint } from '@/lib/souls-controller'
import { generateController } from '@/lib/generate-controller'
import { multimodalController } from '@/lib/multimodal-controller'
import { benchmarkController } from '@/lib/benchmark-controller'

export default function LabsPage() {
  const [modelStatus, setModelStatus] = useState<any>(null)
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [prompt, setPrompt] = useState('')
  const [response, setResponse] = useState('')
  const [chatting, setChatting] = useState(false)

  const [chatLog, setChatLog] = useState<{ role: string; text: string }[]>([])
  const [tab, setTab] = useState<'chat' | 'train'>('chat')
  const [personalizationEnabled, setPersonalizationEnabled] = useState(() => {
    try { return localStorage.getItem('labs_personalization') !== 'false' } catch { return true }
  })
  const [visionFile, setVisionFile] = useState<File | null>(null)
  const [visionPreview, setVisionPreview] = useState('')
  const [visionCaption, setVisionCaption] = useState('')
  const [visionBusy, setVisionBusy] = useState(false)
  const [initText, setInitText] = useState('')
  const [initSoul, setInitSoul] = useState('custom')
  const [initEpochs, setInitEpochs] = useState(5)
  const [initializing, setInitializing] = useState(false)
  const [trainProgress, setTrainProgress] = useState(0)
  const [trainCelebration, setTrainCelebration] = useState<string | null>(null)
  const [history, setHistory] = useState<any[]>([])
  const [perplexity, setPerplexity] = useState<number | null>(null)
  const [pplLoading, setPplLoading] = useState(false)
  const chatEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    refreshModel()
    listCheckpoints()
    try {
      const saved = JSON.parse(localStorage.getItem('labs_training_history') || '[]')
      setHistory(saved)
    } catch {}
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatLog])

  const refreshModel = async () => {
    try {
      const health = await modelController.getHealth()
      setModelStatus(health)
    } catch {}
  }

  const listCheckpoints = async () => {
    try {
      const result = await soulsController.listCheckpoints()
      setCheckpoints((result.checkpoints || []).slice(0, 20))
    } catch {}
  }

  const handleInitTraining = async () => {
    if (!initText.trim()) return
    setInitializing(true)
    setTrainProgress(0)
    const prog = setInterval(() => setTrainProgress(p => Math.min(p + 5, 90)), 1000)
    try {
      const { apiClient } = await import('@/lib/http-client')
      await apiClient.post('/auto-train/start', {
        teacher_model: 'gpt2',
        epochs: initEpochs,
        soul_name: initSoul,
        source_text: initText,
      })
      setTrainProgress(95)
      await refreshModel()
      await listCheckpoints()
      let pplValue = perplexity
      try {
        const ppl = await benchmarkController.run({ dataset: 'the quick brown fox jumps over the lazy dog' })
        if (ppl.perplexity) {
          setPerplexity(ppl.perplexity)
          pplValue = ppl.perplexity
        }
      } catch {}
      const saved = JSON.parse(localStorage.getItem('labs_training_history') || '[]')
      saved.push({ soul: initSoul, epochs: initEpochs, ppl: pplValue, time: new Date().toISOString() })
      localStorage.setItem('labs_training_history', JSON.stringify(saved.slice(-50)))
      setHistory(saved)
      const cheers = ['Trained!', 'Model ready!', 'Fresh model!', 'All done!', 'Ready to chat!']
      setTrainCelebration(cheers[Math.floor(Math.random() * cheers.length)])
      setTimeout(() => setTrainCelebration(null), 3000)
    } catch {
      setTrainCelebration('Training failed')
      setTimeout(() => setTrainCelebration(null), 3000)
    } finally {
      clearInterval(prog)
      setTrainProgress(100)
      setTimeout(() => setTrainProgress(0), 1500)
      setInitializing(false)
    }
  }

  const handleChat = async () => {
    if (!prompt.trim()) return
    setChatting(true)
    setResponse('')
    let fullResponse = ''
    try {
      await generateController.generateStream(
        { prompt, max_new_tokens: 100, temperature: 0.8 },
        (token) => {
          setResponse(prev => prev + token)
          fullResponse += token
        },
        () => {
          if (fullResponse) {
            setChatLog(prev => [...prev, { role: 'user', text: prompt }, { role: 'assistant', text: fullResponse }])
          }
        },
        (error) => setResponse(`Error: ${error}`),
      )
    } catch (e: any) {
      setResponse(`Error: ${e.message}`)
    } finally {
      setChatting(false)
    }
  }

  const handlePpl = async () => {
    setPplLoading(true)
    try {
      const ppl = await benchmarkController.run({ dataset: 'the quick brown fox jumps over the lazy dog' })
      if (ppl.perplexity) setPerplexity(ppl.perplexity)
    } catch {} finally { setPplLoading(false) }
  }

  const handleVision = async () => {
    if (!visionFile) return
    setVisionBusy(true)
    setVisionCaption('processing...')
    try {
      const dataUrl = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = () => resolve(reader.result as string)
        reader.onerror = reject
        reader.readAsDataURL(visionFile)
      })
      const result = await multimodalController.trainImage(dataUrl, visionFile.name)
      setVisionCaption(result.caption || JSON.stringify(result))
    } catch (e: any) {
      setVisionCaption(`Error: ${e.message}`)
    } finally {
      setVisionBusy(false)
    }
  }

  const handleReset = async () => {
    setResponse('')
    setPrompt('')
    setChatLog([])
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader
        left={<AppRouteHeaderLead title="Playground" subtitle="Train and chat with custom models" />}
      />

      <div className="space-y-4">

      {/* Tab bar */}
      <div className="flex gap-2">
        <Button size="sm" variant={tab === 'chat' ? 'default' : 'outline'} onClick={() => setTab('chat')}>Chat</Button>
        <Button size="sm" variant={tab === 'train' ? 'default' : 'outline'} onClick={() => setTab('train')}>Train</Button>
      </div>

      {/* Model status */}
      <Card>
        <CardHeader><CardTitle className="text-base">Model Status</CardTitle></CardHeader>
        <CardContent className="flex gap-4 text-sm flex-wrap items-center">
          <Badge variant={modelStatus?.model_loaded ? 'default' : 'secondary'}>
            {modelStatus?.model_loaded ? 'Loaded' : 'No model'}
          </Badge>
          {modelStatus?.model_type && <span className="text-muted-foreground">Model: {modelStatus.model_type}</span>}
          {modelStatus?.num_parameters && <span className="text-muted-foreground">Params: {(modelStatus.num_parameters / 1e3).toFixed(0)}K</span>}
          {modelStatus?.model_loaded && (
            <Button size="sm" variant="ghost" onClick={handlePpl} disabled={pplLoading} className="text-xs">{pplLoading ? '...' : 'Score'}</Button>
          )}
          {perplexity != null && <span className="text-muted-foreground">PPL: {perplexity.toFixed(1)}</span>}
          <div className="flex gap-1 ml-auto">
            {modelStatus?.model_loaded && (
              <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={async () => {
                if (checkpoints.length > 0) {
                  window.open(`/auto-train/checkpoints/${encodeURIComponent(checkpoints[0].name)}/download`)
                }
              }}>Export</Button>
            )}
            <Button size="sm" variant="ghost" onClick={refreshModel}>Refresh</Button>
          </div>
        </CardContent>
      </Card>

      {/* Quick train */}
      {!modelStatus?.model_loaded && (
        <Card>
          <CardHeader><CardTitle className="text-base">Quick Train</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            <Textarea value={initText} onChange={e => setInitText(e.target.value)} placeholder="Paste training text here..." rows={3} />
            <div className="flex gap-3 items-end">
              <div>
                <label className="text-xs text-muted-foreground block">Soul</label>
                <Input value={initSoul} onChange={e => setInitSoul(e.target.value)} className="w-28" />
              </div>
              <div>
                <label className="text-xs text-muted-foreground block">Epochs</label>
                <Input type="number" value={initEpochs} onChange={e => setInitEpochs(Number(e.target.value))} min={1} max={100} className="w-20" />
              </div>
              <div className="flex gap-1">
                {[{ label: 'Fast', e: 3 }, { label: 'Balanced', e: 10 }, { label: 'Deep', e: 30 }].map(p => (
                  <Button key={p.label} size="sm" variant={initEpochs === p.e ? 'default' : 'outline'} className="h-7 text-xs" onClick={() => setInitEpochs(p.e)}>{p.label}</Button>
                ))}
              </div>
              <Button onClick={handleInitTraining} disabled={initializing || !initText.trim()}>
                {initializing ? 'Training...' : 'Train'}
              </Button>
            </div>
            {trainProgress > 0 && (
              <div className="h-1 w-full bg-muted rounded overflow-hidden">
                <div className="h-full bg-gradient-to-r from-primary to-violet-500 rounded transition-all duration-500 animate-pulse" style={{ width: `${trainProgress}%` }} />
              </div>
            )}
            {trainCelebration && (
              <div className="flex items-center gap-2 text-sm font-medium">
                <span>{trainCelebration}</span>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Checkpoints */}
      {checkpoints.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-base">Recent Checkpoints</CardTitle></CardHeader>
          <CardContent className="space-y-1 max-h-48 overflow-y-auto">
            {checkpoints.slice(0, 10).map(ck => (
              <div key={ck.name} className="flex items-center justify-between text-xs p-1.5 rounded hover:bg-muted/30">
                <div className="truncate flex-1">
                  <span className="font-medium">{ck.name?.slice(0, 28)}</span>
                  <span className="text-muted-foreground ml-2">{ck.soul} · {ck.size_mb}MB</span>
                </div>
                <div className="flex gap-1">
                  <Button size="sm" variant="ghost" className="h-5 text-xs" onClick={async () => {
                    try {
                      await soulsController.loadCheckpoint(ck.name)
                      await refreshModel()
                    } catch {}
                  }}>Load</Button>
                  <Button size="sm" variant="ghost" className="h-5 text-xs" onClick={() => window.open(`/auto-train/checkpoints/${encodeURIComponent(ck.name)}/download`)}>↓</Button>
                </div>
              </div>
              ))}
            </CardContent>
          </Card>
        )}

      {/* Chat tab */}
      {tab === 'chat' && (
        <>
          {/* Chat log */}
          {chatLog.length > 0 && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-base">Conversation</CardTitle>
                <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={handleReset}>Clear</Button>
              </CardHeader>
              <CardContent className="space-y-2 max-h-80 overflow-y-auto">
                {chatLog.map((msg, i) => (
                  <div key={i} className={cn('flex gap-2 text-sm', msg.role === 'user' ? 'justify-end' : 'justify-start')}>
                    <div className={cn('max-w-[80%] p-2 rounded-lg', msg.role === 'user' ? 'bg-primary/10' : 'bg-muted/30')}>
                      <p className="text-xs text-muted-foreground mb-1">{msg.role === 'user' ? 'You' : 'AI'}</p>
                      <p className="whitespace-pre-wrap">{msg.text}</p>
                    </div>
                  </div>
                ))}
                <div ref={chatEndRef} />
              </CardContent>
            </Card>
          )}

          {/* Chat input */}
          <Card>
            <CardContent className="space-y-3">
              <div className="flex gap-2">
                <Input
                  value={prompt}
                  onChange={e => setPrompt(e.target.value)}
                  placeholder="Type a message..."
                  onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleChat() } }}
                />
                <Button onClick={handleChat} disabled={chatting || !prompt.trim()}>
                  {chatting ? '...' : 'Send'}
                </Button>
              </div>
              {response && (
                <div className="p-3 rounded border text-sm whitespace-pre-wrap bg-muted/20">
                  {response}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Personalization toggle */}
          <Card>
            <CardHeader><CardTitle className="text-base">Personalization</CardTitle></CardHeader>
            <CardContent className="flex items-center justify-between">
              <span className="text-sm">Use my feedback adapters</span>
              <Toggle
                checked={personalizationEnabled}
                onChange={(checked: boolean) => {
                  setPersonalizationEnabled(checked)
                  try { localStorage.setItem('labs_personalization', String(checked)) } catch {}
                }}
                label=""
              />
            </CardContent>
          </Card>
        </>
      )}

      {/* Vision */}
      {tab === 'chat' && (
        <FoldSection heading="Vision — SoulVisionCNN image captioning">
          <Card>
            <CardContent className="space-y-3">
              <input type="file" accept="image/*" onChange={e => {
                const f = e.target.files?.[0]
                if (f) { setVisionFile(f); setVisionPreview(URL.createObjectURL(f)); setVisionCaption('') }
              }} className="text-sm" />
              {visionPreview && <img src={visionPreview} alt="Vision preview" className="max-h-40 rounded border object-contain bg-muted" />}
              {visionFile && <Button size="sm" onClick={handleVision} disabled={visionBusy}>{visionBusy ? '...' : 'Describe Image'}</Button>}
              {visionCaption && <div className="p-2 bg-muted/30 rounded text-xs whitespace-pre-wrap">{visionCaption}</div>}
            </CardContent>
          </Card>
        </FoldSection>
      )}

      {/* Training history */}
      {history.length > 0 && (
        <FoldSection heading={`${history.length} past runs`}>
          <div className="mt-1 space-y-1 max-h-40 overflow-y-auto">
            {[...history].reverse().slice(0, 20).map((r: any, i: number) => (
              <div key={i} className="p-1.5 rounded bg-muted/30 flex justify-between">
                <span className="text-muted-foreground">{r.soul} · {r.epochs} epochs{r.ppl ? ` · PPL ${r.ppl}` : ''}</span>
                <span className="text-muted-foreground">{new Date(r.timestamp || r.time).toLocaleDateString()}</span>
              </div>
            ))}
          </div>
        </FoldSection>
      )}
      </div>
    </div>
  )
}
