'use client'

import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { knowledgeController } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'

export function KnowledgeOperationsCard() {
  const [searchQuery, setSearchQuery] = useState('')
  const [searchPath, setSearchPath] = useState('.')
  const [searchResults, setSearchResults] = useState<Array<{ path: string; line: number; snippet: string; score: number }>>([])
  const [searching, setSearching] = useState(false)

  const [dedupContent, setDedupContent] = useState('')
  const [dedupResult, setDedupResult] = useState<{ is_duplicate: boolean; best_match: string | null; score: number } | null>(null)
  const [checkingDup, setCheckingDup] = useState(false)

  const [gapData, setGapData] = useState<{ gaps: Array<{ topic: string; suggestion: string }>; total_facts: number } | null>(null)
  const [loadingGaps, setLoadingGaps] = useState(false)

  const addToast = useToastStore(s => s.addToast)

  const handleSearch = async () => {
    if (!searchQuery.trim()) return
    setSearching(true)
    try {
      const res = await knowledgeController.searchFiles(searchQuery, searchPath)
      setSearchResults(res.results)
    } catch { addToast('Search failed', 'error') }
    setSearching(false)
  }

  const handleCheckDup = async () => {
    if (!dedupContent.trim()) return
    setCheckingDup(true)
    try {
      const res = await knowledgeController.checkDuplicate(dedupContent)
      setDedupResult(res)
    } catch { addToast('Duplicate check failed', 'error') }
    setCheckingDup(false)
  }

  const handleFindGaps = async () => {
    setLoadingGaps(true)
    try {
      const res = await knowledgeController.gaps()
      setGapData(res)
    } catch { addToast('Gap analysis failed', 'error') }
    setLoadingGaps(false)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Semantic Tools</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* File search */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">Search codebase</label>
          <div className="flex items-center gap-2">
            <input
              className="flex-1 h-8 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-1 focus:ring-primary/30"
              placeholder="e.g. how does embedding work"
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSearch()}
              aria-label="File search query"
            />
            <input
              className="h-8 rounded-md border border-border bg-background px-2 text-xs w-28 focus:outline-none focus:ring-1 focus:ring-primary/30"
              placeholder="path (.)"
              value={searchPath}
              onChange={e => setSearchPath(e.target.value)}
              aria-label="Search path"
            />
            <Button size="sm" onClick={handleSearch} disabled={searching || !searchQuery.trim()}>
              {searching ? 'Searching\u2026' : 'Search'}
            </Button>
          </div>
          {searchResults.length > 0 && (
            <div className="space-y-1.5 mt-2 max-h-48 overflow-y-auto">
              {searchResults.map((r, i) => (
                <div key={i} className="text-xs rounded bg-muted/40 px-2 py-1.5 flex items-center gap-2">
                  <span className="font-mono text-muted-foreground shrink-0">{r.score.toFixed(3)}</span>
                  <span className="font-mono text-muted-foreground shrink-0">{r.path}:{r.line}</span>
                  <span className="truncate">{r.snippet.replace(/\n/g, ' ')}</span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="h-px bg-border/30" />

        {/* Duplicate check */}
        <div className="space-y-2">
          <label className="text-xs font-medium text-muted-foreground">Check for duplicates</label>
          <div className="flex items-center gap-2">
            <input
              className="flex-1 h-8 rounded-md border border-border bg-background px-3 text-sm focus:outline-none focus:ring-1 focus:ring-primary/30"
              placeholder={'Paste content to check\u2026'}
              value={dedupContent}
              onChange={e => setDedupContent(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleCheckDup()}
              aria-label="Duplicate check content"
            />
            <Button size="sm" variant="outline" onClick={handleCheckDup} disabled={checkingDup || !dedupContent.trim()}>
              {checkingDup ? 'Checking\u2026' : 'Check'}
            </Button>
          </div>
          {dedupResult && (
            <div className={`text-xs rounded px-2 py-1.5 ${dedupResult.is_duplicate ? 'bg-destructive/10 text-destructive' : 'bg-green/10 text-green-600'}`}>
              {dedupResult.is_duplicate
                ? `Duplicate (score: ${dedupResult.score.toFixed(3)}) \u2014 existing: "${dedupResult.best_match?.slice(0, 80)}\u2026"`
                : `Unique (best match: ${dedupResult.score.toFixed(3)})`
              }
            </div>
          )}
        </div>

        <div className="h-px bg-border/30" />

        {/* Knowledge gaps */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-muted-foreground">Knowledge gaps</label>
            <Button size="sm" variant="ghost" onClick={handleFindGaps} disabled={loadingGaps}>
              {loadingGaps ? 'Analyzing\u2026' : 'Analyze'}
            </Button>
          </div>
          {gapData && (
            <div className="space-y-1">
              {gapData.gaps.length === 0 ? (
                <p className="text-xs text-muted-foreground">{'No significant gaps \u2014 '}{gapData.total_facts} facts across all topics</p>
              ) : (
                gapData.gaps.map((g, i) => (
                  <div key={i} className="text-xs rounded bg-muted/40 px-2 py-1.5">
                    <span className="font-medium">{g.topic}:</span> <span className="text-muted-foreground">{g.suggestion}</span>
                  </div>
                ))
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
