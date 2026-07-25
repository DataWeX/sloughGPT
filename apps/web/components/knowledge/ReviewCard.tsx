'use client'

import { useEffect, useState, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { knowledgeController, type KnowledgeItem } from '@/lib/knowledge-controller'
import { useToastStore } from '@/lib/toast-store'

const PERF_LABELS: Array<{ value: number; label: string; color: string }> = [
  { value: 0.0, label: 'Again', color: 'text-red-500' },
  { value: 0.3, label: 'Hard', color: 'text-orange-500' },
  { value: 0.6, label: 'Good', color: 'text-yellow-500' },
  { value: 1.0, label: 'Easy', color: 'text-green-500' },
]

export function ReviewCard() {
  const [dueIds, setDueIds] = useState<string[]>([])
  const [stats, setStats] = useState<{ due_count: number; total_scheduled: number } | null>(null)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [currentItem, setCurrentItem] = useState<KnowledgeItem | null>(null)
  const [showAnswer, setShowAnswer] = useState(false)
  const [scheduling, setScheduling] = useState(false)
  const addToast = useToastStore(s => s.addToast)

  const fetchDue = useCallback(async () => {
    try {
      const res = await knowledgeController.getDueReviews()
      setDueIds(res.due_ids)
      setStats(res.stats)
      setCurrentIndex(0)
      setShowAnswer(false)
    } catch { /* no reviews endpoint available */ }
  }, [])

  useEffect(() => { fetchDue() }, [fetchDue])

  useEffect(() => {
    if (dueIds.length === 0 || currentIndex >= dueIds.length) {
      setCurrentItem(null)
      return
    }
    const id = dueIds[currentIndex]
    knowledgeController.list(1, 0).then(items => {
      const found = items.find(i => i.id === id)
      if (found) setCurrentItem(found)
    }).catch(() => {})
  }, [dueIds, currentIndex])

  const handleRate = async (performance: number) => {
    if (!currentItem) return
    setScheduling(true)
    try {
      await knowledgeController.scheduleReview(currentItem.id, performance)
      const next = currentIndex + 1
      if (next >= dueIds.length) {
        addToast('All reviews completed', 'success')
        await fetchDue()
      } else {
        setCurrentIndex(next)
        setShowAnswer(false)
      }
    } catch { addToast('Failed to schedule review', 'error') }
    setScheduling(false)
  }

  if (dueIds.length === 0 || stats === null) return null

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Spaced Repetition</CardTitle>
        <div className="flex items-center gap-2">
          <Badge variant="default" label={`${stats.due_count} due`} />
          {stats.total_scheduled > 0 && (
            <span className="text-xs text-muted-foreground">{stats.total_scheduled} total</span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {currentItem ? (
          <>
            <div className="text-sm leading-relaxed p-3 rounded-md bg-muted/40 border border-border/40">
              {currentItem.content}
            </div>
            {currentItem.topic && (
              <div className="text-xs text-muted-foreground">
                Topic: <span className="font-medium text-foreground">{currentItem.topic}</span>
              </div>
            )}
            {!showAnswer ? (
              <Button size="sm" variant="outline" onClick={() => setShowAnswer(true)} className="w-full">
                Show Answer
              </Button>
            ) : (
              <div className="flex items-center gap-2">
                {PERF_LABELS.map(p => (
                  <Button
                    key={p.value}
                    size="sm"
                    variant="outline"
                    disabled={scheduling}
                    onClick={() => handleRate(p.value)}
                    className={`flex-1 ${p.color}`}
                  >
                    {p.label}
                  </Button>
                ))}
              </div>
            )}
            <div className="text-xs text-muted-foreground text-center">
              {currentIndex + 1} / {dueIds.length}
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground text-center py-2">
            No reviews due right now. Check back later.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
