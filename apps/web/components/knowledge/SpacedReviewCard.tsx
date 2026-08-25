'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardHeader, CardTitle, Button, Skeleton } from '@sloughgpt/strui'
import { knowledgeController, type KnowledgeItem } from '@/lib/knowledge-controller'

interface Props {
  addToast: (msg: string, type?: 'success' | 'error' | 'info') => void
}

export function SpacedReviewCard({ addToast }: Props) {
  const [dueIds, setDueIds] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [currentIndex, setCurrentIndex] = useState(0)
  const [showAnswer, setShowAnswer] = useState(false)
  const [reviewing, setReviewing] = useState(false)
  const [currentItem, setCurrentItem] = useState<KnowledgeItem | null>(null)
  const [completed, setCompleted] = useState(0)

  const fetchDue = useCallback(async () => {
    setLoading(true)
    try {
      const result = await knowledgeController.getDueReviews()
      setDueIds(result.due_ids ?? [])
      setCurrentIndex(0)
      setCompleted(0)
      setShowAnswer(false)
    } catch {
      addToast('Could not load review items', 'error')
      setDueIds([])
    } finally {
      setLoading(false)
    }
  }, [addToast])

  useEffect(() => {
    let active = true
    const load = async () => {
      setLoading(true)
      try {
        const result = await knowledgeController.getDueReviews()
        if (active) {
          setDueIds(result.due_ids ?? [])
          setCurrentIndex(0)
          setCompleted(0)
          setShowAnswer(false)
        }
      } catch {
        if (active) {
          addToast('Could not load review items', 'error')
          setDueIds([])
        }
      } finally {
        if (active) setLoading(false)
      }
    }
    void load()
    return () => { active = false }
  }, [addToast])

  useEffect(() => {
    if (dueIds.length === 0 || currentIndex >= dueIds.length) {
      setCurrentItem(null)
      return
    }
    let active = true
    const id = dueIds[currentIndex]
    knowledgeController.list(1, 0).then(items => {
      if (active) setCurrentItem(items.find(i => i.id === id) ?? null)
    }).catch(() => { if (active) setCurrentItem(null) })
    return () => { active = false }
  }, [dueIds, currentIndex])

  const handleReview = useCallback(async (performance: number) => {
    if (!currentItem) return
    setReviewing(true)
    try {
      await knowledgeController.scheduleReview(currentItem.id, performance)
      setCompleted(prev => prev + 1)
      setShowAnswer(false)
      if (currentIndex < dueIds.length - 1) {
        setCurrentIndex(prev => prev + 1)
      } else {
        addToast('All caught up!', 'success')
        void fetchDue()
      }
    } catch {
      addToast('Could not save review', 'error')
    } finally {
      setReviewing(false)
    }
  }, [currentItem, currentIndex, dueIds.length, addToast, fetchDue])

  if (loading) {
    return (
      <Card className="border-primary/20">
        <CardHeader className="pb-2">
          <Skeleton className="h-4 w-24" />
          <Skeleton className="h-1.5 w-full" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (dueIds.length === 0) return null

  const progress = dueIds.length > 0 ? (completed / dueIds.length) * 100 : 0

  return (
    <Card className="border-primary/20">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium">Quick review</CardTitle>
          <span className="text-xs text-muted-foreground">{completed} of {dueIds.length}</span>
        </div>
        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-300 rounded-full"
            style={{ width: `${progress}%` }}
          />
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {currentItem ? (
          <>
            <div className="rounded-lg bg-muted/30 p-4">
              <p className="text-sm leading-relaxed">{currentItem.content}</p>
              {currentItem.topic && (
                <p className="text-xs text-muted-foreground mt-2">{currentItem.topic}</p>
              )}
            </div>

            {showAnswer ? (
              <div className="space-y-2">
                <p className="text-xs text-muted-foreground">Did you remember this?</p>
                <div className="flex gap-2">
                  <Button size="sm" variant="outline" onClick={() => void handleReview(0.2)} disabled={reviewing} className="flex-1">
                    No
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => void handleReview(0.5)} disabled={reviewing} className="flex-1">
                    Kind of
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => void handleReview(0.8)} disabled={reviewing} className="flex-1">
                    Yes
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => void handleReview(1.0)} disabled={reviewing} className="flex-1">
                    Easily
                  </Button>
                </div>
              </div>
            ) : (
              <Button size="sm" onClick={() => setShowAnswer(true)} className="w-full">
                Show answer
              </Button>
            )}
          </>
        ) : (
          <div className="text-center py-4">
            <p className="text-sm font-medium text-success">All done!</p>
            <p className="text-xs text-muted-foreground mt-1">Nothing to review right now.</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
