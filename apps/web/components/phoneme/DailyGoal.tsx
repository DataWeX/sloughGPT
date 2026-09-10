'use client'

import { useState, useEffect, useMemo, useCallback } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge, Button } from '@sloughgpt/strui'
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'

const STORAGE_KEY = 'sloughgpt-phoneme-daily-goal'
const GOALS = [5, 10, 15, 20, 25, 30]

interface GoalState {
  date: string
  target: number
  completed: number
}

function getToday(): string {
  return new Date().toISOString().split('T')[0]
}

function loadGoal(): GoalState {
  if (typeof window === 'undefined') return { date: getToday(), target: 10, completed: 0 }
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}')
    const today = getToday()
    if (stored.date === today) return stored
    return { date: today, target: stored.target || 10, completed: 0 }
  } catch {
    return { date: getToday(), target: 10, completed: 0 }
  }
}

function saveGoal(state: GoalState) {
  if (typeof window === 'undefined') return
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)) } catch {}
}

export default function DailyGoal() {
  const [goal, setGoal] = useState<GoalState>(loadGoal)
  const history = usePhonemeStore(s => s.history)

  const todayCount = useMemo(() => {
    const today = getToday()
    return history.filter(e => e.timestamp?.startsWith(today)).length
  }, [history])

  useEffect(() => {
    const updated = { ...goal, completed: todayCount }
    if (updated.completed !== goal.completed) {
      setGoal(updated)
      saveGoal(updated)
    }
  }, [todayCount])

  const updateTarget = useCallback((target: number) => {
    const updated = { ...goal, target }
    setGoal(updated)
    saveGoal(updated)
  }, [goal])

  const progress = Math.min(100, (goal.completed / goal.target) * 100)
  const isComplete = goal.completed >= goal.target

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Daily Goal</span>
          {isComplete ? (
            <Badge variant="default">Complete!</Badge>
          ) : (
            <Badge variant="outline">{goal.completed}/{goal.target}</Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Target:</span>
          <Select value={String(goal.target)} onValueChange={v => updateTarget(Number(v))}>
            <SelectTrigger className="w-[80px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {GOALS.map(g => (
                <SelectItem key={g} value={String(g)}>{g} words</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Today&apos;s progress</span>
            <span className="font-medium">{goal.completed} / {goal.target} words</span>
          </div>
          <div className="h-3 rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isComplete ? 'bg-green-500' : 'bg-primary'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground text-right">
            {progress.toFixed(0)}% complete
          </p>
        </div>

        {isComplete && (
          <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/20 text-sm text-green-700">
            Great job! You&apos;ve reached your daily goal! Keep practicing or set a higher target.
          </div>
        )}

        {!isComplete && goal.completed > 0 && (
          <p className="text-xs text-muted-foreground text-center">
            {goal.target - goal.completed} more words to reach your goal
          </p>
        )}
      </CardContent>
    </Card>
  )
}
