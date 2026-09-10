'use client'

import { useMemo } from 'react'
import { Card, CardHeader, CardTitle, CardContent, Badge } from '@sloughgpt/strui'
import { usePhonemeStore } from '@/lib/phoneme-store'

function getDayKey(date: Date): string {
  return date.toISOString().split('T')[0]
}

function getDayOfWeek(date: Date): number {
  return date.getDay()
}

function getColorClass(count: number): string {
  if (count === 0) return 'bg-muted/30'
  if (count <= 3) return 'bg-green-200'
  if (count <= 6) return 'bg-green-400'
  if (count <= 10) return 'bg-green-500'
  return 'bg-green-700'
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

export default function PracticeCalendar() {
  const history = usePhonemeStore(s => s.history)

  const { calendarData, totalDays, totalWords, currentStreak, longestStreak } = useMemo(() => {
    const dayCounts: Record<string, number> = {}
    history.forEach(entry => {
      if (entry.timestamp) {
        const day = entry.timestamp.split('T')[0]
        dayCounts[day] = (dayCounts[day] || 0) + 1
      }
    })

    const today = new Date()
    const weeks: { date: Date; count: number }[][] = []
    let currentWeek: { date: Date; count: number }[] = []

    for (let i = 364; i >= 0; i--) {
      const date = new Date(today)
      date.setDate(date.getDate() - i)
      const dayOfWeek = getDayOfWeek(date)

      if (dayOfWeek === 0 && currentWeek.length > 0) {
        weeks.push(currentWeek)
        currentWeek = []
      }

      const key = getDayKey(date)
      currentWeek.push({ date, count: dayCounts[key] || 0 })
    }
    if (currentWeek.length > 0) weeks.push(currentWeek)

    const totalDays = Object.keys(dayCounts).length
    const totalWords = Object.values(dayCounts).reduce((a, b) => a + b, 0)

    let currentStreak = 0
    let longestStreak = 0
    let tempStreak = 0
    const sortedDays = Object.keys(dayCounts).sort()

    for (let i = 0; i < sortedDays.length; i++) {
      if (i === 0) {
        tempStreak = 1
      } else {
        const prev = new Date(sortedDays[i - 1])
        const curr = new Date(sortedDays[i])
        const diff = (curr.getTime() - prev.getTime()) / (1000 * 60 * 60 * 24)
        if (diff === 1) {
          tempStreak++
        } else {
          longestStreak = Math.max(longestStreak, tempStreak)
          tempStreak = 1
        }
      }
    }
    longestStreak = Math.max(longestStreak, tempStreak)

    const todayKey = getDayKey(today)
    if (dayCounts[todayKey]) {
      let streak = 0
      let checkDate = new Date(today)
      while (dayCounts[getDayKey(checkDate)]) {
        streak++
        checkDate.setDate(checkDate.getDate() - 1)
      }
      currentStreak = streak
    }

    return { calendarData: weeks, totalDays, totalWords, currentStreak, longestStreak }
  }, [history])

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Practice Calendar</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-center py-4">
            Start practicing to see your activity calendar.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>Practice Calendar</span>
          <div className="flex items-center gap-2">
            {currentStreak > 0 && <Badge variant="default">{currentStreak} day streak</Badge>}
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{totalDays}</p>
            <p className="text-xs text-muted-foreground">Active Days</p>
          </div>
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{totalWords}</p>
            <p className="text-xs text-muted-foreground">Total Words</p>
          </div>
          <div className="p-2 rounded-lg bg-muted/30">
            <p className="text-lg font-bold">{longestStreak}</p>
            <p className="text-xs text-muted-foreground">Best Streak</p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <div className="flex gap-0.5 min-w-[700px]">
            {calendarData.map((week, wi) => (
              <div key={wi} className="flex flex-col gap-0.5">
                {week.map((day, di) => (
                  <div
                    key={di}
                    className={`w-3 h-3 rounded-sm ${getColorClass(day.count)}`}
                    title={`${day.date.toLocaleDateString()}: ${day.count} words`}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-muted-foreground justify-end">
          <span>Less</span>
          <div className="flex gap-0.5">
            <div className="w-3 h-3 rounded-sm bg-muted/30" />
            <div className="w-3 h-3 rounded-sm bg-green-200" />
            <div className="w-3 h-3 rounded-sm bg-green-400" />
            <div className="w-3 h-3 rounded-sm bg-green-500" />
            <div className="w-3 h-3 rounded-sm bg-green-700" />
          </div>
          <span>More</span>
        </div>
      </CardContent>
    </Card>
  )
}
