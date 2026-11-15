'use client'

/** Core identity: values, goals, interests and avoid lists. */
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { PersonalityWorkbench } from './usePersonalityWorkbench'

export function PersonalityIdentity({ wb }: { wb: PersonalityWorkbench }) {
  const {
    avoidInput,
    goalsInput,
    interestsInput,
    setAvoidInput,
    setGoalsInput,
    setInterestsInput,
    setValuesInput,
    valuesInput,
  } = wb

  return (
    <>
      {/* Core Values & Goals */}
      <Card>
        <CardHeader>
          <CardTitle>Core Identity</CardTitle>
          <CardDescription>What matters most and what the system strives toward</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3 sm:space-y-4">
          <div>
            <label className="text-xs sm:text-sm font-medium">Values (comma-separated)</label>
            <input
              type="text"
              value={valuesInput}
              onChange={(e) => setValuesInput(e.target.value)}
              className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
              placeholder="helpfulness, honesty, curiosity"
            />
          </div>
          <div>
            <label className="text-xs sm:text-sm font-medium">Goals (comma-separated)</label>
            <input
              type="text"
              value={goalsInput}
              onChange={(e) => setGoalsInput(e.target.value)}
              className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
              placeholder="Provide accurate responses, Learn from interactions"
            />
          </div>
          <div>
            <label className="text-xs sm:text-sm font-medium">Interests (comma-separated)</label>
            <input
              type="text"
              value={interestsInput}
              onChange={(e) => setInterestsInput(e.target.value)}
              className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
              placeholder="AI, programming, science"
            />
          </div>
          <div>
            <label className="text-xs sm:text-sm font-medium">Avoid (comma-separated)</label>
            <input
              type="text"
              value={avoidInput}
              onChange={(e) => setAvoidInput(e.target.value)}
              className="mt-1 w-full rounded-md border bg-background px-2.5 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm"
              placeholder="being condescending, making things up"
            />
          </div>
        </CardContent>
      </Card>
    </>
  )
}
