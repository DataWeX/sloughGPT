import { Card, CardHeader, CardTitle, CardDescription, CardContent, Input } from '@sloughgpt/strui'

export interface PersonalityCoreIdentityProps {
  values: string
  goals: string
  interests: string
  avoid: string
  onValuesChange: (value: string) => void
  onGoalsChange: (value: string) => void
  onInterestsChange: (value: string) => void
  onAvoidChange: (value: string) => void
}

export function PersonalityCoreIdentity({
  values, goals, interests, avoid,
  onValuesChange, onGoalsChange, onInterestsChange, onAvoidChange,
}: PersonalityCoreIdentityProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Core Identity</CardTitle>
        <CardDescription>What matters most and what the system strives toward</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3 sm:space-y-4">
        <div>
          <label className="text-xs sm:text-sm font-medium">Values (comma-separated)</label>
          <Input
            value={values}
            onChange={(e) => onValuesChange(e.target.value)}
            className="mt-1"
            placeholder="helpfulness, honesty, curiosity"
          />
        </div>
        <div>
          <label className="text-xs sm:text-sm font-medium">Goals (comma-separated)</label>
          <Input
            value={goals}
            onChange={(e) => onGoalsChange(e.target.value)}
            className="mt-1"
            placeholder="Provide accurate responses, Learn from interactions"
          />
        </div>
        <div>
          <label className="text-xs sm:text-sm font-medium">Interests (comma-separated)</label>
          <Input
            value={interests}
            onChange={(e) => onInterestsChange(e.target.value)}
            className="mt-1"
            placeholder="AI, programming, science"
          />
        </div>
        <div>
          <label className="text-xs sm:text-sm font-medium">Avoid (comma-separated)</label>
          <Input
            value={avoid}
            onChange={(e) => onAvoidChange(e.target.value)}
            className="mt-1"
            placeholder="being condescending, making things up"
          />
        </div>
      </CardContent>
    </Card>
  )
}
