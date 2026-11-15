'use client'

/** Import comparison review and the personality evolution history. */
import { PersonalityTrendChart } from './PersonalityTrendChart'
import { TRAIT_LABELS, VOICE_LABELS } from './personality-model'
import { Button, Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { PersonalityWorkbench } from './usePersonalityWorkbench'

export function PersonalityEvolution({ wb }: { wb: PersonalityWorkbench }) {
  const {
    originalProfile,
    personalityHistory,
    profile,
    setProfile,
    setShowComparison,
    showComparison,
  } = wb

  return (
    <>
      {/* Import Comparison */}
      {showComparison && originalProfile && profile && (
        <Card className="border-info/50">
          <CardHeader>
            <CardTitle className="text-info dark:text-info">Imported Profile — Review</CardTitle>
            <CardDescription>Compare imported values with your current profile</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-2 sm:gap-4 text-xs sm:text-sm">
              <div className="font-medium text-muted-foreground">Setting</div>
              <div className="font-medium text-muted-foreground">Current</div>
              <div className="font-medium text-muted-foreground">Imported</div>
              {Object.keys(originalProfile.voice).map((key) => (
                <>
                  <div key={`label-${key}`} className="capitalize">
                    {key}
                  </div>
                  <div key={`old-${key}`}>{(originalProfile.voice[key] * 100).toFixed(0)}%</div>
                  <div
                    key={`new-${key}`}
                    className={
                      profile.voice[key] !== originalProfile.voice[key]
                        ? 'text-info font-medium'
                        : ''
                    }
                  >
                    {(profile.voice[key] * 100).toFixed(0)}%
                  </div>
                </>
              ))}
            </div>
            <div className="flex gap-2 mt-4">
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setShowComparison(false)
                  setProfile(originalProfile)
                }}
              >
                Discard
              </Button>
              <Button size="sm" onClick={() => setShowComparison(false)}>
                Keep Imported
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Personality Evolution */}
      {personalityHistory.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle>Personality Evolution</CardTitle>
            <CardDescription>
              How your voice and traits have changed over interactions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <PersonalityTrendChart
                title="Voice Over Time"
                history={personalityHistory}
                field="voice"
                keys={['warmth', 'confidence', 'humor', 'empathy']}
                labels={VOICE_LABELS}
              />
              <PersonalityTrendChart
                title="Traits Over Time"
                history={personalityHistory}
                field="traits"
                keys={['openness', 'agreeableness', 'conscientiousness']}
                labels={TRAIT_LABELS}
              />
            </div>
          </CardContent>
        </Card>
      )}
    </>
  )
}
