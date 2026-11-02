'use client'

/** Voice characteristics, Big Five traits and communication style. */
import { STYLE_LABELS, TRAIT_LABELS, VOICE_LABELS } from './personality-model'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import type { PersonalityWorkbench } from './usePersonalityWorkbench'

export function PersonalityVoiceStyle({ wb }: { wb: PersonalityWorkbench }) {
  const { handleStyleToggle, handleTraitChange, handleVoiceChange, profile } = wb

  // The page guards on a missing profile before rendering sections.
  if (!profile) return null

  return (
    <>
      {/* Voice Characteristics */}
      <Card>
        <CardHeader>
          <CardTitle>Voice</CardTitle>
          <CardDescription>How the system sounds when communicating</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 sm:space-y-3">
          {Object.entries(profile.voice).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 sm:gap-4">
              <label className="text-xs sm:text-sm w-24 sm:w-32 shrink-0">
                {VOICE_LABELS[key] || key}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={value}
                onChange={(e) => handleVoiceChange(key, parseFloat(e.target.value))}
                className="flex-1 min-w-0"
              />
              <span className="text-[10px] sm:text-xs text-muted-foreground w-8 sm:w-10 text-right shrink-0">
                {(value * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Personality Traits */}
      <Card>
        <CardHeader>
          <CardTitle>Traits</CardTitle>
          <CardDescription>Big Five personality dimensions</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 sm:space-y-3">
          {Object.entries(profile.traits).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 sm:gap-4">
              <label className="text-xs sm:text-sm w-24 sm:w-32 shrink-0">
                {TRAIT_LABELS[key] || key}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={value}
                onChange={(e) => handleTraitChange(key, parseFloat(e.target.value))}
                className="flex-1 min-w-0"
              />
              <span className="text-[10px] sm:text-xs text-muted-foreground w-8 sm:w-10 text-right shrink-0">
                {(value * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Communication Style */}
      <Card>
        <CardHeader>
          <CardTitle>Communication Style</CardTitle>
          <CardDescription>Preferences for how to respond</CardDescription>
        </CardHeader>
        <CardContent className="space-y-1.5 sm:space-y-2">
          {Object.entries(profile.style).map(([key, value]) => (
            <div key={key} className="flex items-center gap-2 sm:gap-3">
              <button
                onClick={() => handleStyleToggle(key)}
                className={`w-10 h-5 rounded-full transition-colors ${value ? 'bg-primary' : 'bg-muted'}`}
              >
                <div
                  className={`w-4 h-4 rounded-full bg-white transition-transform ${value ? 'translate-x-5' : 'translate-x-0.5'}`}
                />
              </button>
              <label className="text-sm">{STYLE_LABELS[key] || key}</label>
            </div>
          ))}
        </CardContent>
      </Card>
    </>
  )
}
