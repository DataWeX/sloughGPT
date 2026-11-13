'use client'

/** Quick presets and the trait-conflict warnings. */
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@sloughgpt/strui'
import type { PersonalityWorkbench } from './usePersonalityWorkbench'

export function PersonalityPresets({ wb }: { wb: PersonalityWorkbench }) {
  const {
    PRESET_DESCRIPTIONS,
    activePreset,
    applyingPreset,
    conflicts,
    handleApplyPreset,
    presetNames,
  } = wb

  return (
    <>
      {/* Presets */}
      {presetNames.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Quick Presets</CardTitle>
            <CardDescription>Apply a pre-configured personality profile</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {presetNames.map((name) => (
                <Button
                  key={name}
                  size="sm"
                  variant={activePreset === name ? 'default' : 'outline'}
                  onClick={() => handleApplyPreset(name)}
                  disabled={applyingPreset !== null}
                  className="flex flex-col items-start h-auto py-2"
                >
                  <span className="capitalize">{name}</span>
                  <span className="text-[10px] text-muted-foreground font-normal">
                    {PRESET_DESCRIPTIONS[name] || ''}
                  </span>
                </Button>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Conflict Warnings */}
      {conflicts.length > 0 && (
        <Card className="border-warning/50">
          <CardHeader>
            <CardTitle className="text-warning dark:text-warning">Personality Conflicts</CardTitle>
            <CardDescription>These settings may work against each other</CardDescription>
          </CardHeader>
          <CardContent className="space-y-2">
            {conflicts.map((c, i) => (
              <div key={i} className="flex items-start gap-2 text-sm p-2 rounded-md bg-warning/5">
                <Badge
                  variant={c.severity === 'medium' ? 'destructive' : 'outline'}
                  className="text-[10px] mt-0.5"
                >
                  {c.severity}
                </Badge>
                <div>
                  <div>{c.message}</div>
                  <div className="text-[10px] text-muted-foreground">{c.fields.join(' + ')}</div>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </>
  )
}
