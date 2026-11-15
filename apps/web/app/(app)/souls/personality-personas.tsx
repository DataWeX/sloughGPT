'use client'

/** Saved personas card with activate/delete, plus the save-persona dialog. */
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  IconHeart,
  Input,
} from '@sloughgpt/strui'
import type { PersonalityWorkbench } from './usePersonalityWorkbench'

export function PersonalityPersonas({ wb }: { wb: PersonalityWorkbench }) {
  const {
    activatingPersona,
    deletingPersona,
    handleActivatePersona,
    handleDeletePersona,
    handleSavePersona,
    personaNameInput,
    profile,
    savedPersonas,
    savingPersona,
    setPersonaNameInput,
    setShowSavePersonaDialog,
    showSavePersonaDialog,
    t,
  } = wb

  return (
    <>
      {/* Saved Personas */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle>{t('personality.savedPersonas')}</CardTitle>
            <CardDescription>Manage saved personality profiles</CardDescription>
          </div>
          <Button size="sm" onClick={() => setShowSavePersonaDialog(true)} disabled={!profile}>
            {t('personality.saveCurrent')}
          </Button>
        </CardHeader>
        <CardContent>
          {savedPersonas.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t('personality.noPersonas')}</p>
          ) : (
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {savedPersonas.map((persona) => (
                <div
                  key={persona.id}
                  className={`rounded-lg border p-3 sm:p-4 space-y-2 ${
                    persona.active ? 'border-primary/50 bg-primary/5' : ''
                  }`}
                  data-active={persona.active ? 'true' : 'false'}
                >
                  <div className="flex items-center gap-2">
                    {/* Rose-blue traits icon — identity marker for a persona */}
                    <span
                      aria-hidden="true"
                      className="w-7 h-7 shrink-0 rounded-full bg-gradient-to-br from-[rgb(var(--chart-5)/0.25)] to-[rgb(var(--chart-4)/0.25)] border border-[rgb(var(--chart-4)/0.4)] flex items-center justify-center"
                    >
                      <IconHeart className="w-3.5 h-3.5 text-[rgb(var(--chart-4))]" filled />
                    </span>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span className="text-sm sm:text-base font-medium truncate">
                          {persona.name}
                        </span>
                        {persona.active && (
                          <Badge variant="outline" className="text-[10px] text-primary shrink-0">
                            Active
                          </Badge>
                        )}
                      </div>
                      <div className="text-[10px] sm:text-xs text-muted-foreground line-clamp-2">
                        {persona.values?.join(', ') || '—'}
                      </div>
                    </div>
                  </div>
                  {/* Trait preview — what this persona actually feels like */}
                  {(persona.top_traits?.length ?? 0) > 0 && (
                    <div className="space-y-1" aria-label={`${persona.name} defining traits`}>
                      {persona.top_traits!.map(({ key, value }) => (
                        <div key={key} className="flex items-center gap-2">
                          <span className="text-[10px] text-muted-foreground capitalize w-24 shrink-0 truncate">
                            {key.replace(/_/g, ' ')}
                          </span>
                          <div className="flex-1 h-1 rounded-full bg-muted/50 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-gradient-to-r from-[rgb(var(--chart-5))] to-[rgb(var(--chart-4))]"
                              style={{ width: `${Math.round(value * 100)}%` }}
                            />
                          </div>
                          <span className="text-[10px] font-mono text-muted-foreground w-8 text-right shrink-0">
                            {Math.round(value * 100)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div className="flex gap-2 pt-1">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleActivatePersona(persona.id)}
                      disabled={persona.active || activatingPersona === persona.id}
                    >
                      {activatingPersona === persona.id
                        ? '...'
                        : persona.active
                          ? 'Active'
                          : t('personality.activate')}
                    </Button>
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => {
                        if (window.confirm(`Delete persona "${persona.name}"?`)) {
                          handleDeletePersona(persona.id)
                        }
                      }}
                      disabled={deletingPersona === persona.id}
                    >
                      {deletingPersona === persona.id ? '...' : t('personality.delete')}
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Save Persona Dialog */}
      {showSavePersonaDialog && (
        <Card className="border-primary/50">
          <CardHeader>
            <CardTitle>{t('personality.savePersona')}</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 sm:space-y-3">
            <div>
              <label className="text-xs sm:text-sm font-medium">
                {t('personality.personaName')}
              </label>
              <Input
                value={personaNameInput}
                onChange={(e) => setPersonaNameInput(e.target.value)}
                placeholder="My Persona"
                className="mt-1"
              />
            </div>
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={handleSavePersona}
                disabled={savingPersona || !personaNameInput.trim()}
              >
                {savingPersona ? '...' : t('personality.savePersona')}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setShowSavePersonaDialog(false)
                  setPersonaNameInput('')
                }}
              >
                Cancel
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </>
  )
}
