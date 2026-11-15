'use client'

/** Data tab: seed/clear mutations with confirmation dialogs. */
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  TabsContent,
} from '@sloughgpt/strui'
import type { AllSettingsWorkbench } from './useAllSettingsWorkbench'

export function AllSettingsDataTab({ wb }: { wb: AllSettingsWorkbench }) {
  const {
    handleClearBeliefs,
    handleClearEpisodes,
    handleResetPersonality,
    handleSeed,
    seedCount,
    seeding,
    setSeedCount,
    t,
  } = wb

  return (
    <TabsContent value="data" className="mt-6 space-y-6">
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.seed_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.seed_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            <Input
              type="number"
              min={1}
              max={100}
              value={seedCount}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setSeedCount(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))
              }
              className="w-20 h-8 text-xs text-right"
              aria-label="Seed count"
            />
            <Button size="sm" variant="secondary" onClick={handleSeed} disabled={seeding}>
              {seeding
                ? t('consciousness_all_settings.seeding')
                : t('consciousness_all_settings.seed_button')}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">
              {t('consciousness_all_settings.clear_title')}
            </CardTitle>
            <CardDescription>{t('consciousness_all_settings.clear_desc')}</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.clear_episodes_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.clear_episodes_desc')}
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm">
                  {t('consciousness_all_settings.clear_button')}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    {t('consciousness_all_settings.clear_episodes_confirm_title')}
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    {t('consciousness_all_settings.clear_episodes_confirm_desc')}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>{t('consciousness_all_settings.cancel')}</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleClearEpisodes}
                    className="bg-destructive text-destructive-foreground"
                  >
                    {t('consciousness_all_settings.clear_button')}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.clear_beliefs_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.clear_beliefs_desc')}
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm">
                  {t('consciousness_all_settings.clear_button')}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    {t('consciousness_all_settings.clear_beliefs_confirm_title')}
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    {t('consciousness_all_settings.clear_beliefs_confirm_desc')}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>{t('consciousness_all_settings.cancel')}</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleClearBeliefs}
                    className="bg-destructive text-destructive-foreground"
                  >
                    {t('consciousness_all_settings.clear_button')}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">
                {t('consciousness_all_settings.reset_personality_label')}
              </p>
              <p className="text-xs text-muted-foreground">
                {t('consciousness_all_settings.reset_personality_desc')}
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm">
                  {t('consciousness_all_settings.reset_button')}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>
                    {t('consciousness_all_settings.reset_personality_confirm_title')}
                  </AlertDialogTitle>
                  <AlertDialogDescription>
                    {t('consciousness_all_settings.reset_personality_confirm_desc')}
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>{t('consciousness_all_settings.cancel')}</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleResetPersonality}
                    className="bg-destructive text-destructive-foreground"
                  >
                    {t('consciousness_all_settings.reset_button')}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </CardContent>
      </Card>
    </TabsContent>
  )
}
