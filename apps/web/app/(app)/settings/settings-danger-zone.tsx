'use client'

/** Danger zone: clear chat history and reset all settings. */
import { VersionBadge, WEB_VERSION } from './settings-ui'
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
  CardFooter,
  CardHeader,
  CardTitle,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsDangerZone({ wb }: { wb: SettingsWorkbench }) {
  const { clearChat, resetAllSettings } = wb

  return (
    <>
      {/* Danger zone */}
      <Card className="border-destructive/30">
        <CardHeader>
          <div>
            <CardTitle className="text-base text-destructive">Danger zone</CardTitle>
            <CardDescription>Irreversible actions</CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Clear chat history</p>
              <p className="text-xs text-muted-foreground">
                Removes all saved conversations from this browser
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm">
                  Clear
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Clear all chat history?</AlertDialogTitle>
                  <AlertDialogDescription>This cannot be undone.</AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={clearChat}
                    className="bg-destructive text-destructive-foreground"
                  >
                    Clear
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
          <div className="border-t border-border/30" />
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">Reset all settings</p>
              <p className="text-xs text-muted-foreground">
                Restore theme, model defaults, and custom instructions to defaults
              </p>
            </div>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm">
                  Reset
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Reset all settings?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will clear your theme preference, default model settings, and custom
                    instructions. Chat history is not affected.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={resetAllSettings}
                    className="bg-destructive text-destructive-foreground"
                  >
                    Reset
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>
    </>
  )
}
