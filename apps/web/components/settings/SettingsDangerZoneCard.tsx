'use client'

import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Button } from '@sloughgpt/strui'

interface SettingsDangerZoneCardProps {
  onClearChat: () => void
  onResetSettings: () => void
}

export function SettingsDangerZoneCard({ onClearChat, onResetSettings }: SettingsDangerZoneCardProps) {
  return (
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
            <p className="text-xs text-muted-foreground">Removes all saved conversations from this browser</p>
          </div>
          <Button type="button" variant="destructive" size="sm" onClick={onClearChat}>Clear</Button>
        </div>
        <div className="border-t border-border/30" />
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Reset all settings</p>
            <p className="text-xs text-muted-foreground">Restore theme, model defaults, and custom instructions to defaults</p>
          </div>
          <Button type="button" variant="destructive" size="sm" onClick={onResetSettings}>Reset</Button>
        </div>
      </CardContent>
      <CardFooter className="justify-end" />
    </Card>
  )
}
