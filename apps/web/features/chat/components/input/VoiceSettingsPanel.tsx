'use client'

import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem, Slider, Switch } from '@sloughgpt/strui'
import type { VoiceSettings } from '@/features/chat/hooks/useVoiceChat'

interface VoiceSettingsPanelProps {
  settings: VoiceSettings
  availableVoices: Array<{ name: string; lang: string }>
  updateSettings: (s: Partial<VoiceSettings>) => void
}

export function VoiceSettingsPanel({ settings, availableVoices, updateSettings }: VoiceSettingsPanelProps) {
  return (
    <div className="absolute inset-x-0 top-full mt-2 mx-4 p-3 rounded-xl border border-border/50 bg-card shadow-lg z-10 space-y-3">
      <Slider
        label="Speech Rate"
        value={[settings.rate]}
        min={0.5}
        max={2}
        step={0.1}
        showValue
        formatValue={(v) => v.toFixed(1)}
        onValueChange={([v]) => updateSettings({ rate: v })}
        size="sm"
      />
      <Slider
        label="Pitch"
        value={[settings.pitch]}
        min={0.5}
        max={2}
        step={0.1}
        showValue
        formatValue={(v) => v.toFixed(1)}
        onValueChange={([v]) => updateSettings({ pitch: v })}
        size="sm"
      />
      <Slider
        label="Interrupt Sensitivity"
        value={[settings.interruptThreshold]}
        min={0.05}
        max={0.5}
        step={0.05}
        showValue
        formatValue={(v) => v.toFixed(2)}
        onValueChange={([v]) => updateSettings({ interruptThreshold: v })}
        size="sm"
      />
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Auto-resume Listening</label>
        <Switch
          checked={settings.autoResume}
          onCheckedChange={(checked) => updateSettings({ autoResume: checked })}
          aria-label="Auto-resume listening"
        />
      </div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Push to talk</label>
        <Switch
          checked={settings.pushToTalk}
          onCheckedChange={(checked) => updateSettings({ pushToTalk: checked })}
          aria-label="Push to talk"
        />
      </div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Stream speech (speak as ready)</label>
        <Switch
          checked={settings.streamingTTS}
          onCheckedChange={(checked) => updateSettings({ streamingTTS: checked })}
          aria-label="Stream speech"
        />
      </div>

      {availableVoices.length > 0 && (
        <>
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-muted-foreground">Voice</label>
          </div>
          <Select
            value={settings.voiceName || ''}
            onValueChange={(v) => updateSettings({ voiceName: v || null })}
          >
            <SelectTrigger className="w-full text-xs">
              <SelectValue placeholder="Default" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Default</SelectItem>
              {availableVoices.filter(v => v.lang.startsWith('en')).map(v => (
                <SelectItem key={v.name} value={v.name}>{v.name} ({v.lang})</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </>
      )}
    </div>
  )
}
