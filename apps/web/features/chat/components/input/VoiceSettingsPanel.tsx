'use client'

import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@sloughgpt/strui'
import { cn } from '@sloughgpt/strui'
import type { VoiceSettings } from '@/features/chat/hooks/useVoiceChat'

interface VoiceSettingsPanelProps {
  settings: VoiceSettings
  availableVoices: Array<{ name: string; lang: string }>
  updateSettings: (s: Partial<VoiceSettings>) => void
}

export function VoiceSettingsPanel({ settings, availableVoices, updateSettings }: VoiceSettingsPanelProps) {
  return (
    <div className="absolute inset-x-0 top-full mt-2 mx-4 p-3 rounded-xl border border-border/50 bg-card shadow-lg z-10 space-y-3">
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Speech Rate</label>
        <input
          type="range"
          min={0.5}
          max={2}
          step={0.1}
          value={settings.rate}
          onChange={(e) => updateSettings({ rate: parseFloat(e.target.value) })}
          className="w-24 h-1 accent-primary"
        />
      </div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Pitch</label>
        <input
          type="range"
          min={0.5}
          max={2}
          step={0.1}
          value={settings.pitch}
          onChange={(e) => updateSettings({ pitch: parseFloat(e.target.value) })}
          className="w-24 h-1 accent-primary"
        />
      </div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Interrupt Sensitivity</label>
        <input
          type="range"
          min={0.05}
          max={0.5}
          step={0.05}
          value={settings.interruptThreshold}
          onChange={(e) => updateSettings({ interruptThreshold: parseFloat(e.target.value) })}
          className="w-24 h-1 accent-primary"
        />
      </div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Auto-resume Listening</label>
        <button
          type="button"
          onClick={() => updateSettings({ autoResume: !settings.autoResume })}
          role="switch"
          aria-checked={settings.autoResume}
          aria-label="Auto-resume listening"
          className={cn('w-9 h-5 rounded-full transition-colors', settings.autoResume ? 'bg-primary' : 'bg-muted-foreground/30')}
        >
          <div className={cn('w-4 h-4 rounded-full bg-white shadow transition-transform', settings.autoResume ? 'translate-x-4.5' : 'translate-x-0.5')} />
        </button>
      </div>
      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Push to talk</label>
        <button
          type="button"
          onClick={() => updateSettings({ pushToTalk: !settings.pushToTalk })}
          role="switch"
          aria-checked={settings.pushToTalk}
          aria-label="Push to talk"
          className={cn('w-9 h-5 rounded-full transition-colors', settings.pushToTalk ? 'bg-primary' : 'bg-muted-foreground/30')}
        >
          <div className={cn('w-4 h-4 rounded-full bg-white shadow transition-transform', settings.pushToTalk ? 'translate-x-4.5' : 'translate-x-0.5')} />
        </button>
      </div>

      <div className="flex items-center justify-between">
        <label className="text-xs font-medium text-muted-foreground">Stream speech (speak as ready)</label>
        <button
          type="button"
          onClick={() => updateSettings({ streamingTTS: !settings.streamingTTS })}
          role="switch"
          aria-checked={settings.streamingTTS}
          aria-label="Stream speech"
          className={cn('w-9 h-5 rounded-full transition-colors', settings.streamingTTS ? 'bg-primary' : 'bg-muted-foreground/30')}
        >
          <div className={cn('w-4 h-4 rounded-full bg-white shadow transition-transform', settings.streamingTTS ? 'translate-x-4.5' : 'translate-x-0.5')} />
        </button>
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
