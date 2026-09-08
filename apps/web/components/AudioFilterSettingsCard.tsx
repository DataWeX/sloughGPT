'use client'

import { useState, useEffect, useCallback } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@sloughgpt/strui'
import { Slider } from '@sloughgpt/strui'
import { Switch } from '@sloughgpt/strui'
import { Button } from '@sloughgpt/strui'
import { Badge } from '@sloughgpt/strui'
import { Skeleton } from '@sloughgpt/strui'
import {
  audioFilterController,
  type AudioFilterConfig,
  type AudioFilterUpdate,
} from '@/lib/audio-filter-controller'
import { useToastStore } from '@/lib/toast-store'

const MODE_LABELS: Record<string, string> = {
  none: 'Disabled',
  noise_gate: 'Noise Gate',
  agc: 'Auto Gain',
  normalize: 'Normalize',
  vad: 'Voice Detection',
  all: 'All Filters',
}

function SliderField({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  format?: (v: number) => string
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-xs text-muted-foreground">{label}</label>
        <span className="text-xs font-mono text-muted-foreground">
          {format ? format(value) : value}
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]: number[]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      />
    </div>
  )
}

export function AudioFilterSettingsCard() {
  const addToast = useToastStore(s => s.addToast)
  const [config, setConfig] = useState<AudioFilterConfig | null>(null)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [pending, setPending] = useState<AudioFilterUpdate>({})

  useEffect(() => {
    audioFilterController
      .getConfig()
      .then(c => {
        setConfig(c)
        setLoading(false)
      })
      .catch(() => setLoading(false))
  }, [])

  const update = useCallback(
    (patch: AudioFilterUpdate) => {
      setPending(prev => ({ ...prev, ...patch }))
    },
    [],
  )

  const handleSave = useCallback(async () => {
    if (Object.keys(pending).length === 0) return
    setSaving(true)
    try {
      await audioFilterController.updateConfig(pending)
      const fresh = await audioFilterController.getConfig()
      setConfig(fresh)
      setPending({})
      addToast('Audio filter settings saved', 'success')
    } catch {
      addToast('Failed to save audio filter settings', 'error')
    } finally {
      setSaving(false)
    }
  }, [pending, addToast])

  const handleReset = useCallback(async () => {
    setSaving(true)
    try {
      await audioFilterController.updateConfig({ mode: 'all' })
      const fresh = await audioFilterController.getConfig()
      setConfig(fresh)
      setPending({})
      addToast('Audio filter reset to defaults', 'success')
    } catch {
      addToast('Failed to reset audio filter', 'error')
    } finally {
      setSaving(false)
    }
  }, [addToast])

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Audio Filter</CardTitle>
          <CardDescription>Voice sensitivity and noise filtering</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-4 w-48" />
          <Skeleton className="h-4 w-64" />
          <Skeleton className="h-4 w-56" />
        </CardContent>
      </Card>
    )
  }

  if (!config) return null

  const mode = pending.mode ?? config.mode
  const hasChanges = Object.keys(pending).length > 0

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base">Audio Filter</CardTitle>
            <CardDescription>
              Voice sensitivity and noise filtering applied before transcription
            </CardDescription>
          </div>
          <Badge variant={mode === 'none' ? 'secondary' : 'default'}>
            {MODE_LABELS[mode] ?? mode}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Mode toggle */}
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Enable all filters</p>
            <p className="text-xs text-muted-foreground">
              Noise gate + auto gain + normalization + voice detection
            </p>
          </div>
          <Switch
            checked={mode === 'all'}
            onCheckedChange={(checked) => update({ mode: checked ? 'all' : 'none' })}
            aria-label="Toggle audio filter"
          />
        </div>

        {mode !== 'none' && (
          <>
            <div className="border-t border-border/30" />

            {/* Noise Gate */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Noise Gate</p>
              <SliderField
                label="Threshold"
                value={pending.noise_gate_threshold_db ?? config.noise_gate_threshold_db}
                onChange={(v) => update({ noise_gate_threshold_db: v })}
                min={-80}
                max={0}
                step={1}
                format={(v) => `${v} dB`}
              />
              <SliderField
                label="Attack"
                value={pending.noise_gate_attack_ms ?? config.noise_gate_attack_ms}
                onChange={(v) => update({ noise_gate_attack_ms: v })}
                min={1}
                max={50}
                step={1}
                format={(v) => `${v}ms`}
              />
              <SliderField
                label="Release"
                value={pending.noise_gate_release_ms ?? config.noise_gate_release_ms}
                onChange={(v) => update({ noise_gate_release_ms: v })}
                min={10}
                max={500}
                step={10}
                format={(v) => `${v}ms`}
              />
            </div>

            <div className="border-t border-border/30" />

            {/* AGC */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Automatic Gain Control</p>
              <SliderField
                label="Target level"
                value={pending.agc_target_db ?? config.agc_target_db}
                onChange={(v) => update({ agc_target_db: v })}
                min={-60}
                max={0}
                step={1}
                format={(v) => `${v} dB`}
              />
              <SliderField
                label="Max gain"
                value={pending.agc_max_gain_db ?? config.agc_max_gain_db}
                onChange={(v) => update({ agc_max_gain_db: v })}
                min={0}
                max={60}
                step={1}
                format={(v) => `${v} dB`}
              />
            </div>

            <div className="border-t border-border/30" />

            {/* Normalization */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Loudness Normalization</p>
              <SliderField
                label="Target LUFS"
                value={pending.target_lufs ?? config.target_lufs}
                onChange={(v) => update({ target_lufs: v })}
                min={-50}
                max={0}
                step={1}
                format={(v) => `${v} LUFS`}
              />
            </div>

            <div className="border-t border-border/30" />

            {/* VAD */}
            <div className="space-y-3">
              <p className="text-sm font-medium">Voice Activity Detection</p>
              <SliderField
                label="Energy threshold"
                value={pending.vad_energy_threshold_db ?? config.vad_energy_threshold_db}
                onChange={(v) => update({ vad_energy_threshold_db: v })}
                min={-80}
                max={0}
                step={1}
                format={(v) => `${v} dB`}
              />
              <SliderField
                label="Min speech duration"
                value={pending.vad_min_speech_ms ?? config.vad_min_speech_ms}
                onChange={(v) => update({ vad_min_speech_ms: v })}
                min={50}
                max={1000}
                step={50}
                format={(v) => `${v}ms`}
              />
            </div>
          </>
        )}

        {/* Actions */}
        {hasChanges && (
          <div className="flex items-center justify-end gap-2 pt-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setPending({})}
              disabled={saving}
            >
              Discard
            </Button>
            <Button
              type="button"
              size="sm"
              onClick={handleSave}
              disabled={saving}
            >
              {saving ? 'Saving...' : 'Save'}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
