'use client'

import { useState, useEffect, useCallback } from 'react'
import { PageContainer } from '@/components/PageContainer'
import { consciousnessController } from '@/lib/consciousness-controller'
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
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
  Input,
  Skeleton,
  Switch,
} from '@sloughgpt/strui'
import { useToastStore } from '@/lib/toast-store'
import { extractErrorMessage } from '@/lib/error-utils'
import { useLocale } from '@/hooks/useLocale'

const VERSIONS_STORAGE_KEY = 'consciousness_versions'
const AUTO_SAVE_KEY = 'consciousness_auto_save'
const MAX_AUTO_SAVES = 20

interface ConfigSnapshot {
  level: number
  enabled: boolean
}

interface PersonalitySnapshot {
  voice?: string
  traits?: string[]
  style?: string
}

interface SavedVersion {
  id: string
  name: string
  timestamp: string
  config: ConfigSnapshot
  personality: PersonalitySnapshot
  autoSaved?: boolean
}

function loadVersions(): SavedVersion[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(VERSIONS_STORAGE_KEY)
    if (!raw) return []
    return JSON.parse(raw)
  } catch {
    return []
  }
}

function saveVersions(versions: SavedVersion[]) {
  try {
    localStorage.setItem(VERSIONS_STORAGE_KEY, JSON.stringify(versions))
  } catch { /* ignored */ }
}

function loadAutoSaves(): SavedVersion[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = localStorage.getItem(AUTO_SAVE_KEY)
    if (!raw) return []
    return JSON.parse(raw)
  } catch {
    return []
  }
}

function saveAutoSaves(saves: SavedVersion[]) {
  try {
    localStorage.setItem(AUTO_SAVE_KEY, JSON.stringify(saves))
  } catch { /* ignored */ }
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

function buildDiff(a: string, b: string): { value: string; type: 'same' | 'changed' }[] {
  if (a === b) return [{ value: a, type: 'same' }]
  return [
    { value: a, type: 'changed' },
    { value: b, type: 'changed' },
  ]
}

const LEVEL_LABELS = ['Off', 'Basic', 'Full', 'Deep']

export default function ConsciousnessVersionsPage() {
  const addToast = useToastStore(s => s.addToast)
  const { t } = useLocale()
  const [config, setConfig] = useState<ConfigSnapshot | null>(null)
  const [personality, setPersonality] = useState<PersonalitySnapshot | null>(null)
  const [versions, setVersions] = useState<SavedVersion[]>([])
  const [autoSaves, setAutoSaves] = useState<SavedVersion[]>([])
  const [autoSaveEnabled, setAutoSaveEnabled] = useState(false)
  const [loading, setLoading] = useState(true)
  const [savingName, setSavingName] = useState('')
  const [showSaveDialog, setShowSaveDialog] = useState(false)
  const [compareLeft, setCompareLeft] = useState<string>('')
  const [compareRight, setCompareRight] = useState<string>('')
  const [showCompare, setShowCompare] = useState(false)
  const [restoreTarget, setRestoreTarget] = useState<SavedVersion | null>(null)
  const [showRestoreDialog, setShowRestoreDialog] = useState(false)

  const fetchConfig = useCallback(async () => {
    try {
      const data = await consciousnessController.getStatus() as any
      setConfig({ level: data?.level ?? 0, enabled: data?.enabled ?? false })
    } catch (e) {
      console.error('Failed to fetch consciousness config', e)
    }
  }, [])

  const fetchPersonality = useCallback(async () => {
    try {
      const data = await consciousnessController.getPersonality() as any
      setPersonality({
        voice: data?.voice || data?.tone || '',
        traits: data?.traits || data?.personality_traits || [],
        style: data?.style || data?.response_style || '',
      })
    } catch (e) {
      console.error('Failed to fetch consciousness personality', e)
    }
  }, [])

  useEffect(() => {
    setVersions(loadVersions())
    setAutoSaves(loadAutoSaves())
    const savedAutoSave = loadAutoSaves()
    setAutoSaveEnabled(savedAutoSave.length > 0 || localStorage.getItem(AUTO_SAVE_KEY + '_enabled') === 'true')
    Promise.all([fetchConfig(), fetchPersonality()]).finally(() => setLoading(false))
  }, [fetchConfig, fetchPersonality])

  const handleSave = () => {
    if (!config || !personality) return
    const name = savingName.trim()
    if (!name) {
      addToast('Version name is required', 'error')
      return
    }
    const version: SavedVersion = {
      id: crypto.randomUUID(),
      name,
      timestamp: new Date().toISOString(),
      config: { ...config },
      personality: { ...personality },
    }
    const updated = [version, ...versions]
    setVersions(updated)
    saveVersions(updated)
    setSavingName('')
    setShowSaveDialog(false)
    addToast(`Version "${name}" saved`, 'success')
  }

  const handleDelete = (id: string) => {
    const updated = versions.filter(v => v.id !== id)
    setVersions(updated)
    saveVersions(updated)
    addToast('Version deleted', 'success')
  }

  const handleRestore = async () => {
    if (!restoreTarget || !config) return
    if (autoSaveEnabled) {
      const autoSavePoint: SavedVersion = {
        id: crypto.randomUUID(),
        name: `Auto-save before restore`,
        timestamp: new Date().toISOString(),
        config: { ...config },
        personality: { ...(personality || {}) },
        autoSaved: true,
      }
      const updated = [autoSavePoint, ...loadAutoSaves()].slice(0, MAX_AUTO_SAVES)
      saveAutoSaves(updated)
      setAutoSaves(updated)
    }

    try {
      await consciousnessController.updateConfig(restoreTarget.config as unknown as Record<string, unknown>)

      if (restoreTarget.personality) {
        await consciousnessController.updatePersonality(restoreTarget.personality as Record<string, number>)
      }

      setConfig(restoreTarget.config)
      setPersonality(restoreTarget.personality)
      setShowRestoreDialog(false)
      setRestoreTarget(null)
      addToast(`Restored version "${restoreTarget.name}"`, 'success')
      fetchConfig()
      fetchPersonality()
    } catch (e) {
      addToast(extractErrorMessage(e), 'error')
    }
  }

  const handleAutoSave = () => {
    if (!autoSaveEnabled) {
      setAutoSaveEnabled(true)
      localStorage.setItem(AUTO_SAVE_KEY + '_enabled', 'true')
      return
    }
    setAutoSaveEnabled(false)
    localStorage.removeItem(AUTO_SAVE_KEY + '_enabled')
  }

  const getCompareData = () => {
    const allItems = [...versions, ...autoSaves]
    const left = allItems.find(v => v.id === compareLeft)
    const right = allItems.find(v => v.id === compareRight)
    return { left, right }
  }

  const allItems = [...versions, ...autoSaves]

  const renderDiffRow = (label: string, leftVal: string, rightVal: string) => {
    const leftType = leftVal === rightVal ? 'same' : 'changed'
    const rightType = rightVal === leftVal ? 'same' : 'changed'
    return (
      <div className="flex items-center gap-3 text-sm">
        <span className="w-24 text-muted-foreground shrink-0">{label}</span>
        <span className={`flex-1 rounded px-2 py-1 font-mono text-xs ${leftType === 'same' ? 'bg-muted/50' : 'bg-red-500/10 text-red-600 dark:text-red-400'}`}>
          {leftVal}
        </span>
        <span className="text-muted-foreground text-xs">→</span>
        <span className={`flex-1 rounded px-2 py-1 font-mono text-xs ${rightType === 'same' ? 'bg-muted/50' : 'bg-green-500/10 text-green-600 dark:text-green-400'}`}>
          {rightVal}
        </span>
      </div>
    )
  }

  if (loading) {
    return (
      <PageContainer title={t('consciousness_versions.page_title')}>
        <div className="space-y-6 p-6">
          {[1, 2, 3].map(i => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-3 w-64" />
              </CardHeader>
              <CardContent className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-3/4" />
              </CardContent>
            </Card>
          ))}
        </div>
      </PageContainer>
    )
  }

  return (
    <PageContainer title={t('consciousness_versions.page_title')}>
      <div className="space-y-6 p-6">
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">{t('consciousness_versions.current_title')}</CardTitle>
                <CardDescription>{t('consciousness_versions.current_desc')}</CardDescription>
              </div>
              <Button size="sm" onClick={() => setShowSaveDialog(true)}>
                {t('consciousness_versions.save_button')}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {config && (
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.level')}</p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <Badge variant={config.enabled ? 'default' : 'secondary'}>
                      {config.enabled ? LEVEL_LABELS[config.level] || `Level ${config.level}` : 'Off'}
                    </Badge>
                    <span className="text-muted-foreground text-xs">({config.level}/3)</span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.status')}</p>
                  <div className="flex items-center gap-1.5 mt-0.5">
                    <span className={`inline-block w-2 h-2 rounded-full ${config.enabled ? 'bg-success' : 'bg-muted-foreground/50'}`} />
                    <span className="text-sm font-medium">{config.enabled ? 'Active' : 'Disabled'}</span>
                  </div>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.voice')}</p>
                  <p className="text-sm font-medium mt-0.5">{personality?.voice || '—'}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.style')}</p>
                  <p className="text-sm font-medium mt-0.5">{personality?.style || '—'}</p>
                </div>
              </div>
            )}
            {personality?.traits && personality.traits.length > 0 && (
              <div>
                <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.traits')}</p>
                <div className="flex flex-wrap gap-1 mt-1">
                  {personality.traits.map((trait, i) => (
                    <Badge key={i} variant="outline" className="text-xs">{trait}</Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">{t('consciousness_versions.auto_save_title')}</CardTitle>
                <CardDescription>{t('consciousness_versions.auto_save_desc')}</CardDescription>
              </div>
              <Switch
                checked={autoSaveEnabled}
                onCheckedChange={handleAutoSave}
                aria-label="Toggle auto-save"
              />
            </div>
          </CardHeader>
          {autoSaveEnabled && autoSaves.length > 0 && (
            <CardContent>
              <p className="text-xs text-muted-foreground">{autoSaves.length} {t('consciousness_versions.auto_saves_count')}</p>
            </CardContent>
          )}
        </Card>

        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-base">{t('consciousness_versions.history_title')}</CardTitle>
                <CardDescription>{t('consciousness_versions.history_desc')}</CardDescription>
              </div>
              {versions.length >= 2 && (
                <Button size="sm" variant="outline" onClick={() => setShowCompare(true)}>
                  {t('consciousness_versions.compare_button')}
                </Button>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {versions.length === 0 && autoSaves.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t('consciousness_versions.no_versions')}</p>
            ) : (
              <div className="space-y-2">
                {versions.map(version => (
                  <div
                    key={version.id}
                    className="flex items-center justify-between rounded-md border border-border/30 p-3"
                  >
                    <div className="space-y-0.5 min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium truncate">{version.name}</p>
                        <Badge variant="outline" className="text-xs shrink-0">{LEVEL_LABELS[version.config.level] || `L${version.config.level}`}</Badge>
                      </div>
                      <p className="text-xs text-muted-foreground">{formatDate(version.timestamp)}</p>
                      {version.personality?.traits && version.personality.traits.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {version.personality.traits.slice(0, 3).map((trait, i) => (
                            <Badge key={i} variant="secondary" className="text-[10px]">{trait}</Badge>
                          ))}
                          {version.personality.traits.length > 3 && (
                            <Badge variant="secondary" className="text-[10px]">+{version.personality.traits.length - 3}</Badge>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="flex gap-2 ml-3">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => { setRestoreTarget(version); setShowRestoreDialog(true) }}
                      >
                        {t('consciousness_versions.restore')}
                      </Button>
                      <Button
                        size="sm"
                        variant="destructive"
                        onClick={() => handleDelete(version.id)}
                      >
                        {t('consciousness_versions.delete')}
                      </Button>
                    </div>
                  </div>
                ))}
                {autoSaves.length > 0 && (
                  <>
                    <div className="border-t border-border/30 pt-2 mt-2">
                      <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium mb-2">{t('consciousness_versions.auto_saves')}</p>
                    </div>
                    {autoSaves.map(version => (
                      <div
                        key={version.id}
                        className="flex items-center justify-between rounded-md border border-border/30 p-3 opacity-75"
                      >
                        <div className="space-y-0.5 min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-medium truncate">{version.name}</p>
                            <Badge variant="secondary" className="text-xs shrink-0">Auto</Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">{formatDate(version.timestamp)}</p>
                        </div>
                        <div className="flex gap-2 ml-3">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => { setRestoreTarget(version); setShowRestoreDialog(true) }}
                          >
                            {t('consciousness_versions.restore')}
                          </Button>
                        </div>
                      </div>
                    ))}
                  </>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <AlertDialog open={showSaveDialog} onOpenChange={setShowSaveDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('consciousness_versions.save_dialog_title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('consciousness_versions.save_dialog_desc')}</AlertDialogDescription>
          </AlertDialogHeader>
          <Input
            value={savingName}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSavingName(e.target.value)}
            placeholder={t('consciousness_versions.name_placeholder')}
            onKeyDown={(e: React.KeyboardEvent) => { if (e.key === 'Enter') handleSave() }}
            autoFocus
          />
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => { setSavingName(''); setShowSaveDialog(false) }}>{t('consciousness_settings.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={handleSave}>{t('consciousness_versions.save')}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={showRestoreDialog} onOpenChange={setShowRestoreDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('consciousness_versions.restore_confirm_title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('consciousness_versions.restore_confirm_desc')}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => { setRestoreTarget(null); setShowRestoreDialog(false) }}>{t('consciousness_settings.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={handleRestore}>{t('consciousness_versions.restore')}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog open={showCompare} onOpenChange={setShowCompare}>
        <AlertDialogContent className="max-w-2xl">
          <AlertDialogHeader>
            <AlertDialogTitle>{t('consciousness_versions.compare_title')}</AlertDialogTitle>
            <AlertDialogDescription>{t('consciousness_versions.compare_desc')}</AlertDialogDescription>
          </AlertDialogHeader>
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.version_left')}</label>
                <select
                  value={compareLeft}
                  onChange={(e) => setCompareLeft(e.target.value)}
                  className="w-full rounded-md border border-border/30 bg-background px-3 py-2 text-sm"
                >
                  <option value="">{t('consciousness_versions.select_version')}</option>
                  {allItems.map(v => (
                    <option key={v.id} value={v.id}>{v.name} ({formatDate(v.timestamp)})</option>
                  ))}
                </select>
              </div>
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.version_right')}</label>
                <select
                  value={compareRight}
                  onChange={(e) => setCompareRight(e.target.value)}
                  className="w-full rounded-md border border-border/30 bg-background px-3 py-2 text-sm"
                >
                  <option value="">{t('consciousness_versions.select_version')}</option>
                  {allItems.map(v => (
                    <option key={v.id} value={v.id}>{v.name} ({formatDate(v.timestamp)})</option>
                  ))}
                </select>
              </div>
            </div>
            {compareLeft && compareRight && (() => {
              const { left, right } = getCompareData()
              if (!left || !right) return null
              return (
                <div className="space-y-3 rounded-md border border-border/30 p-4">
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium">{t('consciousness_versions.config_diff')}</p>
                  {renderDiffRow(t('consciousness_versions.level'), LEVEL_LABELS[left.config.level] || `L${left.config.level}`, LEVEL_LABELS[right.config.level] || `L${right.config.level}`)}
                  {renderDiffRow(t('consciousness_versions.enabled'), left.config.enabled ? 'Yes' : 'No', right.config.enabled ? 'Yes' : 'No')}
                  <div className="border-t border-border/30 pt-3 mt-3">
                    <p className="text-xs text-muted-foreground uppercase tracking-wider font-medium mb-2">{t('consciousness_versions.personality_diff')}</p>
                    {renderDiffRow(t('consciousness_versions.voice'), left.personality?.voice || '—', right.personality?.voice || '—')}
                    {renderDiffRow(t('consciousness_versions.style'), left.personality?.style || '—', right.personality?.style || '—')}
                    <div className="mt-2">
                      <div className="flex items-center gap-3 text-sm">
                        <span className="w-24 text-muted-foreground shrink-0">{t('consciousness_versions.traits')}</span>
                        <span className="flex-1 flex flex-wrap gap-1">
                          {(left.personality?.traits || []).map((trait, i) => (
                            <Badge key={i} variant="outline" className="text-xs bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/30">{trait}</Badge>
                          ))}
                        </span>
                        <span className="text-muted-foreground text-xs">→</span>
                        <span className="flex-1 flex flex-wrap gap-1">
                          {(right.personality?.traits || []).map((trait, i) => (
                            <Badge key={i} variant="outline" className="text-xs bg-green-500/10 text-green-600 dark:text-green-400 border-green-500/30">{trait}</Badge>
                          ))}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })()}
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>{t('consciousness_settings.cancel')}</AlertDialogCancel>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </PageContainer>
  )
}
