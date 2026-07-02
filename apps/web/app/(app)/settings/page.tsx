'use client'

export const dynamic = 'force-dynamic'

import { AppRouteHeader, AppRouteHeaderLead } from '@/components/AppRouteHeader'
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
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Slider } from '@/components/ui/slider'
import { ToggleGroup as ToggleGroupRadix, ToggleGroupItem } from '@/components/ui/toggle-group'
import { useToastStore } from '@/lib/toast-store'
import { useSettings, useUpdateSettings } from '@/lib/store'
import { useApiHealth } from '@/hooks/useApiHealth'
import Link from 'next/link'

function SettingsSlider({
  label, value, onChange, min, max, step, formatValue,
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  formatValue?: (v: number) => string
}) {
  const display = formatValue ? formatValue(value) : value.toString()
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="text-sm font-medium">{label}</label>
        <span className="text-sm text-muted-foreground">{display}</span>
      </div>
      <Slider value={[value]} onValueChange={([v]) => onChange(v)} min={min} max={max} step={step} />
    </div>
  )
}

export default function SettingsPage() {
  const settings = useSettings()
  const updateSettings = useUpdateSettings()
  const addToast = useToastStore(s => s.addToast)
  const { state: apiHealth } = useApiHealth()

  const clearChat = () => {
    localStorage.removeItem('man_messages')
    localStorage.removeItem('man_chat_sessions')
    localStorage.removeItem('man_current_session')
    addToast('Chat history cleared', 'success')
  }

  const resetAllSettings = () => {
    localStorage.removeItem('man_settings')
    window.location.reload()
  }

  return (
    <div className="sl-page mx-auto max-w-4xl">
      <AppRouteHeader left={<AppRouteHeaderLead title="Settings" />} />

      <div className="space-y-4">
        {/* Appearance */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Appearance</CardTitle>
            <CardDescription>Theme preference</CardDescription>
          </CardHeader>
          <CardContent>
            <ToggleGroupRadix type="single" value={settings.theme} onValueChange={(v) => v && updateSettings({ theme: v as 'light' | 'dark' | 'system' })}>
              <ToggleGroupItem value="light">Light</ToggleGroupItem>
              <ToggleGroupItem value="dark">Dark</ToggleGroupItem>
              <ToggleGroupItem value="system">System</ToggleGroupItem>
            </ToggleGroupRadix>
          </CardContent>
        </Card>

        {/* Chat defaults */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Chat defaults</CardTitle>
            <CardDescription>Default model and generation settings</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <SettingsSlider
                label="Temperature"
                value={settings.defaultTemp}
                onChange={(v) => updateSettings({ defaultTemp: v })}
                min={0}
                max={2}
                step={0.1}
              />
              <SettingsSlider
                label="Max tokens"
                value={settings.defaultMaxTokens}
                onChange={(v) => updateSettings({ defaultMaxTokens: v })}
                min={50}
                max={1000}
                step={50}
              />
            </div>
            <div className="pt-2">
              <SettingsSlider
                label="Auto-collapse messages longer than"
                value={settings.collapsibleMessageLength}
                onChange={(v) => updateSettings({ collapsibleMessageLength: v })}
                min={0}
                max={2000}
                step={50}
                formatValue={(v) => v === 0 ? 'Disabled' : `${v} chars`}
              />
            </div>
          </CardContent>
        </Card>

        {/* Memory */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Memory</CardTitle>
            <CardDescription>Custom instructions included with every prompt</CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              className="min-h-[120px]"
              placeholder="e.g., You are a helpful coding assistant. Keep responses concise..."
              value={settings.customContext}
              onChange={(e) => updateSettings({ customContext: e.target.value })}
              aria-label="Custom instructions"
            />
          </CardContent>
        </Card>

        {/* Knowledge base */}
        <Link href="/knowledge" className="block">
          <Card className="cursor-pointer hover:border-primary/30 transition-colors">
            <CardHeader>
              <CardTitle className="text-base">Knowledge base</CardTitle>
              <CardDescription>Store, edit, and browse facts the AI can reference</CardDescription>
            </CardHeader>
          </Card>
        </Link>

        {/* Tokenizer */}
        <Link href="/tokenizer" className="block">
          <Card className="cursor-pointer hover:border-primary/30 transition-colors">
            <CardHeader>
              <CardTitle className="text-base">Tokenizer</CardTitle>
              <CardDescription>Byte-pair encoding — explore vocab, tokens, merges</CardDescription>
            </CardHeader>
          </Card>
        </Link>

        {/* Chat commands reference */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Chat commands</CardTitle>
            <CardDescription>Type <kbd className="inline-flex items-center rounded border border-border/40 bg-muted px-1.5 py-0.5 text-[11px] font-mono">/</kbd> in chat to open the command palette</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
              {[
                ['/help', 'Show all commands'],
                ['/clear', 'Clear the chat'],
                ['/temp <n>', 'Set temperature (0–2)'],
                ['/model <name>', 'Switch model'],
                ['/soul <name>', 'Switch soul'],
                ['/export', 'Export chat as MD'],
                ['/file', 'Attach a file'],
                ['/knowledge <q>', 'Search knowledge'],
                ['/goto <path>', 'Navigate to page'],
                ['/summarize', 'Summarise chat'],
                ['/feedback +/- [r]', 'Rate response'],
                ['/translate <lang>', 'Translate reply'],
                ['/search <q>', 'Search conversations'],
                ['/archive', 'Archive & start fresh'],
                ['/rename <n>', 'Rename conversation'],
              ].map(([cmd, desc]) => (
                <div key={cmd} className="flex items-center gap-2">
                  <code className="text-xs text-primary font-mono shrink-0">{cmd}</code>
                  <span className="text-xs text-muted-foreground truncate">{desc}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* System health */}
        <Link href="/monitoring" className="block">
          <Card className="cursor-pointer hover:border-primary/30 transition-colors">
            <CardHeader>
              <CardTitle className="text-base">System health</CardTitle>
              <CardDescription>Backend status and resource usage</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex items-center gap-3 text-sm">
                <div className={`w-2 h-2 rounded-full ${
                  apiHealth === null ? 'bg-warning' :
                  apiHealth === 'offline' ? 'bg-destructive' :
                  'bg-success'
                }`} />
                <span>{
                  apiHealth === null ? 'Connecting…' :
                  apiHealth === 'offline' ? 'Server offline' :
                  apiHealth.model_loaded ? `${apiHealth.model_type} loaded` :
                  'Online, no model loaded'
                }</span>
              </div>
            </CardContent>
          </Card>
        </Link>

        {/* Export / Import settings */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Backup & restore</CardTitle>
            <CardDescription>Export your settings to a file, or import from a backup</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <Button size="sm" variant="outline" onClick={() => {
                const data = JSON.stringify(settings, null, 2)
                const blob = new Blob([data], { type: 'application/json' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url; a.download = 'sloughgpt-settings.json'; a.click()
                URL.revokeObjectURL(url)
                addToast('Settings exported', 'success')
              }}>Export settings</Button>
              <Button size="sm" variant="outline" onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'; input.accept = '.json'
                input.onchange = async (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (!file) return
                  try {
                    const text = await file.text()
                    const data = JSON.parse(text)
                    updateSettings(data)
                    addToast('Settings imported', 'success')
                  } catch {
                    addToast('Invalid settings file', 'error')
                  }
                }
                input.click()
              }}>Import settings</Button>
            </div>
          </CardContent>
        </Card>

        {/* Danger zone */}
        <Card className="border-destructive/30">
          <CardHeader>
            <CardTitle className="text-base text-destructive">Danger zone</CardTitle>
            <CardDescription>Irreversible actions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">Clear chat history</p>
                <p className="text-xs text-muted-foreground">Removes all saved conversations from this browser</p>
              </div>
              <AlertDialog>
                <AlertDialogTrigger asChild>
                  <Button type="button" variant="destructive" size="sm">Clear</Button>
                </AlertDialogTrigger>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Clear all chat history?</AlertDialogTitle>
                    <AlertDialogDescription>This cannot be undone.</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <AlertDialogAction onClick={clearChat} className="bg-destructive text-destructive-foreground">Clear</AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          </CardContent>
        </Card>

        {/* Reset all settings */}
        <Card className="border-destructive/30">
          <CardHeader>
            <CardTitle className="text-base text-destructive">Reset all settings</CardTitle>
            <CardDescription>Restore all settings to their defaults</CardDescription>
          </CardHeader>
          <CardContent>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button type="button" variant="destructive" size="sm">Reset</Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Reset all settings?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will clear your theme preference, default model settings, and custom instructions. Chat history is not affected.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={resetAllSettings} className="bg-destructive text-destructive-foreground">Reset</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
