'use client'

/** Chat defaults, memory, command reference and keyboard shortcuts. */
import { VersionBadge, WEB_VERSION } from './settings-ui'
import { SettingsChatDefaultsCard } from '@/components/settings/SettingsChatDefaultsCard'
import { SettingsMemoryCard } from '@/components/settings/SettingsMemoryCard'
import { DEFAULT_SETTINGS } from '@/lib/store'
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from '@sloughgpt/strui'
import type { SettingsWorkbench } from './useSettingsWorkbench'

export function SettingsInteraction({ wb }: { wb: SettingsWorkbench }) {
  const { settings, updateSettings } = wb

  return (
    <>
      {/* Chat defaults */}
      <SettingsChatDefaultsCard
        temperature={settings.defaultTemp ?? 0.7}
        maxTokens={settings.defaultMaxTokens ?? 512}
        topP={settings.defaultTopP ?? 0.9}
        topK={settings.defaultTopK ?? 50}
        streaming={settings.streaming ?? true}
        collapsibleMessageLength={settings.collapsibleMessageLength ?? 0}
        onTemperatureChange={(v) => updateSettings({ defaultTemp: v })}
        onMaxTokensChange={(v) => updateSettings({ defaultMaxTokens: v })}
        onTopPChange={(v) => updateSettings({ defaultTopP: v })}
        onTopKChange={(v) => updateSettings({ defaultTopK: v })}
        onStreamingChange={(v) => updateSettings({ streaming: v })}
        onCollapsibleMessageLengthChange={(v) => updateSettings({ collapsibleMessageLength: v })}
        onReset={() =>
          updateSettings({
            defaultTemp: DEFAULT_SETTINGS.defaultTemp,
            defaultMaxTokens: DEFAULT_SETTINGS.defaultMaxTokens,
            defaultTopP: DEFAULT_SETTINGS.defaultTopP,
            defaultTopK: DEFAULT_SETTINGS.defaultTopK,
          })
        }
        version={WEB_VERSION}
      />

      {/* Memory */}
      <SettingsMemoryCard
        customContext={settings.customContext ?? ''}
        onChange={(v) => updateSettings({ customContext: v })}
        version={WEB_VERSION}
      />

      {/* Chat commands reference */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">Chat commands</CardTitle>
            <CardDescription>
              Type{' '}
              <kbd className="inline-flex items-center rounded border border-border/40 bg-muted px-1.5 py-0.5 text-[11px] font-mono">
                /
              </kbd>{' '}
              in chat to open the command palette
            </CardDescription>
          </div>
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
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>

      {/* Keyboard shortcuts */}
      <Card>
        <CardHeader>
          <div>
            <CardTitle className="text-base">Keyboard shortcuts</CardTitle>
            <CardDescription>Global shortcuts available on any page</CardDescription>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
            {[
              ['Ctrl+1–9', 'Navigate pages (Chat, Models, etc.)'],
              ['Ctrl+N', 'New conversation'],
              ['Ctrl+\\', 'Toggle sidebar'],
              ['Ctrl+K', 'Command palette'],
              ['Ctrl+Shift+F', 'Search conversations'],
              ['Ctrl+Shift+A', 'Open settings'],
              ['Ctrl+Shift+M', 'Toggle dark mode'],
              ['?', 'Show keyboard shortcuts'],
              ['Esc', 'Close dialog / Cancel'],
            ].map(([key, desc]) => (
              <div key={key} className="flex items-center gap-2">
                <kbd className="text-xs text-primary font-mono shrink-0 min-w-[80px]">{key}</kbd>
                <span className="text-xs text-muted-foreground truncate">{desc}</span>
              </div>
            ))}
          </div>
        </CardContent>
        <CardFooter className="justify-end">
          <VersionBadge label="v" version={WEB_VERSION} />
        </CardFooter>
      </Card>
    </>
  )
}
