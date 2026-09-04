import { useState } from 'react'
import type { ReactNode } from 'react'
import type { Meta, StoryObj } from '@storybook/react'

import { AttachmentChip } from '@/components/ai/attachment-chip'
import { ChatThread } from '@/components/ai/chat-thread'
import { Citation } from '@/components/ai/citation'
import { CodeSnippet } from '@/components/ai/code-snippet'
import { EmptyState } from '@/components/ai/empty-state'
import { JobStatus } from '@/components/ai/job-status'
import { MessageBubble } from '@/components/ai/message-bubble'
import { PromptComposer } from '@/components/ai/prompt-composer'
import { ReasoningPanel } from '@/components/ai/reasoning-panel'
import { SourceList } from '@/components/ai/source-list'
import { StreamingAssistantPlaceholder } from '@/components/ai/streaming-assistant-placeholder'
import { TokenMeter } from '@/components/ai/token-meter'
import { ToolCallCard } from '@/components/ai/tool-call-card'
import { TypingIndicator } from '@/components/ai/typing-indicator'
import { Breadcrumbs } from '@/components/composed/breadcrumbs'
import { Chip } from '@/components/composed/chip'
import { CopyButton } from '@/components/composed/copy-button'
import { EmptyCard } from '@/components/composed/empty-card'
import { FoldSection } from '@/components/composed/fold-section'
import { FormField } from '@/components/composed/form-field'
import { InlineBanner } from '@/components/composed/inline-banner'
import { KeyValueList } from '@/components/composed/key-value-list'
import { Kbd } from '@/components/composed/kbd'
import { KpiGrid } from '@/components/composed/kpi-grid'
import { ListRow } from '@/components/composed/list-row'
import { PageHeader } from '@/components/composed/page-header'
import { ProgressBar } from '@/components/composed/progress-bar'
import { ScrollPanel } from '@/components/composed/scroll-panel'
import { SearchInput } from '@/components/composed/search-input'
import { SectionHeader } from '@/components/composed/section-header'
import { SettingsRow } from '@/components/composed/settings-row'
import { Skeleton } from '@/components/composed/skeleton'
import { StatCard } from '@/components/composed/stat-card'
import { StatusDot } from '@/components/composed/status-dot'
import { StepIndicator } from '@/components/composed/step-indicator'
import { Timeline } from '@/components/composed/timeline'
import { Toolbar } from '@/components/composed/toolbar'
import { ActionCard } from '@/components/composed/action-card'
import { ChipGroup } from '@/components/composed/chip-group'
import { DetailRow } from '@/components/composed/detail-row'
import { InfoCard } from '@/components/composed/info-card'
import { InsightsCard } from '@/components/composed/insights-card'
import { MetricsCard } from '@/components/composed/metrics-card'
import { SortDropdown } from '@/components/composed/sort-dropdown'
import { TabGroup } from '@/components/composed/tab-group'
import { StatusBadge } from '@/components/composed/status-badge'
import { DetailList } from '@/components/composed/detail-list'
import { LoadingCard } from '@/components/composed/loading-card'
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from '@/components/ui/alert-dialog'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'

const meta = {
  title: 'Docs/Component gallery',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Single scrollable overview of primitives, composed blocks, and AI surfaces. Use individual stories under **UI**, **Composed**, and **AI** for full Controls and docs.',
      },
    },
  },
  tags: ['autodocs'],
} satisfies Meta

export default meta

function Section({
  kicker,
  title,
  subtitle,
  children,
}: {
  kicker: string
  title: string
  subtitle?: string
  children: ReactNode
}) {
  return (
    <section className="space-y-5">
      <div className="space-y-1 border-b border-border pb-4">
        <p className="text-xs font-bold uppercase tracking-[0.2em] text-primary">{kicker}</p>
        <h2 className="sl-h2">{title}</h2>
        {subtitle ? <p className="max-w-3xl text-sm text-muted-foreground">{subtitle}</p> : null}
      </div>
      <div className="sl-card-solid p-5 sm:p-6">{children}</div>
    </section>
  )
}

function GalleryCanvas() {
  const [prompt, setPrompt] = useState('Ask about the lattice…')
  const [on, setOn] = useState(true)

  return (
    <div className="str-safe-x pb-24 pt-10">
      <div className="mx-auto max-w-6xl space-y-16 px-4 md:px-6">
        <header className="space-y-4 border-b border-border pb-10">
          <Breadcrumbs
            className="text-[0.7rem]"
            items={[{ label: 'strui', href: '#' }, { label: 'Gallery' }]}
          />
          <div className="flex flex-wrap items-end justify-between gap-6">
            <div className="min-w-0 space-y-2">
              <h1 className="sl-h1">Component gallery</h1>
              <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
                Pastel lattice · Outfit + JetBrains Mono · sharp corners · Radix under the hood. Toggle{' '}
                <strong className="font-medium text-foreground">Surface</strong> in the toolbar for light/dark.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <StatusDot tone="success" label="Design tokens" showLabel />
              <StatusDot tone="primary" pulse label="Storybook" showLabel />
              <Kbd>⌘</Kbd>
              <Kbd>K</Kbd>
            </div>
          </div>
        </header>

        <Section
          kicker="UI primitives"
          title="Buttons, forms, surfaces"
          subtitle="Shadcn-style API via CVA. Use Controls on each story for variants and sizes."
        >
          <div className="flex flex-wrap gap-2">
            <Button>Primary</Button>
            <Button variant="secondary">Secondary</Button>
            <Button variant="outline">Outline</Button>
            <Button variant="ghost">Ghost</Button>
            <Button variant="destructive">Destructive</Button>
            <Button size="sm">Small</Button>
            <CopyButton text="copy-me" />
          </div>
          <Separator className="my-6" />
          <div className="grid gap-6 sm:grid-cols-2">
            <div className="space-y-3">
              <Label htmlFor="g-in">Input</Label>
              <Input id="g-in" placeholder="Type here…" />
              <Label htmlFor="g-ta">Textarea</Label>
              <Textarea id="g-ta" rows={3} placeholder="Multiline…" />
            </div>
            <div className="space-y-3">
              <Label>Search</Label>
              <SearchInput placeholder="Filter components…" />
              <div className="flex items-center gap-2">
                <Switch id="g-sw" checked={on} onCheckedChange={setOn} />
                <Label htmlFor="g-sw">Notifications</Label>
              </div>
              <div className="flex flex-wrap gap-2 pt-1">
                <Badge>default</Badge>
                <Badge variant="secondary">secondary</Badge>
                <Badge variant="outline">outline</Badge>
                <Badge variant="success">success</Badge>
              </div>
            </div>
          </div>
          <Separator className="my-6" />
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Card</CardTitle>
                <CardDescription>Lattice border and soft fill.</CardDescription>
              </CardHeader>
              <CardContent className="text-sm text-muted-foreground">Card content area.</CardContent>
            </Card>
            <div className="space-y-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Tabs</p>
              <Tabs defaultValue="a">
                <TabsList>
                  <TabsTrigger value="a">Alpha</TabsTrigger>
                  <TabsTrigger value="b">Beta</TabsTrigger>
                </TabsList>
                <TabsContent value="a" className="mt-3 text-sm text-muted-foreground">
                  Tab A content
                </TabsContent>
                <TabsContent value="b" className="mt-3 text-sm text-muted-foreground">
                  Tab B content
                </TabsContent>
              </Tabs>
            </div>
          </div>
          <Separator className="my-6" />
          <div className="flex flex-wrap items-center gap-3">
            <Dialog>
              <DialogTrigger asChild>
                <Button variant="outline" size="sm">
                  Open dialog
                </Button>
              </DialogTrigger>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Dialog</DialogTitle>
                  <DialogDescription>Modal surface for confirmations and forms.</DialogDescription>
                </DialogHeader>
                <p className="text-sm text-muted-foreground">Body copy uses the same tokens as the page.</p>
              </DialogContent>
            </Dialog>
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="secondary" size="sm">
                  Alert dialog
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Destructive action?</AlertDialogTitle>
                  <AlertDialogDescription>This uses the alert palette for emphasis.</AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction>Continue</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  Menu
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuItem>Item one</DropdownMenuItem>
                <DropdownMenuItem>Item two</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </Section>

        <Section
          kicker="Composed"
          title="Dashboards & settings"
          subtitle="Layouts, metrics, filters, and metadata patterns for app shells."
        >
          <PageHeader
            title="Page header"
            description="Optional description sits beside actions on wide screens."
            actions={
              <>
                <Button size="sm">Primary</Button>
                <Button size="sm" variant="secondary">
                  Secondary
                </Button>
              </>
            }
          />
          <div className="mt-8 space-y-6">
            <SectionHeader title="Section header" description="Smaller than page title." action={<Chip variant="primary">v2</Chip>} />
            <Toolbar className="flex flex-wrap gap-2">
              <SearchInput className="max-w-xs" placeholder="Search…" />
              <Button size="sm" variant="ghost">
                Refresh
              </Button>
            </Toolbar>
            <KpiGrid>
            <StatCard label="Requests" value="12.4k" trend={{ value: 3, positive: true }} />
            <StatCard label="Latency" value="420ms" />
            </KpiGrid>
            <InlineBanner
              variant="warning"
              title="Quota"
              description="You are approaching the workspace cap."
              action={<Button size="sm">Upgrade</Button>}
            />
            <div className="grid gap-6 lg:grid-cols-2">
              <EmptyCard message="No experiments" action={<Button size="sm" className="w-full sm:w-auto">Create</Button>} />
              <div className="space-y-3">
                <Skeleton className="h-4 w-full" />
                <Skeleton className="h-4 w-2/3" />
              </div>
            </div>
            <KeyValueList
              dense
              items={[
                { label: 'Run id', value: 'run_01a', mono: true },
                { label: 'Dataset', value: 'shakespeare' },
              ]}
            />
            <ProgressBar value={62} />
            <StepIndicator steps={['Prepare', 'Train', 'Export']} current={1} />
            <div className="flex flex-wrap gap-2">
              <Chip>filter</Chip>
              <Chip variant="outline" onRemove={() => {}}>
                removable
              </Chip>
            </div>
            <ScrollPanel className="max-h-32 p-3 font-mono text-xs">
              <div>line 1 · log output</div>
              <div>line 2 · log output</div>
              <div>line 3 · log output</div>
            </ScrollPanel>
            <ListRow label="Conversation" value="Updated 2h ago" action="›" />
            <FoldSection heading="Fold section">
              <p className="text-sm">Collapsed details — dataset manifest, env overrides, etc.</p>
            </FoldSection>
            <Timeline
              items={[
                { id: '1', title: 'Queued', meta: 'job_7f3' },
                { id: '2', title: 'Running', meta: 'step 1200' },
              ]}
            />
            <FormField id="gf" label="Form field" hint="Helper under the control.">
              <Input id="gf" />
            </FormField>
            <SettingsRow title="Setting row" description="Two-column settings row pattern." control={<Switch defaultChecked />} />
          </div>

          <Separator className="my-8" />

          <div className="space-y-6">
            <p className="text-xs font-bold uppercase tracking-[0.2em] text-primary">New Composites</p>
            <div className="grid gap-6 lg:grid-cols-2">
              <ActionCard
                title="System Health"
                subtitle="Real-time metrics"
                actions={<Button size="sm" variant="ghost">Refresh</Button>}
              >
                <div className="space-y-2 text-xs">
                  <div className="flex justify-between"><span className="text-muted-foreground">CPU</span><span>45%</span></div>
                  <div className="flex justify-between"><span className="text-muted-foreground">Memory</span><span>8.2 GB</span></div>
                </div>
              </ActionCard>
              <InsightsCard
                title="Training Insights"
                kpis={[{ label: 'Loss', value: '0.23' }, { label: 'Accuracy', value: '94%' }]}
                details={[{ label: 'Epochs', value: 12 }, { label: 'Learning rate', value: '2e-4' }]}
              />
            </div>
            <MetricsCard title="Overview" columns={4}>
              <StatCard label="Total Jobs" value={42} />
              <StatCard label="Active" value={3} trend={{ value: 12, positive: true }} />
              <StatCard label="Failed" value={1} trend={{ value: 5, positive: false }} />
              <StatCard label="Avg Time" value="2.4h" />
            </MetricsCard>
            <div className="flex flex-wrap gap-3">
              <StatusBadge tone="success">Running</StatusBadge>
              <StatusBadge tone="warning">Queued</StatusBadge>
              <StatusBadge tone="destructive">Failed</StatusBadge>
            </div>
            <ChipGroup chips={[{ label: 'Python', tone: 'primary' }, { label: 'PyTorch', tone: 'success' }, { label: 'GPU', tone: 'warning' }]} />
            <InfoCard icon={<IconCpu className="h-4 w-4" />} title="GPU" description="NVIDIA RTX 4090" tone="success" size="sm" />
            <DetailList items={[{ label: 'Model', value: 'GPT-2' }, { label: 'Path', value: '/models/gpt2', mono: true }]} />
            <LoadingCard title="Loading..." rows={2} />
            <SortDropdown value="newest" options={[{ value: 'newest', label: 'Newest' }, { value: 'oldest', label: 'Oldest' }]} onChange={() => {}} />
            <TabGroup defaultValue="a" tabs={[{ value: 'a', label: 'Tab A', content: <p className="text-sm">Content A</p> }, { value: 'b', label: 'Tab B', content: <p className="text-sm">Content B</p> }]} />
          </div>
        </Section>

        <Section
          kicker="AI"
          title="Chat, agents, RAG"
          subtitle="Streaming affordances, citations, tool calls, and job badges."
        >
          <div className="space-y-6">
            <div className="flex flex-wrap gap-2">
              <JobStatus status="running" />
              <JobStatus status="success" />
              <JobStatus status="error" />
            </div>
            <div className="flex flex-col gap-2">
              <MessageBubble role="user">User message bubble</MessageBubble>
              <MessageBubble role="assistant">Assistant transcript — readable prose, no card.</MessageBubble>
            </div>
            <TypingIndicator />
            <StreamingAssistantPlaceholder lines={2} />
            <TokenMeter total={1800} contextLimit={8000} />
            <ToolCallCard name="read_file" argsPreview='{"path": "README.md"}' state="pending" />
            <CodeSnippet>{`def train():\n    return "ok"`}</CodeSnippet>
            <div className="flex flex-wrap gap-1 text-sm">
              <Citation index={1} href="https://example.com" />
              <Citation index={2} />
            </div>
            <ReasoningPanel title="Reasoning" defaultOpen>
              <p className="text-sm text-muted-foreground">Optional chain-of-thought panel.</p>
            </ReasoningPanel>
            <SourceList
              sources={[
                { title: 'Docs', url: 'https://example.com', snippet: '…' },
                { title: 'Notebook', snippet: 'local' },
              ]}
            />
            <AttachmentChip name="image.png" onRemove={() => {}} />
            <EmptyState title="No messages" description="Start a conversation.">
              <Button size="sm">New chat</Button>
            </EmptyState>
            <ChatThread className="max-h-48 border border-border">
              <MessageBubble role="user">Short thread</MessageBubble>
              <MessageBubble role="assistant">Inside scroll region.</MessageBubble>
            </ChatThread>
            <PromptComposer
              value={prompt}
              onValueChange={setPrompt}
              onSubmit={() => {}}
              placeholder="Message…"
            />
          </div>
        </Section>

        <footer className="border-t border-border pt-8 text-center text-xs text-muted-foreground">
          strui · SloughGPT design system — see per-component stories for full controls and props tables.
        </footer>
      </div>
    </div>
  )
}

export const AllComponents: StoryObj = {
  render: () => <GalleryCanvas />,
}
