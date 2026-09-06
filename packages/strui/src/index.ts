// UI Library — our custom components from apps/web/components/ui
// Combined with @sloughgpt/strui composed + AI components

// Core utilities
export { cn } from './lib/cn'

// Versioning system
export {
  STRUI_VERSION,
  FEATURE_VERSIONS,
  getComponentFeature,
  listFeatures,
  type FeatureVersion,
  type FeatureName,
} from './versions'

// Button
export { Button, buttonVariants, type ButtonProps } from './components/ui/button'

// Badge
export { Badge, badgeVariants, type BadgeProps } from './components/ui/badge'

// Card
export { Card, CardHeader, CardTitle, CardContent, CardFooter, CardDescription, cardVariants } from './components/ui/card'

// ActionCard — card with title + actions header (exported from composed)

// Input
export { Input, inputFieldClassName, type InputProps } from './components/ui/input'

// SearchInput (from our input.tsx — overrides composed/search-input)
export { SearchInput, type SearchInputProps } from './components/ui/input'

// Textarea
export { Textarea, type TextareaProps } from './components/ui/textarea'

// Label
export { Label } from './components/ui/label'

// Switch
export { Switch } from './components/ui/switch'

// Separator
export { Separator, type SeparatorProps } from './components/ui/separator'

// Checkbox
export { Checkbox, type CheckboxProps } from './components/ui/checkbox'

// Radio
export { Radio, type RadioProps } from './components/ui/radio'
export { RadioGroup, type RadioGroupProps } from './components/ui/radio-group'

// Progress
export { Progress, type ProgressProps } from './components/ui/progress'

// Select
export { Select, SelectGroup, SelectValue, SelectTrigger, SelectContent, SelectItem, SelectSeparator } from './components/ui/select'

// Tabs
export { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs'

// Toggle Group
export { ToggleGroup, ToggleGroupItem } from './components/ui/toggle-group'

// Slider
export { Slider, RangeSlider } from './components/ui/slider'

// Dialog
export { Dialog, DialogPortal, DialogOverlay, DialogTrigger, DialogContent, DialogHeader, DialogFooter, DialogTitle, DialogDescription, DialogClose } from './components/ui/dialog'

// Alert Dialog
export { AlertDialog, AlertDialogPortal, AlertDialogOverlay, AlertDialogTrigger, AlertDialogContent, AlertDialogHeader, AlertDialogFooter, AlertDialogTitle, AlertDialogDescription, AlertDialogAction, AlertDialogCancel } from './components/ui/alert-dialog'

// Dropdown Menu
export { DropdownMenu, DropdownMenuTrigger, DropdownMenuContent, DropdownMenuItem, DropdownMenuCheckboxItem, DropdownMenuRadioGroup, DropdownMenuRadioItem, DropdownMenuLabel, DropdownMenuSeparator, DropdownMenuSub, DropdownMenuSubTrigger, DropdownMenuSubContent, DropdownMenuGroup } from './components/ui/dropdown-menu'

// Collapsible
export { Collapsible, CollapsibleTrigger, CollapsibleContent } from './components/ui/collapsible'

// Toast + hook
export {
  ToastProvider,
  ToastViewport,
  Toast,
  ToastTitle,
  ToastDescription,
  ToastClose,
  ToastAction,
  useToast,
  type ToastOptions,
  type ToastVariant,
} from './components/ui/toast'

// Tooltip
export {
  Tooltip,
  TooltipTrigger,
  TooltipContent,
  SimpleTooltip,
} from './components/ui/tooltip'

// Popover
export {
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverClose,
  PopoverAnchor,
} from './components/ui/popover'

// Display components
export { StatCard } from './components/ui/stat-card'
export { KpiGrid } from './components/ui/kpi-grid'
export { ListRow } from './components/ui/list-row'
export { ListSection } from './components/ui/list-section'
export { EmptyCard } from './components/ui/empty-card'
export { Skeleton } from './components/ui/skeleton'
export { LoadingDots } from './components/ui/loading-dots'

// Specialized components
export { Avatar, AvatarGroup } from './components/ui/avatar'
export { Spinner } from './components/ui/spinner'
export { Divider } from './components/ui/divider'
export { CardDeck } from './components/ui/card-deck'
export { Pagination } from './components/ui/pagination'
export { SearchField } from './components/ui/search-field'
export { EmptyState } from './components/ui/empty-state'

// Tags
export { Chip, Chips, TagInput } from './components/ui/tags'

// Icons (50+ from our components/ui/icons)
export {
  IconSearch, IconPlus, IconChevronLeft, IconChevronDown, IconChevronRight, IconChevronUp,
  IconMenu, IconX, IconCheck, IconChat, IconStar, IconPin, IconClock,
  IconSettings, IconCopy, IconRefresh, IconTrash, IconEdit, IconMessage,
  IconSend, IconUser, IconHome, IconCog, IconFolder, IconDocument,
  IconDownload, IconUpload, IconExternalLink, IconModel, IconModels, IconBrain,
  IconHeart, IconThumbUp, IconThumbDown, IconInfo, IconAlert,
  IconCheckCircle, IconError, IconEye, IconFilter, IconSort, IconMore,
  IconMoon, IconSun, IconActivity, IconCompare, IconTraining, IconBenchmark,
  IconTokenizer, IconExport, IconLabs, IconAgents, IconLogin, IconVision,
  IconFile, IconChangelog,
  IconPlay, IconMapPin, IconBolt, IconChart, IconDot, IconDotOutline,
  IconSparkle, IconCode, IconBeaker, IconBookmark, IconSpeaker,
  IconMic, IconMicFilled, IconStop, IconImage, IconGrid, IconCloudOff,
  IconRewrite, IconExplain, IconTranslate,
  AudioWaveform, LossCurve,
} from './components/ui/icons'

// Error store + panel
export { ErrorPanel } from './components/ui/error-panel'
export { useErrorStore, type AppError, type ErrorSeverity } from './lib/error-store'

// === Composed components (non-overlapping with our UI) ===
export { PageHeader, type PageHeaderProps } from './components/composed/page-header'
export { AppShell, type AppShellProps } from './components/composed/app-shell'
export { NavRail, NavRailLink, type NavRailProps, type NavRailLinkProps } from './components/composed/nav-rail'
export { FormField, type FormFieldProps } from './components/composed/form-field'
export { SettingsRow, type SettingsRowProps } from './components/composed/settings-row'
export { InlineBanner, bannerVariants, type InlineBannerProps } from './components/composed/inline-banner'
export { Toolbar, type ToolbarProps } from './components/composed/toolbar'
export { CopyButton, type CopyButtonProps } from './components/composed/copy-button'
export { KeyValueList, type KeyValueListProps, type KeyValueItem } from './components/composed/key-value-list'
export { ProgressBar, type ProgressBarProps } from './components/composed/progress-bar'
export { StepIndicator, type StepIndicatorProps } from './components/composed/step-indicator'
export { SectionHeader, type SectionHeaderProps } from './components/composed/section-header'
export { ScrollPanel, type ScrollPanelProps } from './components/composed/scroll-panel'
export { StatusDot, STATUS_DOT_TONE_CLASSES, type StatusDotProps, type StatusDotTone } from './components/composed/status-dot'
export { ModelStatusPill, type ModelStatusProps, type ModelStatus } from './components/composed/model-status'
export { FoldSection, type FoldSectionProps } from './components/composed/fold-section'
export { Timeline, type TimelineProps, type TimelineItem } from './components/composed/timeline'
export { Breadcrumbs, type BreadcrumbsProps, type BreadcrumbItem } from './components/composed/breadcrumbs'
export { Kbd, type KbdProps } from './components/composed/kbd'
export { ThemeColorPicker, ThemeSwatch, ColorInput, DEFAULT_THEME_SWATCHES, type ThemeSwatch as ThemeSwatchType, type ThemeColorPickerProps } from './components/composed/theme-color-picker'
export { VersionInspector, type VersionInspectorProps, type FeatureVersionEntry } from './components/composed/version-inspector'
export { ActionCard, type ActionCardProps } from './components/composed/action-card'
export { InsightsCard, type InsightsCardProps, type InsightsCardDetail } from './components/composed/insights-card'

// === AI components (non-overlapping) ===
export { MessageBubble, type MessageBubbleProps } from './components/ai/message-bubble'
export { ChatThread, type ChatThreadProps } from './components/ai/chat-thread'
export { PromptComposer, type PromptComposerProps } from './components/ai/prompt-composer'
export { TypingIndicator, type TypingIndicatorProps } from './components/ai/typing-indicator'
export { ToolCallCard, type ToolCallCardProps, type ToolCallState } from './components/ai/tool-call-card'
export { CodeSnippet, type CodeSnippetProps } from './components/ai/code-snippet'
export { TokenMeter, type TokenMeterProps } from './components/ai/token-meter'
export { ChatLayout, type ChatLayoutProps } from './components/ai/chat-layout'
export { ModelPicker, type ModelPickerProps, type ModelOption } from './components/ai/model-picker'
export { Citation, type CitationProps } from './components/ai/citation'
export { ReasoningPanel, type ReasoningPanelProps } from './components/ai/reasoning-panel'
export { AttachmentChip, type AttachmentChipProps } from './components/ai/attachment-chip'
export { SourceList, type SourceListProps, type SourceItem } from './components/ai/source-list'
export { StreamingAssistantPlaceholder, type StreamingAssistantPlaceholderProps } from './components/ai/streaming-assistant-placeholder'
export { JobStatus, type JobStatusProps, type JobStatusState } from './components/ai/job-status'
