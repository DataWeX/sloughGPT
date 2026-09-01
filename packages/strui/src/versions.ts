/**
 * Strui Component Versioning System
 *
 * Maps every strui component to its corresponding backend feature version.
 * Used for tracking component ↔ backend feature alignment across the stack.
 *
 * Convention:
 *   - Major bump: breaking API change in the component OR backend endpoint
 *   - Minor bump: new feature/prop added to component OR new backend endpoint
 *   - Patch bump: bugfix only
 */

export const STRUI_VERSION = '0.2.0' as const

/**
 * Each entry maps a feature domain to its version and the strui components it owns.
 * The `backend` field is the backend router prefix; `api` is the version string
 * returned by the backend health endpoint for that feature.
 */
export interface FeatureVersion {
  /** Backend router prefix (e.g. "/models", "/training") */
  backend: string
  /** Semantic version of the backend feature */
  api: string
  /** Strui components that consume this feature */
  components: string[]
}

export const FEATURE_VERSIONS: Record<string, FeatureVersion> = {
  core: {
    backend: '/',
    api: '1.0.0',
    components: [
      'Button', 'Badge', 'Card', 'Input', 'Textarea', 'Label', 'Switch',
      'Separator', 'Checkbox', 'Radio', 'RadioGroup', 'Progress', 'Select',
      'Tabs', 'ToggleGroup', 'Slider', 'Dialog', 'AlertDialog', 'DropdownMenu',
      'Collapsible', 'Toast', 'Tooltip', 'Popover', 'Skeleton', 'LoadingDots',
      'Avatar', 'Spinner', 'Divider', 'CardDeck', 'Pagination', 'SearchField',
      'EmptyState', 'Chip', 'Chips', 'TagInput', 'ErrorPanel', 'StatCard',
      'KpiGrid', 'ListRow', 'ListSection', 'EmptyCard',
    ],
  },
  chat: {
    backend: '/',
    api: '1.0.0',
    components: [
      'MessageBubble', 'ChatThread', 'PromptComposer', 'TypingIndicator',
      'StreamingAssistantPlaceholder', 'ChatLayout', 'CodeSnippet',
      'ReasoningPanel', 'AttachmentChip', 'SourceList', 'Citation',
    ],
  },
  models: {
    backend: '/models',
    api: '1.0.0',
    components: ['ModelPicker', 'ModelStatusPill'],
  },
  training: {
    backend: '/training',
    api: '1.0.0',
    components: ['JobStatus', 'TokenMeter'],
  },
  tools: {
    backend: '/agents',
    api: '1.0.0',
    components: ['ToolCallCard'],
  },
  knowledge: {
    backend: '/knowledge',
    api: '1.0.0',
    components: ['Citation', 'SourceList'],
  },
  layout: {
    backend: '/system',
    api: '1.0.0',
    components: [
      'AppShell', 'NavRail', 'PageHeader', 'FormField', 'SettingsRow',
      'InlineBanner', 'Toolbar', 'CopyButton', 'KeyValueList', 'ProgressBar',
      'StepIndicator', 'SectionHeader', 'ScrollPanel', 'StatusDot', 'FoldSection',
      'Timeline', 'Breadcrumbs', 'Kbd', 'ThemeColorPicker',
    ],
  },
  health: {
    backend: '/health',
    api: '1.0.0',
    components: ['StatusDot', 'StatCard', 'KpiGrid'],
  },
  icons: {
    backend: '',
    api: '0.0.0',
    components: [
      'IconSearch', 'IconPlus', 'IconChevronLeft', 'IconChevronDown',
      'IconChevronRight', 'IconChevronUp', 'IconMenu', 'IconX', 'IconCheck',
      'IconChat', 'IconStar', 'IconPin', 'IconClock', 'IconSettings',
      'IconCopy', 'IconRefresh', 'IconTrash', 'IconEdit', 'IconMessage',
      'IconSend', 'IconUser', 'IconHome', 'IconCog', 'IconFolder',
      'IconDocument', 'IconDownload', 'IconUpload', 'IconExternalLink',
      'IconModel', 'IconModels', 'IconBrain', 'IconHeart', 'IconThumbUp',
      'IconThumbDown', 'IconInfo', 'IconAlert', 'IconCheckCircle', 'IconError',
      'IconEye', 'IconFilter', 'IconSort', 'IconMore', 'IconMoon', 'IconSun',
      'IconActivity', 'IconCompare', 'IconTraining', 'IconBenchmark',
      'IconTokenizer', 'IconExport', 'IconLabs', 'IconAgents', 'IconLogin',
      'IconVision', 'IconFile', 'IconChangelog', 'IconPlay', 'IconMapPin',
      'IconBolt', 'IconChart', 'IconDot', 'IconDotOutline', 'IconSparkle',
      'IconCode', 'IconBeaker', 'IconBookmark', 'IconSpeaker', 'IconMic',
      'IconMicFilled', 'IconStop', 'IconImage', 'IconGrid', 'IconCloudOff',
      'IconRewrite', 'IconExplain', 'IconTranslate',
    ],
  },
} as const

/** All known component names flattened */
export type FeatureName = keyof typeof FEATURE_VERSIONS

/**
 * Look up which feature a component belongs to.
 * Returns { feature, version } or null if unknown.
 */
export function getComponentFeature(componentName: string): { feature: string; version: string } | null {
  for (const [feature, info] of Object.entries(FEATURE_VERSIONS)) {
    if (info.components.includes(componentName)) {
      return { feature, version: info.api }
    }
  }
  return null
}

/**
 * Return all features as a flat list for display/debug.
 */
export function listFeatures(): Array<{ name: string; backend: string; api: string; componentCount: number }> {
  return Object.entries(FEATURE_VERSIONS).map(([name, info]) => ({
    name,
    backend: info.backend,
    api: info.api,
    componentCount: info.components.length,
  }))
}
