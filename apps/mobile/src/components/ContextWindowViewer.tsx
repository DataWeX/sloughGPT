import React, { useState, useCallback, useMemo } from 'react'
import { ScrollView } from 'react-native'
import { XStack, YStack, Text } from 'tamagui'
import { useColors } from '../theme/colors'
import { Icon } from './Icon'
import { copyToClipboard } from '../services/clipboard'
import { toast } from '../services/toast'
import { estimateTokens, type ContextWindowItem } from '../services/context-items'

export type { ContextWindowItem }

const TYPE_LABELS: Record<ContextWindowItem['type'], string> = {
  system: 'System Prompt',
  knowledge: 'Knowledge Base',
  agent: 'Agent Instructions',
  user: 'User Message',
  assistant: 'Assistant Response',
}

/** Tone color per item type — all keys exist in the useColors() palette. */
const TYPE_TONES: Record<
  ContextWindowItem['type'],
  'primary' | 'accent' | 'warning' | 'success' | 'error'
> = {
  system: 'primary',
  knowledge: 'accent',
  agent: 'warning',
  user: 'success',
  assistant: 'error',
}

interface ContextWindowViewerProps {
  items: ContextWindowItem[]
  totalTokens?: number
  onRefresh?: () => void
}

function ItemCard({
  item,
  isExpanded,
  onToggle,
}: {
  item: ContextWindowItem
  isExpanded: boolean
  onToggle: () => void
}) {
  const colors = useColors()
  const [copied, setCopied] = useState(false)
  const toneColor = colors[TYPE_TONES[item.type]]

  const handleCopy = useCallback(async () => {
    const ok = await copyToClipboard(item.content)
    if (ok) {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } else {
      toast.error('Failed to copy')
    }
  }, [item.content])

  return (
    <YStack
      borderWidth={1}
      borderColor={colors.border}
      borderRadius={10}
      overflow="hidden"
      marginBottom={6}
    >
      <XStack
        testID={`context-item-${item.label}`}
        paddingHorizontal={10}
        paddingVertical={8}
        alignItems="center"
        justifyContent="space-between"
        onPress={onToggle}
        pressStyle={{ backgroundColor: colors.backgroundHover }}
      >
        <XStack alignItems="center" gap={6} flex={1}>
          <YStack
            paddingHorizontal={6}
            paddingVertical={2}
            borderRadius={5}
            borderWidth={1}
            borderColor={colors.border}
            backgroundColor={colors.secondary}
          >
            <Text fontSize={9} fontWeight="600" color={toneColor}>
              {TYPE_LABELS[item.type]}
            </Text>
          </YStack>
          <Text fontSize={12} color="$color" numberOfLines={1} flex={1}>
            {item.label}
          </Text>
        </XStack>
        <XStack alignItems="center" gap={6}>
          {item.tokenCount ? (
            <Text fontSize={9} color="$color10">
              ~{item.tokenCount} tokens
            </Text>
          ) : null}
          <Icon
            name={isExpanded ? 'chevron-up' : 'chevron-down'}
            size={12}
            color={colors.textMuted}
          />
        </XStack>
      </XStack>
      {isExpanded && (
        <YStack paddingHorizontal={10} paddingBottom={10} paddingTop={2}>
          <YStack
            backgroundColor={colors.primaryAlpha(0.05)}
            borderRadius={8}
            padding={8}
            position="relative"
          >
            <ScrollView style={{ maxHeight: 192 }} nestedScrollEnabled>
              <Text fontSize={11} color="$color10" fontFamily="monospace">
                {item.content}
              </Text>
            </ScrollView>
            <XStack
              testID="copy-context-item"
              position="absolute"
              top={6}
              right={6}
              width={24}
              height={24}
              borderRadius={6}
              alignItems="center"
              justifyContent="center"
              backgroundColor="$background"
              onPress={handleCopy}
              pressStyle={{ opacity: 0.6 }}
            >
              <Icon name={copied ? 'check' : 'copy'} size={12} color={colors.primary} />
            </XStack>
          </YStack>
        </YStack>
      )}
    </YStack>
  )
}

export function ContextWindowViewer({ items, totalTokens, onRefresh }: ContextWindowViewerProps) {
  const colors = useColors()
  const [expanded, setExpanded] = useState<Set<string>>(new Set())
  const [showAll, setShowAll] = useState(false)

  const estimatedTokens = useMemo(() => estimateTokens(items, totalTokens), [items, totalTokens])

  const toggleItem = useCallback((label: string) => {
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(label)) {
        next.delete(label)
      } else {
        next.add(label)
      }
      return next
    })
  }, [])

  const toggleAll = useCallback(() => {
    if (showAll) {
      setExpanded(new Set())
    } else {
      setExpanded(new Set(items.map((i) => i.label)))
    }
    setShowAll(!showAll)
  }, [showAll, items])

  if (items.length === 0) {
    return (
      <YStack padding={16} alignItems="center">
        <Text fontSize={12} color="$color10">
          No context items loaded
        </Text>
      </YStack>
    )
  }

  return (
    <YStack>
      <XStack justifyContent="space-between" alignItems="center" marginBottom={8}>
        <XStack alignItems="center" gap={6}>
          <Text fontSize={13} fontWeight="600" color="$color">
            Context Window
          </Text>
          <Text fontSize={10} color="$color10">
            {items.length} items · ~{estimatedTokens.toLocaleString()} tokens
          </Text>
        </XStack>
        <XStack gap={6}>
          {onRefresh && (
            <XStack
              paddingHorizontal={8}
              paddingVertical={4}
              borderRadius={6}
              backgroundColor={colors.primaryAlpha(0.08)}
              onPress={onRefresh}
              pressStyle={{ opacity: 0.6 }}
            >
              <Text fontSize={10} fontWeight="600" color={colors.primary}>
                Refresh
              </Text>
            </XStack>
          )}
          <XStack
            testID="context-toggle-all"
            paddingHorizontal={8}
            paddingVertical={4}
            borderRadius={6}
            backgroundColor={colors.primaryAlpha(0.08)}
            onPress={toggleAll}
            pressStyle={{ opacity: 0.6 }}
          >
            <Text fontSize={10} fontWeight="600" color={colors.primary}>
              {showAll ? 'Collapse All' : 'Expand All'}
            </Text>
          </XStack>
        </XStack>
      </XStack>
      {items.map((item) => (
        <ItemCard
          key={item.label}
          item={item}
          isExpanded={expanded.has(item.label)}
          onToggle={() => toggleItem(item.label)}
        />
      ))}
    </YStack>
  )
}
