import React, { useState, useCallback } from 'react'
import { TextInput } from 'react-native'
import { XStack, YStack, Text } from 'tamagui'
import { useColors } from '../theme/colors'
import { triggerHaptic } from '../services/haptics'

interface ContextInjectionBarProps {
  onInject: (context: string) => void
  disabled?: boolean
}

export function ContextInjectionBar({ onInject, disabled = false }: ContextInjectionBarProps) {
  const colors = useColors()
  const [context, setContext] = useState('')
  const [showInput, setShowInput] = useState(false)

  const inputStyle = {
    flex: 1,
    fontSize: 13,
    color: colors.text,
    backgroundColor: colors.muted,
    borderRadius: 8,
    paddingHorizontal: 8,
    paddingVertical: 6,
    borderWidth: 1,
    borderColor: colors.border,
  } as const

  const handleInject = useCallback(() => {
    const trimmed = context.trim()
    if (!trimmed) {
      return
    }
    onInject(trimmed)
    triggerHaptic('light')
    setContext('')
    setShowInput(false)
  }, [context, onInject])

  const handleCancel = useCallback(() => {
    setShowInput(false)
    setContext('')
  }, [])

  if (!showInput) {
    return (
      <YStack
        testID="inject-context-trigger"
        accessibilityRole="button"
        accessibilityLabel="Inject context"
        accessibilityState={{ disabled }}
        paddingHorizontal={10}
        paddingVertical={6}
        alignSelf="flex-start"
        disabled={disabled}
        opacity={disabled ? 0.5 : 1}
        onPress={() => setShowInput(true)}
        pressStyle={{ opacity: 0.6 }}
      >
        <Text fontSize={11} color="$color10">
          + Context
        </Text>
      </YStack>
    )
  }

  return (
    <XStack paddingHorizontal={12} paddingVertical={6} alignItems="center" gap={8}>
      <TextInput
        testID="context-input"
        value={context}
        onChangeText={setContext}
        placeholder="Add context..."
        placeholderTextColor={colors.textMuted}
        autoFocus
        autoCapitalize="none"
        style={inputStyle}
      />
      <XStack
        testID="context-send"
        paddingHorizontal={10}
        paddingVertical={6}
        borderRadius={8}
        backgroundColor={context.trim() ? colors.primaryAlpha(0.12) : colors.primaryAlpha(0.05)}
        disabled={!context.trim()}
        onPress={handleInject}
        pressStyle={{ opacity: 0.6 }}
      >
        <Text
          fontSize={12}
          fontWeight="600"
          color={context.trim() ? colors.primary : colors.textMuted}
        >
          Send
        </Text>
      </XStack>
      <XStack
        testID="context-cancel"
        paddingHorizontal={10}
        paddingVertical={6}
        borderRadius={8}
        onPress={handleCancel}
        pressStyle={{ opacity: 0.6 }}
      >
        <Text fontSize={12} color="$color10">
          Cancel
        </Text>
      </XStack>
    </XStack>
  )
}
