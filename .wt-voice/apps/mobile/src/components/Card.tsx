import type { ReactNode } from 'react'
import { YStack } from 'tamagui'
import { useColors } from '../theme/colors'

interface CardProps {
  children: ReactNode
  variant?: 'default' | 'compact' | 'elevated'
  style?: Record<string, unknown>
  [key: string]: unknown
}

/**
 * Shared card component for mobile screens.
 * - default: borderRadius=16, borderWidth=1, padding=16
 * - compact: borderRadius=12, borderWidth=0.5, padding=14
 * - elevated: like default + shadow
 */
export function Card({ children, variant = 'default', style, ...rest }: CardProps) {
  const colors = useColors()
  const isCompact = variant === 'compact'
  const isElevated = variant === 'elevated'

  return (
    <YStack
      padding={isCompact ? 14 : 16}
      borderRadius={isCompact ? 12 : 16}
      backgroundColor={colors.white}
      borderWidth={isCompact ? 0.5 : 1}
      borderColor={colors.border}
      gap={isElevated ? 12 : 10}
      {...(isElevated ? {
        shadowColor: 'black',
        shadowOffset: { width: 0, height: 2 },
        shadowOpacity: 0.06,
        shadowRadius: 8,
        elevation: 2,
      } : {})}
      {...style}
      {...rest}
    >
      {children}
    </YStack>
  )
}
