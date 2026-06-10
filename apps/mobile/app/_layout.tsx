import { useEffect } from 'react'
import { View, Text, useColorScheme } from 'react-native'
import { useFonts } from 'expo-font'
import { Stack, useNavigationContainerRef } from 'expo-router'
import { TamaguiProvider, Theme } from 'tamagui'
import * as SplashScreen from 'expo-splash-screen'
import * as Sentry from '@sentry/react-native'
import config from '../tamagui.config'
import { PerformanceTracker } from '../lib/analytics'

// Initialize Sentry
import '../sentry.config'

SplashScreen.preventAutoHideAsync()

export default function RootLayout() {
  const colorScheme = useColorScheme()
  const themeName = colorScheme === 'dark' ? 'dark' : 'light'
  const navigationRef = useNavigationContainerRef()
  const [loaded] = useFonts({
    Outfit: require('../assets/fonts/Outfit-400.ttf'),
    OutfitMedium: require('../assets/fonts/Outfit-500.ttf'),
    OutfitSemibold: require('../assets/fonts/Outfit-600.ttf'),
    OutfitBold: require('../assets/fonts/Outfit-700.ttf'),
    JetBrainsMono: require('../assets/fonts/JetBrainsMono-400.ttf'),
    JetBrainsMonoMedium: require('../assets/fonts/JetBrainsMono-500.ttf'),
  })

  useEffect(() => {
    if (loaded) {
      SplashScreen.hideAsync()
    }
  }, [loaded])

  // Track navigation changes
  useEffect(() => {
    if (!navigationRef) return

    const unsubscribe = navigationRef.addListener('state', () => {
      const currentRoute = navigationRef.getCurrentRoute()
      if (currentRoute) {
        PerformanceTracker.trackNavigation('unknown', (currentRoute as any).name)
      }
    })

    return unsubscribe
  }, [navigationRef])

  if (!loaded) {
    return null
  }

  return (
    <Sentry.ErrorBoundary fallback={<ErrorFallback />}>
      <TamaguiProvider config={config as any} defaultTheme={themeName}>
        <Theme name={themeName}>
          <Stack screenOptions={{ headerShown: false }}>
            <Stack.Screen name="(auth)" />
            <Stack.Screen name="(tabs)" />
          </Stack>
        </Theme>
      </TamaguiProvider>
    </Sentry.ErrorBoundary>
  )
}

function ErrorFallback() {
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', padding: 20 }}>
      <Text style={{ fontSize: 18, fontWeight: 'bold', marginBottom: 10 }}>
        Something went wrong
      </Text>
      <Text style={{ textAlign: 'center', color: '#666' }}>
        We've been notified and are working to fix the issue.
      </Text>
    </View>
  )
}
