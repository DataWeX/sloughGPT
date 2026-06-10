import { Tabs } from 'expo-router'
import { useTheme } from 'tamagui'
import { MessageCircle, Layers, BookOpen, Settings } from '@tamagui/lucide-icons'

export default function TabLayout() {
  const theme = useTheme()

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: String(theme.primary?.val ?? '#7C52C4'),
        tabBarInactiveTintColor: String(theme.placeholderColor?.val ?? '#999999'),
        tabBarStyle: {
          backgroundColor: String(theme.background?.val ?? '#FFFFFF'),
          borderTopColor: String(theme.borderColor?.val ?? '#E0E0E0'),
        },
      }}
    >
      <Tabs.Screen
        name="chat"
        options={{
          title: 'Chat',
          tabBarIcon: ({ color, size }) => (
            <MessageCircle size={size} color={String(color)} />
          ),
        }}
      />
      <Tabs.Screen
        name="models"
        options={{
          title: 'Models',
          tabBarIcon: ({ color, size }) => (
            <Layers size={size} color={String(color)} />
          ),
        }}
      />
      <Tabs.Screen
        name="knowledge"
        options={{
          title: 'Knowledge',
          tabBarIcon: ({ color, size }) => (
            <BookOpen size={size} color={String(color)} />
          ),
        }}
      />
      <Tabs.Screen
        name="settings"
        options={{
          title: 'Settings',
          tabBarIcon: ({ color, size }) => (
            <Settings size={size} color={String(color)} />
          ),
        }}
      />
    </Tabs>
  )
}
