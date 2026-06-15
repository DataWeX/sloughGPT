import { YStack, Text, Button } from 'tamagui'
import { useRouter } from 'expo-router'

export default function ChatScreen() {
  const router = useRouter()
  return (
    <YStack flex={1} justifyContent="center" alignItems="center" padding="$4">
      <Text fontSize="$6" fontWeight="600">Chat</Text>
      <Text marginTop="$2" color="$placeholderColor" textAlign="center">
        Simple chat page
      </Text>
      <Button marginTop="$4" onPress={() => router.push('/knowledge')}>
        Go to Knowledge
      </Button>
    </YStack>
  )
}
