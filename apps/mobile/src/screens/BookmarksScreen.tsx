import React, {useEffect, useState} from 'react';
import {FlatList, Alert} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {YStack, XStack, Text, useTheme} from 'tamagui';
import {useChatStore} from '../stores/chat-store';
import {getBookmarks, removeBookmark, type Bookmark} from '../services/bookmarks';
import {triggerHaptic} from '../services/haptics';
import {Icon} from '../components/Icon';

export function BookmarksScreen() {
  const theme = useTheme();
  const {loadSession} = useChatStore();
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);

  useEffect(() => {
    getBookmarks().then(setBookmarks);
  }, []);

  const handleRemove = async (id: string) => {
    Alert.alert('Remove bookmark', 'Remove this bookmark?', [
      {text: 'Cancel', style: 'cancel'},
      {
        text: 'Remove',
        style: 'destructive',
        onPress: async () => {
          await removeBookmark(id);
          setBookmarks(prev => prev.filter(b => b.id !== id));
          triggerHaptic('light');
        },
      },
    ]);
  };

  const handleOpen = async (bookmark: Bookmark) => {
    await loadSession(bookmark.sessionId);
  };

  return (
    <SafeAreaView style={{flex: 1}} edges={['top']}>
      <YStack flex={1}>
        <YStack paddingHorizontal={16} paddingTop={12} paddingBottom={8}>
          <Text fontSize={20} fontWeight="600" letterSpacing={-0.2} color="$color">
            Bookmarks
          </Text>
          <Text fontSize={11} fontWeight="500" color="$color10" marginTop={2}>
            {bookmarks.length} saved messages
          </Text>
        </YStack>

        <FlatList
          data={bookmarks}
          keyExtractor={b => b.id}
          contentContainerStyle={{paddingHorizontal: 16, paddingTop: 12, gap: 8}}
          renderItem={({item: b}) => (
            <YStack
              backgroundColor="$background"
              borderRadius={12}
              padding={14}
              borderWidth={0.5}
              borderColor="$borderColor"
              onPress={() => handleOpen(b)}
              onLongPress={() => handleRemove(b.id)}>
              <XStack justifyContent="space-between" alignItems="center" marginBottom={4}>
                <Text fontSize={11} fontWeight="600" color="$color9">
                  {b.role === 'user' ? 'You' : 'AI'}
                </Text>
                <Text fontSize={11} fontWeight="500" color="$color10">
                  {new Date(b.savedAt).toLocaleDateString()}
                </Text>
              </XStack>
              <Text fontSize={14} color="$color" lineHeight={20} numberOfLines={3}>
                {b.content}
              </Text>
            </YStack>
          )}
          ListEmptyComponent={
            <YStack alignItems="center" paddingTop={96} paddingHorizontal={32}>
              <Icon name="star" size={48} color={(theme.color10?.val || '#9B95A8')} />
              <Text fontSize={20} fontWeight="600" color="$color" marginBottom={8} textAlign="center">
                No bookmarks yet
              </Text>
              <Text fontSize={14} color="$color11" textAlign="center">
                Long-press any message and tap Bookmark to save it here.
              </Text>
            </YStack>
          }
        />
      </YStack>
    </SafeAreaView>
  );
}
