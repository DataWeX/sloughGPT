import React, {useEffect, useState, useCallback} from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import {SafeAreaView} from 'react-native-safe-area-context';
import {useChatStore} from '../stores/chat-store';
import {getBookmarks, removeBookmark, type Bookmark} from '../services/bookmarks';
import {triggerHaptic} from '../services/haptics';
import {colors, spacing, radii, typography} from '../theme';

export function BookmarksScreen() {
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
    <SafeAreaView style={styles.safe} edges={['top']}>
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.title}>Bookmarks</Text>
          <Text style={styles.subtitle}>{bookmarks.length} saved messages</Text>
        </View>

        <FlatList
          data={bookmarks}
          keyExtractor={b => b.id}
          contentContainerStyle={styles.list}
          renderItem={({item: b}) => (
            <TouchableOpacity
              style={styles.bookmarkItem}
              onPress={() => handleOpen(b)}
              onLongPress={() => handleRemove(b.id)}
              activeOpacity={0.7}>
              <View style={styles.bookmarkHeader}>
                <Text style={styles.role}>{b.role === 'user' ? 'You' : 'AI'}</Text>
                <Text style={styles.date}>
                  {new Date(b.savedAt).toLocaleDateString()}
                </Text>
              </View>
              <Text style={styles.content} numberOfLines={3}>
                {b.content}
              </Text>
            </TouchableOpacity>
          )}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyIcon}>☆</Text>
              <Text style={styles.emptyTitle}>No bookmarks yet</Text>
              <Text style={styles.emptySubtitle}>
                Long-press any message and tap Bookmark to save it here.
              </Text>
            </View>
          }
        />
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: colors.background,
  },
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    paddingBottom: spacing.sm,
  },
  title: {
    ...typography.h2,
    color: colors.text,
  },
  subtitle: {
    ...typography.small,
    color: colors.textMuted,
    marginTop: 2,
  },
  list: {
    paddingHorizontal: spacing.lg,
    paddingTop: spacing.md,
    gap: spacing.sm,
  },
  bookmarkItem: {
    backgroundColor: colors.surface,
    borderRadius: radii.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  bookmarkHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  role: {
    ...typography.small,
    color: colors.primary,
    fontWeight: '600',
  },
  date: {
    ...typography.small,
    color: colors.textMuted,
  },
  content: {
    ...typography.body,
    color: colors.text,
    lineHeight: 20,
  },
  emptyContainer: {
    alignItems: 'center',
    paddingTop: spacing.xxxl * 2,
    paddingHorizontal: spacing.xxxl,
  },
  emptyIcon: {
    fontSize: 48,
    color: colors.textMuted,
    marginBottom: spacing.md,
  },
  emptyTitle: {
    ...typography.h2,
    color: colors.text,
    marginBottom: spacing.sm,
  },
  emptySubtitle: {
    ...typography.body,
    color: colors.textSecondary,
    textAlign: 'center',
  },
});
