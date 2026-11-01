import {useState, useCallback, useRef, useEffect} from 'react';
import {FlatList} from 'react-native';
import {useChatStore} from '../../stores/chat-store';
import {triggerHaptic} from '../../services/haptics';
import type {Message} from '../../types';

export function useChatScroll(
  flatListRef: React.RefObject<FlatList | null>,
  messages: Message[],
  streaming: boolean,
  activeSessionId: string | null,
  regenerate: (id: string) => void,
  recordFeedback: (id: string, positive: boolean) => void,
  deleteMessage: (id: string) => void,
  searchQuery: string,
  selectMode: boolean,
  selectedIds: Set<string>,
  toggleSelectMessage: (id: string) => void,
  setEditingMessage: (v: string | null) => void,
  setReplyTo: (m: Message | null) => void,
  setForwardTo: (m: Message | null) => void,
  setSelectMode: (v: boolean) => void,
  setSelectedIds: (v: Set<string>) => void,
) {
  const [atBottom, setAtBottom] = useState(true);
  const lastHeaderTap = useRef<number>(0);

  useEffect(() => {
    if (atBottom && messages.length > 0) {
      setTimeout(() => {
        flatListRef.current?.scrollToEnd({animated: true});
      }, 50);
    }
  }, [messages, atBottom]);

  const renderItem = useCallback(
    ({item}: {item: Message}) => {
      const {MessageBubble} = require('../../components/MessageBubble');
      return (
        <MessageBubble
          message={item}
          sessionId={activeSessionId || undefined}
          highlight={searchQuery ? item.content.toLowerCase().includes(searchQuery.toLowerCase()) : false}
          searchQuery={searchQuery || undefined}
          onRegenerate={
            item.role === 'assistant' ? () => regenerate(item.id) : undefined
          }
          onFeedback={
            item.role === 'assistant'
              ? (positive: boolean) => recordFeedback(item.id, positive)
              : undefined
          }
          onDelete={() => deleteMessage(item.id)}
          onEdit={
            item.role === 'user' ? (newContent: string) => setEditingMessage(newContent) : undefined
          }
          onReply={() => setReplyTo(item)}
          onForward={() => setForwardTo(item)}
          selectMode={selectMode}
          selected={selectedIds.has(item.id)}
          onSelect={() => toggleSelectMessage(item.id)}
          onLongPressSelect={() => {
            if (!selectMode) {
              setSelectMode(true);
              setSelectedIds(new Set([item.id]));
            }
          }}
        />
      );
    },
    [regenerate, recordFeedback, searchQuery, deleteMessage, activeSessionId, selectMode, selectedIds, toggleSelectMessage],
  );

  const keyExtractor = useCallback((item: Message) => item.id, []);

  const onScroll = useCallback((e: any) => {
    const {contentOffset, contentSize, layoutMeasurement} = e.nativeEvent;
    const distFromBottom =
      contentSize.height - layoutMeasurement.height - contentOffset.y;
    setAtBottom(distFromBottom < 50);
  }, []);

  return {
    atBottom,
    setAtBottom,
    lastHeaderTap,
    renderItem,
    keyExtractor,
    onScroll,
  };
}
