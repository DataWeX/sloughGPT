import {useState, useCallback} from 'react';
import {triggerHaptic} from '../../services/haptics';

export function useMessageSelect(deleteMessage: (id: string) => void) {
  const [selectMode, setSelectMode] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const toggleSelectMode = useCallback(() => {
    setSelectMode(s => !s);
    setSelectedIds(new Set());
  }, []);

  const toggleSelectMessage = useCallback((id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const deleteSelected = useCallback(async () => {
    for (const id of selectedIds) deleteMessage(id);
    setSelectedIds(new Set());
    setSelectMode(false);
    triggerHaptic('medium');
  }, [selectedIds, deleteMessage]);

  return {
    selectMode,
    setSelectMode,
    selectedIds,
    setSelectedIds,
    toggleSelectMode,
    toggleSelectMessage,
    deleteSelected,
  };
}
