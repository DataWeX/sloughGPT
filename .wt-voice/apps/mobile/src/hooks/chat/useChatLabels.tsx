import {useState, useEffect, useCallback} from 'react';
import * as pinsService from '../../services/pins';
import * as starsService from '../../services/stars';
import * as labelsService from '../../services/labels';
import {getCachedActiveSessionId} from '../../services/offline-cache';
import {useChatStore} from '../../stores/chat-store';
import type {Session} from '../../types';

export function useChatLabels(
  sessions: Session[],
  activeSessionId: string | null,
  loadSession: (id: string) => Promise<void>,
  refreshSessions: () => Promise<void>,
) {
  const [labelFilter, setLabelFilter] = useState<string | null>(null);
  const [sessionLabels, setSessionLabels] = useState<Record<string, string[]>>({});
  const [allLabels, setAllLabels] = useState<string[]>([]);
  const [labelInput, setLabelInput] = useState('');
  const [starredIds, setStarredIds] = useState<string[]>([]);
  const [pinnedIds, setPinnedIds] = useState<string[]>([]);
  const [showArchived, setShowArchived] = useState(false);

  const safeSessions = sessions ?? [];
  const activeSessions = safeSessions.filter(s => !s.archived);
  const archivedSessions = safeSessions.filter(s => s.archived);

  useEffect(() => {
    if (activeSessionId) {
      pinsService.getPinnedIds(activeSessionId).then(setPinnedIds);
    } else {
      setPinnedIds([]);
    }
  }, [activeSessionId]);

  useEffect(() => {
    refreshSessions();
    starsService.getStarredIds().then(setStarredIds);
    labelsService.getAllDistinctLabels().then(setAllLabels);
    (async () => {
      const cachedId = await getCachedActiveSessionId();
      if (cachedId && !useChatStore.getState().activeSessionId) {
        await loadSession(cachedId);
      }
    })();
  }, []);

  const sortedActiveSessions = [...activeSessions].sort((a, b) => {
    const aStarred = starredIds.includes(a.id);
    const bStarred = starredIds.includes(b.id);
    if (aStarred && !bStarred) return -1;
    if (!aStarred && bStarred) return 1;
    if (aStarred && bStarred) {
      return starredIds.indexOf(a.id) - starredIds.indexOf(b.id);
    }
    return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
  });

  return {
    labelFilter,
    setLabelFilter,
    sessionLabels,
    setSessionLabels,
    allLabels,
    setAllLabels,
    labelInput,
    setLabelInput,
    starredIds,
    setStarredIds,
    pinnedIds,
    setPinnedIds,
    showArchived,
    setShowArchived,
    safeSessions,
    activeSessions,
    archivedSessions,
    sortedActiveSessions,
  };
}
