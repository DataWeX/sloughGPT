import '@testing-library/jest-dom/vitest'
import { vi } from 'vitest'

if (typeof window !== 'undefined') {
  class ResizeObserverMock {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  ;(window as any).ResizeObserver = ResizeObserverMock
}

vi.mock('@/lib/db', () => {
  const noop = async () => {}
  const noopVal = async <T = unknown>(..._args: unknown[]): Promise<T | undefined> => undefined
  const noopArr = async <T = unknown>(..._args: unknown[]): Promise<T[]> => []
  return {
    chatDB: {
      saveSession: noop,
      loadSessions: noopArr,
      loadSession: noopVal,
      deleteSession: noop,
      updateSession: noop,
      clearAllSessions: noop,
      getUnsyncedSessions: noopArr,
      markSynced: noop,
      savePendingMessage: noop,
      getPendingMessages: noopArr,
      deletePendingMessage: noop,
      clearPendingMessages: noop,
      searchAllSessions: noopArr,
      getKnowledge: noopArr,
      addKnowledge: noop,
      updateKnowledge: noop,
      deleteKnowledge: noop,
      clearKnowledge: noop,
      importKnowledge: noop,
      getBookmarks: noopArr,
      addBookmark: noop,
      removeBookmark: noop,
      clearBookmarks: noop,
      getPrompts: noopArr,
      savePrompt: noop,
      deletePrompt: noop,
      clearPrompts: noop,
      importPrompts: noop,
      getDraft: async () => '',
      saveDraft: noop,
      deleteDraft: noop,
      getKV: noopVal,
      setKV: noop,
      deleteKV: noop,
      addError: noop,
      getErrors: noopArr,
      clearErrors: noop,
    },
  }
})
