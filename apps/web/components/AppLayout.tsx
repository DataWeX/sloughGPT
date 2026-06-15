'use client';

import { useEffect, useState } from 'react';
import { useApiMonitor } from '@/lib/api-monitor-store';
import { useToastStore } from '@/lib/toast-store';
import { GlobalErrorHandler } from '@/components/GlobalErrorHandler';
import { ErrorPanel } from '@/components/ui/error-panel';
import { StatusBar } from '@/components/StatusBar';
import { ToastContainer } from '@/components/chat/Toast';
import { KeyboardShortcutsModal } from '@/components/KeyboardShortcutsModal';
import { DebugOverlay } from '@/components/DebugOverlay';
import { CommandPalette } from '@/components/CommandPalette';
import { useGlobalShortcuts } from '@/hooks/useGlobalShortcuts';
import { useBackendWatcher } from '@/hooks/useBackendWatcher';

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const apiStatus = useApiMonitor(s => s.status);
  const toasts = useToastStore(s => s.toasts);
  const dismissToast = useToastStore(s => s.dismissToast);
  const clearToasts = useToastStore(s => s.clearToasts);
  const [showShortcuts, setShowShortcuts] = useState(false);
  const [showDebug, setShowDebug] = useState(false);

  useGlobalShortcuts();
  useBackendWatcher();

  useEffect(() => {
    const handler = () => setShowShortcuts(true);
    window.addEventListener('toggle-shortcuts', handler);
    return () => window.removeEventListener('toggle-shortcuts', handler);
  }, []);

  return (
    <div className="flex h-dvh max-h-dvh flex-col bg-background pt-[env(safe-area-inset-top)]">
      {apiStatus === 'reloading' && (
        <div className="shrink-0 flex items-center justify-center gap-2 h-7 bg-warning/15 border-b border-warning/30 text-[11px] text-warning font-medium">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-warning animate-pulse" />
          Backend restarting — reconnecting…
        </div>
      )}
      <main id="main-content" className="flex-1 overflow-hidden">
        {children}
        <StatusBar />
      </main>
      <GlobalErrorHandler />
      <ErrorPanel />
      <ToastContainer toasts={toasts} onDismiss={dismissToast} onClearAll={clearToasts} />
      <KeyboardShortcutsModal open={showShortcuts} onOpenChange={setShowShortcuts} />
      <DebugOverlay open={showDebug} onOpenChange={setShowDebug} />
      <CommandPalette />
    </div>
  );
}
