import {useState, useCallback} from 'react';

export interface ChatModals {
  showDrawer: boolean;
  showSoulPicker: boolean;
  showSettings: boolean;
  showChatSettings: boolean;
  showSystemPrompt: boolean;
  showSearch: boolean;
  showSearchSessions: boolean;
  showInfo: boolean;
}

export function useChatModals() {
  const [showDrawer, setShowDrawer] = useState(false);
  const [showSoulPicker, setShowSoulPicker] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [showChatSettings, setShowChatSettings] = useState(false);
  const [showSystemPrompt, setShowSystemPrompt] = useState(false);
  const [showSearch, setShowSearch] = useState(false);
  const [showSearchSessions, setShowSearchSessions] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  const dismissAllModals = useCallback(() => {
    setShowDrawer(false);
    setShowSoulPicker(false);
    setShowSettings(false);
    setShowSearch(false);
    setShowInfo(false);
  }, []);

  return {
    showDrawer, setShowDrawer,
    showSoulPicker, setShowSoulPicker,
    showSettings, setShowSettings,
    showChatSettings, setShowChatSettings,
    showSystemPrompt, setShowSystemPrompt,
    showSearch, setShowSearch,
    showSearchSessions, setShowSearchSessions,
    showInfo, setShowInfo,
    dismissAllModals,
  };
}
