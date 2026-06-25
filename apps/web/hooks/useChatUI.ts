'use client'

import { useState, useCallback, useRef } from 'react'

export function useChatUI() {
  const [showSettings, setShowSettings] = useState(false)
  const [showConversationViewer, setShowConversationViewer] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [showConversationSearch, setShowConversationSearch] = useState(false)
  const [showMobileSearch, setShowMobileSearch] = useState(false)
  const [matchIndex, setMatchIndex] = useState(0)
  const [toolPanelOpen, setToolPanelOpen] = useState(true)
  const [voiceMode, setVoiceMode] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const chatScreenRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  const toggleSettings = useCallback(() => {
    setShowSettings(prev => !prev)
  }, [])

  const handleSearchChange = useCallback((value: string) => {
    setSearchQuery(value)
    setMatchIndex(0)
  }, [])

  const handleSearchClear = useCallback(() => {
    setSearchQuery('')
  }, [])

  return {
    showSettings, setShowSettings,
    showConversationViewer, setShowConversationViewer,
    searchQuery, setSearchQuery,
    showConversationSearch, setShowConversationSearch,
    showMobileSearch, setShowMobileSearch,
    matchIndex, setMatchIndex,
    toolPanelOpen, setToolPanelOpen,
    voiceMode, setVoiceMode,
    sidebarOpen, setSidebarOpen,
    chatScreenRef,
    searchInputRef,
    toggleSettings,
    handleSearchChange,
    handleSearchClear,
  }
}
