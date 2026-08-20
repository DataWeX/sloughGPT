import React, {createContext, useContext, useState, useCallback} from 'react';

interface SidebarContextType {
  visible: boolean;
  open: () => void;
  close: () => void;
  toggle: () => void;
  activeScreen: string;
  navigate: (screen: string) => void;
}

const SidebarCtx = createContext<SidebarContextType>({
  visible: false,
  open: () => {},
  close: () => {},
  toggle: () => {},
  activeScreen: 'Chat',
  navigate: () => {},
});

export function SidebarProvider({children}: {children: React.ReactNode}) {
  const [visible, setVisible] = useState(false);
  const [activeScreen, setActiveScreen] = useState('Chat');

  const open = useCallback(() => setVisible(true), []);
  const close = useCallback(() => setVisible(false), []);
  const toggle = useCallback(() => setVisible(v => !v), []);
  const navigate = useCallback((screen: string) => {
    setActiveScreen(screen);
    setVisible(false);
  }, []);

  return (
    <SidebarCtx.Provider value={{visible, open, close, toggle, activeScreen, navigate}}>
      {children}
    </SidebarCtx.Provider>
  );
}

export const useSidebar = () => useContext(SidebarCtx);
