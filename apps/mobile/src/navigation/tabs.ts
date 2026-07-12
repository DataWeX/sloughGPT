import type {ComponentType} from 'react';
import {ChatScreen} from '../screens/ChatScreen';
import {ModelsScreen} from '../screens/ModelsScreen';

export interface TabDefinition {
  name: string;
  label: string;
  icon: string;
  component: ComponentType<any>;
  stack?: boolean;
}

export const ALL_TABS: TabDefinition[] = [
  {name: 'Chat', label: 'Chat', icon: '💬', component: ChatScreen},
  {name: 'Models', label: 'Models', icon: '🧠', component: ModelsScreen},
  {name: 'Settings', label: 'Settings', icon: '⚙️', component: null as any, stack: true},
];
