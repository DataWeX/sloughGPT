import type {ComponentType} from 'react';
import {HomeScreen} from '../screens/HomeScreen';
import {ChatScreen} from '../screens/ChatScreen';
import {ModelsScreen} from '../screens/ModelsScreen';
import {ToolsScreen} from '../screens/ToolsScreen';

export interface TabDefinition {
  name: string;
  label: string;
  icon: string;
  component: ComponentType<any>;
  stack?: boolean;
}

export const ALL_TABS: TabDefinition[] = [
  {name: 'Home', label: 'Home', icon: '🏠', component: HomeScreen},
  {name: 'Chat', label: 'Chat', icon: '💬', component: ChatScreen},
  {name: 'Models', label: 'Models', icon: '🧠', component: ModelsScreen},
  {name: 'Tools', label: 'Tools', icon: '🛠️', component: null as any, stack: true},
  {name: 'Settings', label: 'Settings', icon: '⚙️', component: null as any, stack: true},
];
