import type {StackScreenProps} from '@react-navigation/stack';

export type SettingsStackParamList = {
  SettingsMain: undefined;
  Health: undefined;
  About: undefined;
  Bookmarks: undefined;
  Help: undefined;
  Training: undefined;
  Knowledge: undefined;
  Search: undefined;
  Providers: undefined;
  Souls: undefined;
  Datasets: undefined;
  DatasetDetail: {datasetId: string};
  Notifications: undefined;
  WhatsNew: undefined;
  Legal: undefined;
  Adapters: undefined;
  Feedback: undefined;
  Voice: undefined;
  Companion: undefined;
  Learn: undefined;
  Agents: undefined;
  Multimodal: undefined;
  ModelDetail: {modelId: string};
  Images: undefined;
  Memory: undefined;
  Auth: undefined;
};

export type ToolsStackParamList = {
  ToolsMain: undefined;
  Training: undefined;
  Knowledge: undefined;
  Bookmarks: undefined;
  Search: undefined;
  Health: undefined;
  Souls: undefined;
  Datasets: undefined;
  DatasetDetail: {datasetId: string};
  Notifications: undefined;
  WhatsNew: undefined;
  Legal: undefined;
  Adapters: undefined;
  Feedback: undefined;
  Voice: undefined;
  Companion: undefined;
  Learn: undefined;
  Agents: undefined;
  Multimodal: undefined;
  ModelDetail: {modelId: string};
  Images: undefined;
  Memory: undefined;
  Auth: undefined;
};

export type SettingsScreenProps<T extends keyof SettingsStackParamList> =
  StackScreenProps<SettingsStackParamList, T>;

export type ToolsScreenProps<T extends keyof ToolsStackParamList> =
  StackScreenProps<ToolsStackParamList, T>;
