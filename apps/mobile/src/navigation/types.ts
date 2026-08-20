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
  Tokenizer: undefined;
  Compare: undefined;
  Datasets: undefined;
  DatasetDetail: {datasetId: string};
  Export: undefined;
  Import: undefined;
  Notifications: undefined;
  WhatsNew: undefined;
  Benchmark: undefined;
  Adapters: undefined;
  Feedback: undefined;
  Workflow: undefined;
  Voice: undefined;
  Companion: undefined;
  Learn: undefined;
  Agents: undefined;
  Multimodal: undefined;
  ModelDetail: {modelId: string};
  Errors: undefined;
  Security: undefined;
  Images: undefined;
  Experiments: undefined;
  Files: undefined;
  Registry: undefined;
  Memory: undefined;
};

export type ToolsStackParamList = {
  ToolsMain: undefined;
  Training: undefined;
  Knowledge: undefined;
  Bookmarks: undefined;
  Search: undefined;
  Health: undefined;
  Souls: undefined;
  Tokenizer: undefined;
  Compare: undefined;
  Datasets: undefined;
  DatasetDetail: {datasetId: string};
  Export: undefined;
  Import: undefined;
  Notifications: undefined;
  WhatsNew: undefined;
  Benchmark: undefined;
  Adapters: undefined;
  Feedback: undefined;
  Workflow: undefined;
  Voice: undefined;
  Companion: undefined;
  Learn: undefined;
  Agents: undefined;
  Multimodal: undefined;
  ModelDetail: {modelId: string};
  Errors: undefined;
  Security: undefined;
  Images: undefined;
  Experiments: undefined;
  Files: undefined;
  Registry: undefined;
  Memory: undefined;
};

export type SettingsScreenProps<T extends keyof SettingsStackParamList> =
  StackScreenProps<SettingsStackParamList, T>;

export type ToolsScreenProps<T extends keyof ToolsStackParamList> =
  StackScreenProps<ToolsStackParamList, T>;
