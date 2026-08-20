import React, {useEffect, useState} from 'react';
import {LogBox, StatusBar, useColorScheme} from 'react-native';

try {
  const Screens = require('react-native-screens');
  if (Screens.enableScreens) Screens.enableScreens(false);
} catch (_) {}

if (__DEV__) {
  LogBox.ignoreLogs([
    'Non-serializable values were found in the navigation state',
    'Sending `onAnimatedValueUpdate` with no listeners registered',
    'new NativeEventEmitter',
    'ViewManager',
  ]);
}
import {NavigationContainer, DefaultTheme, DarkTheme} from '@react-navigation/native';
import {createStackNavigator} from '@react-navigation/stack';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import {SettingsScreen} from './src/screens/SettingsScreen';
import {HealthScreen} from './src/screens/HealthScreen';
import {AboutScreen} from './src/screens/AboutScreen';
import {BookmarksScreen} from './src/screens/BookmarksScreen';
import {HelpScreen} from './src/screens/HelpScreen';
import {TrainingScreen} from './src/screens/TrainingScreen';
import {KnowledgeScreen} from './src/screens/KnowledgeScreen';
import {SearchScreen} from './src/screens/SearchScreen';
import {ProvidersScreen} from './src/screens/ProvidersScreen';
import {SoulsScreen} from './src/screens/SoulsScreen';
import {TokenizerScreen} from './src/screens/TokenizerScreen';
import {CompareScreen} from './src/screens/CompareScreen';
import {DatasetsScreen} from './src/screens/DatasetsScreen';
import {DatasetDetailScreen} from './src/screens/DatasetDetailScreen';
import {ExportScreen} from './src/screens/ExportScreen';
import {BenchmarkScreen} from './src/screens/BenchmarkScreen';
import {AdaptersScreen} from './src/screens/AdaptersScreen';
import {FeedbackScreen} from './src/screens/FeedbackScreen';
import {WorkflowScreen} from './src/screens/WorkflowScreen';
import {VoiceScreen} from './src/screens/VoiceScreen';
import {CompanionScreen} from './src/screens/CompanionScreen';
import {LearnScreen} from './src/screens/LearnScreen';
import {AgentsScreen} from './src/screens/AgentsScreen';
import {MultimodalScreen} from './src/screens/MultimodalScreen';
import {ModelDetailScreen} from './src/screens/ModelDetailScreen';
import {ErrorsScreen} from './src/screens/ErrorsScreen';
import {SecurityScreen} from './src/screens/SecurityScreen';
import {ImagesScreen} from './src/screens/ImagesScreen';
import {ExperimentsScreen} from './src/screens/ExperimentsScreen';
import {FilesScreen} from './src/screens/FilesScreen';
import {RegistryScreen} from './src/screens/RegistryScreen';
import {MemoryScreen} from './src/screens/MemoryScreen';
import {useSettingsStore} from './src/stores/settings-store';
import {TamaguiProvider} from './src/theme/TamaguiProvider';
import {ErrorBoundary} from './src/components/ErrorBoundary';
import {LoadingScreen} from './src/components/LoadingScreen';
import {ConnectionStatusBar} from './src/components/ConnectionStatusBar';
import {ToastContainer} from './src/components/ToastContainer';
import {OnboardingScreen, isFirstLaunch} from './src/screens/OnboardingScreen';
import {HomeScreen} from './src/screens/HomeScreen';
import {ChatScreen} from './src/screens/ChatScreen';
import {ModelsScreen} from './src/screens/ModelsScreen';
import {ToolsScreen} from './src/screens/ToolsScreen';
import {SidebarProvider, useSidebar} from './src/contexts/SidebarContext';
import {SidebarDrawer} from './src/components/SidebarDrawer';
import {registerForPushNotifications, onNotification, onNotificationResponse} from './src/services/push-notifications';
import {navigateTo} from './src/services/navigation';
import {Linking} from 'react-native';
import type {ToolsStackParamList, SettingsStackParamList} from './src/navigation/types';

const Stack = createStackNavigator();

function ToolsStack() {
  return (
    <Stack.Navigator screenOptions={{headerShown: false}}>
      <Stack.Screen name="ToolsMain" component={ToolsScreen} />
      <Stack.Screen name="Training" component={TrainingScreen} />
      <Stack.Screen name="Knowledge" component={KnowledgeScreen} />
      <Stack.Screen name="Bookmarks" component={BookmarksScreen} />
      <Stack.Screen name="Search" component={SearchScreen} />
      <Stack.Screen name="Health" component={HealthScreen} />
      <Stack.Screen name="Souls" component={SoulsScreen} />
      <Stack.Screen name="Tokenizer" component={TokenizerScreen} />
      <Stack.Screen name="Compare" component={CompareScreen} />
      <Stack.Screen name="Datasets" component={DatasetsScreen} />
      <Stack.Screen name="DatasetDetail" component={DatasetDetailScreen} />
      <Stack.Screen name="Export" component={ExportScreen} />
      <Stack.Screen name="Benchmark" component={BenchmarkScreen} />
      <Stack.Screen name="Adapters" component={AdaptersScreen} />
      <Stack.Screen name="Feedback" component={FeedbackScreen} />
      <Stack.Screen name="Workflow" component={WorkflowScreen} />
      <Stack.Screen name="Voice" component={VoiceScreen} />
      <Stack.Screen name="Companion" component={CompanionScreen} />
      <Stack.Screen name="Learn" component={LearnScreen} />
      <Stack.Screen name="Agents" component={AgentsScreen} />
      <Stack.Screen name="Multimodal" component={MultimodalScreen} />
      <Stack.Screen name="ModelDetail" component={ModelDetailScreen} />
      <Stack.Screen name="Errors" component={ErrorsScreen} />
      <Stack.Screen name="Security" component={SecurityScreen} />
      <Stack.Screen name="Images" component={ImagesScreen} />
      <Stack.Screen name="Experiments" component={ExperimentsScreen} />
      <Stack.Screen name="Files" component={FilesScreen} />
      <Stack.Screen name="Registry" component={RegistryScreen} />
      <Stack.Screen name="Memory" component={MemoryScreen} />
    </Stack.Navigator>
  );
}

function SettingsStack() {
  return (
    <Stack.Navigator screenOptions={{headerShown: false}}>
      <Stack.Screen name="SettingsMain" component={SettingsScreen} />
      <Stack.Screen name="Health" component={HealthScreen} />
      <Stack.Screen name="About" component={AboutScreen} />
      <Stack.Screen name="Bookmarks" component={BookmarksScreen} />
      <Stack.Screen name="Help" component={HelpScreen} />
      <Stack.Screen name="Training" component={TrainingScreen} />
      <Stack.Screen name="Knowledge" component={KnowledgeScreen} />
      <Stack.Screen name="Search" component={SearchScreen} />
      <Stack.Screen name="Providers" component={ProvidersScreen} />
      <Stack.Screen name="Souls" component={SoulsScreen} />
      <Stack.Screen name="Tokenizer" component={TokenizerScreen} />
      <Stack.Screen name="Compare" component={CompareScreen} />
      <Stack.Screen name="Datasets" component={DatasetsScreen} />
      <Stack.Screen name="DatasetDetail" component={DatasetDetailScreen} />
      <Stack.Screen name="Export" component={ExportScreen} />
      <Stack.Screen name="Benchmark" component={BenchmarkScreen} />
      <Stack.Screen name="Adapters" component={AdaptersScreen} />
      <Stack.Screen name="Feedback" component={FeedbackScreen} />
      <Stack.Screen name="Workflow" component={WorkflowScreen} />
      <Stack.Screen name="Voice" component={VoiceScreen} />
      <Stack.Screen name="Companion" component={CompanionScreen} />
      <Stack.Screen name="Learn" component={LearnScreen} />
      <Stack.Screen name="Agents" component={AgentsScreen} />
      <Stack.Screen name="Multimodal" component={MultimodalScreen} />
      <Stack.Screen name="ModelDetail" component={ModelDetailScreen} />
      <Stack.Screen name="Errors" component={ErrorsScreen} />
      <Stack.Screen name="Security" component={SecurityScreen} />
      <Stack.Screen name="Images" component={ImagesScreen} />
      <Stack.Screen name="Experiments" component={ExperimentsScreen} />
      <Stack.Screen name="Files" component={FilesScreen} />
      <Stack.Screen name="Registry" component={RegistryScreen} />
      <Stack.Screen name="Memory" component={MemoryScreen} />
    </Stack.Navigator>
  );
}

function MainContent() {
  const {activeScreen} = useSidebar();

  const screen = activeScreen.split('/')[0];
  const sub = activeScreen.split('/')[1];

  switch (screen) {
    case 'Home':
      return <HomeScreen />;
    case 'Chat':
      return <ChatScreen />;
    case 'Models':
      return <ModelsScreen />;
    case 'Tools':
      if (sub) {
        return <ToolsStack />;
      }
      return <ToolsStack />;
    case 'Settings':
      if (sub) {
        return <SettingsStack />;
      }
      return <SettingsStack />;
    default:
      return <ChatScreen />;
  }
}

function AppInner() {
  const isDark = useColorScheme() === 'dark';
  const [ready, setReady] = useState(false);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  useEffect(() => {
    setReady(true);
    isFirstLaunch().then(setNeedsOnboarding).catch(() => {});
    registerForPushNotifications().catch(() => {});
    const unsub = onNotification((_title, _body, _data) => {
      if (__DEV__) console.log('[Push]', _title, _body);
    });
    const unsubResponse = onNotificationResponse((data) => {
      const screen = data?.screen;
      if (screen) navigateTo(screen);
    });
    // Handle URL schemes (sloughgpt://chat)
    Linking.getInitialURL().then((url) => {
      if (url) {
        const path = url.replace('sloughgpt://', '').replace('https://sloughgpt.app/', '');
        if (path) navigateTo(path);
      }
    }).catch(() => {});
    // Listen for URLs while app is open
    const linkingSub = Linking.addEventListener('url', (event) => {
      const path = event.url.replace('sloughgpt://', '').replace('https://sloughgpt.app/', '');
      if (path) navigateTo(path);
    });
    return () => { unsub(); unsubResponse(); linkingSub?.remove(); };
  }, []);

  if (!ready) {
    return (
      <SafeAreaProvider>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#110F18' : '#F8F6FC'} />
        <LoadingScreen />
      </SafeAreaProvider>
    );
  }

  if (needsOnboarding) {
    return (
      <SafeAreaProvider>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#110F18' : '#F8F6FC'} />
        <OnboardingScreen onComplete={() => setNeedsOnboarding(false)} />
      </SafeAreaProvider>
    );
  }

  const navTheme = isDark
    ? {
        ...DarkTheme,
        colors: {
          ...DarkTheme.colors,
          background: '#110F18',
          card: '#1C1926',
          text: '#F0ECF5',
          border: '#342E48',
          primary: '#C0AAF4',
        },
      }
    : {
        ...DefaultTheme,
        colors: {
          ...DefaultTheme.colors,
          background: '#F8F6FC',
          card: '#FFFFFF',
          text: '#1A1625',
          border: '#E4E0F2',
          primary: '#7C52C4',
        },
      };

  const linking = {
    prefixes: ['sloughgpt://', 'https://sloughgpt.app'],
    config: {
      screens: {
        Home: '',
        Chat: 'chat',
        Models: 'models',
        Tools: 'tools',
        Settings: 'settings',
      },
    },
    async getInitialURL() {
      const url = await Linking.getInitialURL();
      if (url != null) return url;
      return '';
    },
  };

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#110F18' : '#F8F6FC'} />
      <SidebarProvider>
        <NavigationContainer theme={navTheme} linking={linking}>
          <ConnectionStatusBar />
          <ToastContainer />
          <MainContent />
          <SidebarDrawer />
        </NavigationContainer>
      </SidebarProvider>
    </SafeAreaProvider>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <TamaguiProvider>
        <AppInner />
      </TamaguiProvider>
    </ErrorBoundary>
  );
}
