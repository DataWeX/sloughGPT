import React, {useEffect, useState} from 'react';
import {LogBox, StatusBar, useColorScheme} from 'react-native';

if (__DEV__) {
  LogBox.ignoreAllLogs();
}
import {NavigationContainer, DefaultTheme, DarkTheme} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
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
import {useSettingsStore} from './src/stores/settings-store';
import {TamaguiProvider} from './src/theme/TamaguiProvider';
import {ErrorBoundary} from './src/components/ErrorBoundary';
import {Icon} from './src/components/Icon';
import {LoadingScreen} from './src/components/LoadingScreen';
import {ConnectionStatusBar} from './src/components/ConnectionStatusBar';
import {ToastContainer} from './src/components/ToastContainer';
import {OnboardingScreen, isFirstLaunch} from './src/screens/OnboardingScreen';
import {ToolsScreen} from './src/screens/ToolsScreen';
import {ALL_TABS} from './src/navigation/tabs';
import {registerForPushNotifications, onNotification} from './src/services/push-notifications';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const TAB_ICONS: Record<string, import('./src/components/Icon').IconName> = {
  Chat: 'message-circle',
  Models: 'brain',
  Tools: 'dumbbell',
  Settings: 'settings',
};

function TabIcon({name, focused}: {name: string; focused: boolean}) {
  return (
    <Icon
      name={TAB_ICONS[name] || 'square'}
      size={24}
      color={focused ? (useColorScheme() === 'dark' ? '#C0AAF4' : '#7C52C4') : (useColorScheme() === 'dark' ? '#6B6580' : '#9B95A8')}
    />
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
    </Stack.Navigator>
  );
}

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
    </Stack.Navigator>
  );
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
    return unsub;
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

  return (
    <SafeAreaProvider>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#110F18' : '#F8F6FC'} />
        <NavigationContainer theme={navTheme}>
        <ConnectionStatusBar />
        <ToastContainer />
        <Tab.Navigator
          screenOptions={({route}) => ({
            headerShown: false,
            tabBarIcon: ({focused}) => (
              <TabIcon name={route.name} focused={focused} />
            ),
            tabBarActiveTintColor: isDark ? '#C0AAF4' : '#7C52C4',
            tabBarInactiveTintColor: isDark ? '#968CAC' : '#827A96',
            tabBarStyle: {
              backgroundColor: isDark ? '#1C1926' : '#FFFFFF',
              borderTopColor: isDark ? '#342E48' : '#E4E0F2',
              height: 56,
              paddingBottom: 6,
              paddingTop: 4,
            },
            tabBarLabelStyle: {
              fontSize: 11,
              fontWeight: '500',
            },
          })}>
          {ALL_TABS.map(tab =>
            tab.stack ? (
              <Tab.Screen
                key={tab.name}
                name={tab.name}
                component={tab.name === 'Tools' ? ToolsStack : SettingsStack}
                options={{headerShown: false}}
              />
            ) : (
              <Tab.Screen key={tab.name} name={tab.name} component={tab.component} />
            )
          )}
        </Tab.Navigator>
      </NavigationContainer>
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
