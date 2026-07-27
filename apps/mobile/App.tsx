import React, {useEffect, useState} from 'react';
import {StatusBar, Text, StyleSheet, useColorScheme} from 'react-native';
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
import {useSettingsStore} from './src/stores/settings-store';
import {TamaguiProvider} from './src/theme/TamaguiProvider';
import {ErrorBoundary} from './src/components/ErrorBoundary';
import {LoadingScreen} from './src/components/LoadingScreen';
import {ConnectionStatusBar} from './src/components/ConnectionStatusBar';
import {ToastContainer} from './src/components/ToastContainer';
import {OnboardingScreen, isFirstLaunch} from './src/screens/OnboardingScreen';
import {ALL_TABS} from './src/navigation/tabs';
import {registerForPushNotifications, onNotification} from './src/services/push-notifications';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const TAB_ICONS: Record<string, string> = {
  Chat: '💬',
  Models: '🧠',
  Settings: '⚙️',
};

function TabIcon({name, focused}: {name: string; focused: boolean}) {
  return (
    <Text style={{fontSize: 20, opacity: focused ? 1 : 0.5}}>
      {TAB_ICONS[name] || '•'}
    </Text>
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
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#0F0D18' : '#F5F0FF'} />
        <LoadingScreen />
      </SafeAreaProvider>
    );
  }

  if (needsOnboarding) {
    return (
      <SafeAreaProvider>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#0F0D18' : '#F5F0FF'} />
        <OnboardingScreen onComplete={() => setNeedsOnboarding(false)} />
      </SafeAreaProvider>
    );
  }

  const navTheme = isDark
    ? {
        ...DarkTheme,
        colors: {
          ...DarkTheme.colors,
          background: '#0F0D18',
          card: '#1A1730',
          text: '#F0ECF5',
          border: '#2D2A3A',
          primary: '#C0AAF4',
        },
      }
    : {
        ...DefaultTheme,
        colors: {
          ...DefaultTheme.colors,
          background: '#F5F0FF',
          card: '#FFFFFF',
          text: '#1A1625',
          border: '#E0DCE8',
          primary: '#7C52C4',
        },
      };

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#0F0D18' : '#F5F0FF'} />
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
            tabBarInactiveTintColor: isDark ? '#6B6580' : '#9B95A8',
            tabBarStyle: {
              backgroundColor: isDark ? '#1A1730' : '#FFFFFF',
              borderTopColor: isDark ? '#2D2A3A' : '#E0DCE8',
              height: 60,
              paddingBottom: 8,
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
                component={SettingsStack}
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
