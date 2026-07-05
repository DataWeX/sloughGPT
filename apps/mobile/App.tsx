import React, {useEffect, useState} from 'react';
import {StatusBar, Text, View, StyleSheet} from 'react-native';
import {NavigationContainer, DefaultTheme, DarkTheme} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import {ChatScreen} from './src/screens/ChatScreen';
import {ModelsScreen} from './src/screens/ModelsScreen';
import {TrainingScreen} from './src/screens/TrainingScreen';
import {KnowledgeScreen} from './src/screens/KnowledgeScreen';
import {ActivityScreen} from './src/screens/ActivityScreen';
import {SettingsScreen} from './src/screens/SettingsScreen';
import {HealthScreen} from './src/screens/HealthScreen';
import {AboutScreen} from './src/screens/AboutScreen';
import {useModelStore} from './src/stores/model-store';
import {useSettingsStore} from './src/stores/settings-store';
import {ThemeProvider, useTheme} from './src/theme/ThemeContext';
import {ErrorBoundary} from './src/components/ErrorBoundary';
import {LoadingScreen} from './src/components/LoadingScreen';
import {ConnectionStatusBar} from './src/components/ConnectionStatusBar';
import {ToastContainer} from './src/components/ToastContainer';
import {OnboardingScreen, isFirstLaunch} from './src/screens/OnboardingScreen';
import {colors} from './src/theme';
import {initInference} from './src/services/activity-inference';
import {registerForPushNotifications, onNotification} from './src/services/push-notifications';


const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const TAB_ICONS: Record<string, string> = {
  Chat: '💬',
  Models: '🧠',
  Train: '🏋️',
  Knowledge: '📚',
  Activity: '📱',
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
    </Stack.Navigator>
  );
}

function AppInner() {
  const refresh = useModelStore(s => s.refresh);
  const {isDark} = useTheme();
  const [ready, setReady] = useState(false);
  const [needsOnboarding, setNeedsOnboarding] = useState(false);

  useEffect(() => {
    Promise.all([refresh(), initInference()]).finally(() => setReady(true));
    isFirstLaunch().then(setNeedsOnboarding);

    // Register for push notifications on startup (non-blocking)
    registerForPushNotifications().catch(() => {});

    // Listen for incoming notifications (non-blocking)
    const unsub = onNotification((title, body) => {
      console.log('[Push]', title, body);
    });
    return unsub;
  }, []);

  if (!ready) {
    return (
      <SafeAreaProvider>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#0F0D15' : '#FFFFFF'} />
        <LoadingScreen />
      </SafeAreaProvider>
    );
  }

  if (needsOnboarding) {
    return (
      <SafeAreaProvider>
        <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#0F0D15' : '#FFFFFF'} />
        <OnboardingScreen onComplete={() => setNeedsOnboarding(false)} />
      </SafeAreaProvider>
    );
  }

  const navTheme = isDark
    ? {
        ...DarkTheme,
        colors: {
          ...DarkTheme.colors,
          background: '#0F0D15',
          card: '#1A1725',
          text: '#F0ECF5',
          border: '#2D2A3A',
          primary: '#C0AAF4',
        },
      }
    : {
        ...DefaultTheme,
        colors: {
          ...DefaultTheme.colors,
          background: '#FFFFFF',
          card: '#F5F3F7',
          text: '#1A1625',
          border: '#E0DCE8',
          primary: '#7C52C4',
        },
      };

  return (
    <SafeAreaProvider>
      <StatusBar barStyle={isDark ? 'light-content' : 'dark-content'} backgroundColor={isDark ? '#0F0D15' : '#FFFFFF'} />
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
              backgroundColor: isDark ? '#1A1725' : '#FFFFFF',
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
          <Tab.Screen name="Chat" component={ChatScreen} />
          <Tab.Screen name="Models" component={ModelsScreen} />
          <Tab.Screen name="Train" component={TrainingScreen} />
          <Tab.Screen name="Knowledge" component={KnowledgeScreen} />
          <Tab.Screen name="Activity" component={ActivityScreen} />
          <Tab.Screen
            name="Settings"
            component={SettingsStack}
            options={{headerShown: false}}
          />
        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <ThemeProvider>
        <AppInner />
      </ThemeProvider>
    </ErrorBoundary>
  );
}
