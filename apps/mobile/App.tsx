import React, {useEffect} from 'react';
import {StatusBar, Text} from 'react-native';
import {NavigationContainer} from '@react-navigation/native';
import {createBottomTabNavigator} from '@react-navigation/bottom-tabs';
import {createNativeStackNavigator} from '@react-navigation/native-stack';
import {SafeAreaProvider} from 'react-native-safe-area-context';
import {ChatScreen} from './src/screens/ChatScreen';
import {ModelsScreen} from './src/screens/ModelsScreen';
import {TrainingScreen} from './src/screens/TrainingScreen';
import {KnowledgeScreen} from './src/screens/KnowledgeScreen';
import {SettingsScreen} from './src/screens/SettingsScreen';
import {HealthScreen} from './src/screens/HealthScreen';
import {useModelStore} from './src/stores/model-store';
import {colors} from './src/theme';

const Tab = createBottomTabNavigator();
const Stack = createNativeStackNavigator();

const TAB_ICONS: Record<string, string> = {
  Chat: '💬',
  Models: '🧠',
  Train: '🏋️',
  Knowledge: '📚',
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
    </Stack.Navigator>
  );
}

export default function App() {
  const refresh = useModelStore(s => s.refresh);

  useEffect(() => {
    refresh();
  }, []);

  return (
    <SafeAreaProvider>
      <StatusBar barStyle="dark-content" backgroundColor={colors.background} />
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({route}) => ({
            headerShown: false,
            tabBarIcon: ({focused}) => (
              <TabIcon name={route.name} focused={focused} />
            ),
            tabBarActiveTintColor: colors.primary,
            tabBarInactiveTintColor: colors.textMuted,
            tabBarStyle: {
              backgroundColor: colors.background,
              borderTopColor: colors.border,
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
