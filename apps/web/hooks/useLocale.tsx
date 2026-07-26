'use client'

import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react'

type Locale = 'en' | 'es' | 'fr' | 'de' | 'zh'

interface Translations {
  [key: string]: string
}

const translations: Record<Locale, Translations> = {
  en: {
    'app.name': 'Man',
    'app.console': 'Console',
    'common.starting': 'Starting...',
    'common.starting_sub': 'Model is loading, one moment',
    'common.save': 'Save',
    'nav.chat': 'Chat',
    'nav.models': 'Personalities',
    'nav.datasets': 'Datasets',
    'nav.training': 'Teach me',
    'nav.settings': 'Settings',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agents',
    'home.subtitle.offline': 'API not reachable',
    'home.apiOffline.title': 'API not reachable',
    'home.apiOffline.body': 'API server at {url} is not reachable',
    'home.stats.status': 'Status',
    'home.stats.models': 'Models',
    'home.stats.personality': 'Personality',
    'chat.send': 'Send',
    'sidebar.close': 'Close menu',
    'sidebar.home': 'Home',
  },

  es: {
    'app.name': 'Man',
    'app.console': 'Consola',
    'common.starting': 'Iniciando...',
    'common.starting_sub': 'El modelo está cargando, un momento',
    'common.save': 'Guardar',
    'nav.chat': 'Chat',
    'nav.models': 'Personalidades',
    'nav.datasets': 'Datos',
    'nav.training': 'Enseñame',
    'nav.settings': 'Configuración',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agentes',
    'home.subtitle.offline': 'API no accesible',
    'home.apiOffline.title': 'API no accesible',
    'home.stats.status': 'Estado',
    'home.stats.models': 'Modelos',
    'home.stats.personality': 'Personalidad',
    'chat.send': 'Enviar',
    'sidebar.close': 'Cerrar menú',
    'sidebar.home': 'Inicio',
    'home.apiOffline.body': 'El servidor API en {url} no es accesible',
  },

  fr: {
    'app.name': 'Man',
    'app.console': 'Console',
    'common.starting': 'Démarrage...',
    'common.starting_sub': 'Le modèle se charge, un instant',
    'common.save': 'Enregistrer',
    'nav.chat': 'Chat',
    'nav.models': 'Personnalités',
    'nav.datasets': 'Données',
    'nav.training': 'Enseigne-moi',
    'nav.settings': 'Paramètres',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agents',
    'home.subtitle.offline': 'API inaccessible',
    'home.apiOffline.title': 'API inaccessible',
    'home.stats.status': 'Statut',
    'home.stats.models': 'Modèles',
    'home.stats.personality': 'Personnalité',
    'chat.send': 'Envoyer',
    'sidebar.close': 'Fermer le menu',
    'sidebar.home': 'Accueil',
    'home.apiOffline.body': 'Le serveur API à {url} est inaccessible',
  },

  de: {
    'app.name': 'Man',
    'app.console': 'Konsole',
    'common.starting': 'Starte...',
    'common.starting_sub': 'Modell wird geladen, einen Moment',
    'common.save': 'Speichern',
    'nav.chat': 'Chat',
    'nav.models': 'Persönlichkeiten',
    'nav.datasets': 'Datensätze',
    'nav.training': 'Bring mir bei',
    'nav.settings': 'Einstellungen',
    'nav.multimodal': 'Multimodal',
    'nav.agents': 'Agenten',
    'home.subtitle.offline': 'API nicht erreichbar',
    'home.apiOffline.title': 'API nicht erreichbar',
    'home.stats.status': 'Status',
    'home.stats.models': 'Modelle',
    'home.stats.personality': 'Persönlichkeit',
    'chat.send': 'Senden',
    'sidebar.close': 'Menü schließen',
    'sidebar.home': 'Startseite',
    'home.apiOffline.body': 'API-Server unter {url} ist nicht erreichbar',
  },

  zh: {
    'app.name': 'Man',
    'app.console': '控制台',
    'common.starting': '启动中...',
    'common.starting_sub': '模型正在加载，请稍候',
    'common.save': '保存',
    'nav.chat': '聊天',
    'nav.models': '个性',
    'nav.datasets': '数据集',
    'nav.training': '教我',
    'nav.settings': '设置',
    'nav.multimodal': '多模态',
    'nav.agents': '智能体',
    'home.subtitle.offline': 'API 不可达',
    'home.apiOffline.title': 'API 不可达',
    'home.stats.status': '状态',
    'home.stats.models': '模型',
    'home.stats.personality': '个性',
    'chat.send': '发送',
    'sidebar.close': '关闭菜单',
    'sidebar.home': '首页',
    'home.apiOffline.body': 'API 服务器 {url} 不可达',
  },
}

interface LocaleContextType {
  locale: Locale
  setLocale: (locale: Locale) => void
  t: (key: string, params?: Record<string, string | number>) => string
  locales: Locale[]
}

const LocaleContext = createContext<LocaleContextType | undefined>(undefined)

const LOCALE_KEY = 'man_locale'

export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem(LOCALE_KEY) as Locale
      if (saved && translations[saved]) return saved
    }
    return 'en'
  })

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale)
    if (typeof window !== 'undefined') {
      localStorage.setItem(LOCALE_KEY, newLocale)
      document.documentElement.lang = newLocale
    }
  }, [])

  useEffect(() => {
    document.documentElement.lang = locale
  }, [locale])

  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    let text = translations[locale]?.[key] || translations.en?.[key] || key
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        text = text.replace(`{${k}}`, String(v))
      }
    }
    return text
  }, [locale])

  return (
    <LocaleContext.Provider value={{ locale, setLocale, t, locales: Object.keys(translations) as Locale[] }}>
      {children}
    </LocaleContext.Provider>
  )
}

export function useLocale() {
  const context = useContext(LocaleContext)
  if (!context) {
    throw new Error('useLocale must be used within a LocaleProvider')
  }
  return context
}

export const LOCALES = [
  { code: 'en', name: 'English', flag: '🇺🇸' },
  { code: 'es', name: 'Español', flag: '🇪🇸' },
  { code: 'fr', name: 'Français', flag: '🇫🇷' },
  { code: 'de', name: 'Deutsch', flag: '🇩🇪' },
  { code: 'zh', name: '中文', flag: '🇨🇳' },
] as const
