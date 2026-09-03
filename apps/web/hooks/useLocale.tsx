'use client'

import { createContext, useContext, useState, useCallback, ReactNode, useEffect } from 'react'
import { chatDB } from '@/lib/db'
import { trackEvent } from '@/lib/dev-log'

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
    'nav.agents': 'Agents',
    'nav.knowledge': 'Knowledge',
    'nav.monitoring': 'Health',
    'nav.adapters': 'Adapters',
    'nav.errors': 'Errors',
    'nav.feedback': 'Feedback',
    'nav.planner': 'Planner',
    'nav.files': 'Files',
    'nav.souls': 'Souls',
    'nav.security': 'Security',
    'nav.benchmark': 'Benchmark',
    'nav.tokenizer': 'Tokenizer',
    'nav.auto-train': 'Auto-Train',
    'nav.session': 'Sessions',
    'nav.experiments': 'Experiments',
    'nav.voice': 'Voice',
    'nav.dataset': 'Dataset Detail',
    'nav.shell': 'Shell',
    'nav.vector': 'Vector Store',
    'nav.section.core': 'Core',
    'nav.section.ai': 'AI',
    'nav.section.system': 'System',
    'nav.section.tools': 'Tools',
    'home.subtitle.offline': 'Offline',
    'home.apiOffline.title': 'Service Offline',
    'home.apiOffline.body': 'Service at {url} is not reachable',
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
    'nav.agents': 'Agentes',
    'nav.knowledge': 'Conocimiento',
    'nav.monitoring': 'Salud',
    'nav.adapters': 'Adaptadores',
    'nav.errors': 'Errores',
    'nav.feedback': 'Comentarios',
    'nav.planner': 'Planner',
    'nav.files': 'Archivos',
    'nav.souls': 'Almas',
    'nav.security': 'Seguridad',
    'nav.benchmark': 'Benchmark',
    'nav.tokenizer': 'Tokenizador',
    'nav.vector': 'Almacén Vectorial',
    'nav.auto-train': 'Entreno Auto',
    'nav.session': 'Sesiones',
    'nav.experiments': 'Experimentos',
    'nav.voice': 'Voz',
    'nav.dataset': 'Detalle Dataset',
    'nav.shell': 'Shell',
    'nav.section.core': 'Principal',
    'nav.section.ai': 'IA',
    'nav.section.system': 'Sistema',
    'nav.section.tools': 'Herramientas',
    'home.subtitle.offline': 'Sin conexión',
    'home.apiOffline.title': 'Servicio no accesible',
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
    'nav.agents': 'Agents',
    'nav.knowledge': 'Connaissances',
    'nav.monitoring': 'Santé',
    'nav.adapters': 'Adaptateurs',
    'nav.errors': 'Erreurs',
    'nav.feedback': 'Retour',
    'nav.planner': 'Planner',
    'nav.files': 'Fichiers',
    'nav.souls': 'Âmes',
    'nav.security': 'Sécurité',
    'nav.benchmark': 'Benchmark',
    'nav.tokenizer': 'Tokenizer',
    'nav.vector': 'Magasin Vectoriel',
    'nav.shell': 'Shell',
    'nav.auto-train': 'Entraînement Auto',
    'nav.session': 'Sessions',
    'nav.experiments': 'Expériences',
    'nav.voice': 'Voix',
    'nav.dataset': 'Détail Dataset',
    'nav.section.core': 'Principal',
    'nav.section.ai': 'IA',
    'nav.section.system': 'Système',
    'nav.section.tools': 'Outils',
    'home.subtitle.offline': 'Hors ligne',
    'home.apiOffline.title': 'Service inaccessible',
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
    'nav.agents': 'Agenten',
    'nav.knowledge': 'Wissen',
    'nav.monitoring': 'Gesundheit',
    'nav.adapters': 'Adapter',
    'nav.errors': 'Fehler',
    'nav.feedback': 'Feedback',
    'nav.planner': 'Planner',
    'nav.files': 'Dateien',
    'nav.souls': 'Seelen',
    'nav.security': 'Sicherheit',
    'nav.benchmark': 'Benchmark',
    'nav.tokenizer': 'Tokenizer',
    'nav.vector': 'Vektorspeicher',
    'nav.auto-train': 'Auto-Training',
    'nav.session': 'Sitzungen',
    'nav.experiments': 'Experimente',
    'nav.voice': 'Stimme',
    'nav.dataset': 'Dataset-Details',
    'nav.shell': 'Shell',
    'nav.section.core': 'Kern',
    'nav.section.ai': 'KI',
    'nav.section.system': 'System',
    'nav.section.tools': 'Werkzeuge',
    'home.subtitle.offline': 'Offline',
    'home.apiOffline.title': 'Dienst nicht erreichbar',
    'home.stats.status': 'Status',
    'home.stats.models': 'Modelle',
    'home.stats.personality': 'Persönlichkeit',
    'chat.send': 'Senden',
    'sidebar.close': 'Menü schließen',
    'sidebar.home': 'Startseite',
    'home.apiOffline.body': 'Dienst unter {url} ist nicht erreichbar',
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
    'nav.agents': '智能体',
    'nav.knowledge': '知识库',
    'nav.monitoring': '健康',
    'nav.adapters': '适配器',
    'nav.errors': '错误',
    'nav.feedback': '反馈',
    'nav.planner': 'Planner',
    'nav.files': '文件',
    'nav.souls': '灵魂',
    'nav.security': '安全',
    'nav.benchmark': '基准测试',
    'nav.tokenizer': '分词器',
    'nav.vector': '向量存储',
    'nav.auto-train': '自动训练',
    'nav.session': '会话',
    'nav.experiments': '实验',
    'nav.voice': '语音',
    'nav.dataset': '数据集详情',
    'nav.shell': 'Shell',
    'nav.section.core': '核心',
    'nav.section.ai': 'AI',
    'nav.section.system': '系统',
    'nav.section.tools': '工具',
    'home.subtitle.offline': '离线',
    'home.apiOffline.title': 'Service Unavailable',
    'home.stats.status': '状态',
    'home.stats.models': '模型',
    'home.stats.personality': '个性',
    'chat.send': '发送',
    'sidebar.close': '关闭菜单',
    'sidebar.home': '首页',
    'home.apiOffline.body': 'Service at {url} is not reachable',
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
  const [locale, setLocaleState] = useState<Locale>('en')

  useEffect(() => {
    let cancelled = false
    chatDB.getKV<string>(LOCALE_KEY).then(saved => {
      if (!cancelled && saved && saved in translations) {
        setLocaleState(saved as Locale)
        document.documentElement.lang = saved
        trackEvent('locale_loaded', { locale: saved })
      }
    })
    return () => { cancelled = true }
  }, [])

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(prev => {
      trackEvent('locale_changed', { from: prev, to: newLocale })
      return newLocale
    })
    chatDB.setKV(LOCALE_KEY, newLocale).catch(() => {})
    document.documentElement.lang = newLocale
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
