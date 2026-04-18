import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useMemo,
  ReactNode,
} from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { ThemeMode, ThemeColors } from '../types';
import {
  LIGHT_THEME,
  DARK_THEME,
  FONT_SIZE_BASE,
  FONT_SIZE_STEP,
  FONT_SIZE_MIN_LEVEL,
  FONT_SIZE_MAX_LEVEL,
} from '../constants/theme';

interface SettingsContextValue {
  themeMode: ThemeMode;
  colors: ThemeColors;
  toggleTheme: () => void;
  fontSize: number;
  fontSizeLevel: number;
  increaseFontSize: () => void;
  decreaseFontSize: () => void;
}

const SettingsContext = createContext<SettingsContextValue | null>(null);

const THEME_STORAGE_KEY = '@day2day_darkMode';
const FONT_SIZE_STORAGE_KEY = '@day2day_fontSizeLevel';

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [themeMode, setThemeMode] = useState<ThemeMode>('light');
  const [fontSizeLevel, setFontSizeLevel] = useState(0);
  const [loaded, setLoaded] = useState(false);

  // Load persisted settings
  useEffect(() => {
    (async () => {
      try {
        const [savedTheme, savedFontSize] = await Promise.all([
          AsyncStorage.getItem(THEME_STORAGE_KEY),
          AsyncStorage.getItem(FONT_SIZE_STORAGE_KEY),
        ]);
        if (savedTheme === 'true') setThemeMode('dark');
        if (savedFontSize !== null) {
          const level = parseInt(savedFontSize, 10);
          if (!isNaN(level)) setFontSizeLevel(level);
        }
      } catch (e) {
        // Ignore storage errors
      } finally {
        setLoaded(true);
      }
    })();
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeMode((prev) => {
      const next = prev === 'light' ? 'dark' : 'light';
      AsyncStorage.setItem(
        THEME_STORAGE_KEY,
        next === 'dark' ? 'true' : 'false',
      ).catch(() => {});
      return next;
    });
  }, []);

  const increaseFontSize = useCallback(() => {
    setFontSizeLevel((prev) => {
      const next = Math.min(prev + 1, FONT_SIZE_MAX_LEVEL);
      AsyncStorage.setItem(FONT_SIZE_STORAGE_KEY, String(next)).catch(
        () => {},
      );
      return next;
    });
  }, []);

  const decreaseFontSize = useCallback(() => {
    setFontSizeLevel((prev) => {
      const next = Math.max(prev - 1, FONT_SIZE_MIN_LEVEL);
      AsyncStorage.setItem(FONT_SIZE_STORAGE_KEY, String(next)).catch(
        () => {},
      );
      return next;
    });
  }, []);

  const colors = themeMode === 'dark' ? DARK_THEME : LIGHT_THEME;
  const fontSize = FONT_SIZE_BASE + fontSizeLevel * FONT_SIZE_STEP;

  const value = useMemo<SettingsContextValue>(
    () => ({
      themeMode,
      colors,
      toggleTheme,
      fontSize,
      fontSizeLevel,
      increaseFontSize,
      decreaseFontSize,
    }),
    [themeMode, colors, toggleTheme, fontSize, fontSizeLevel, increaseFontSize, decreaseFontSize],
  );

  if (!loaded) return null; // Avoid flash of wrong theme

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings(): SettingsContextValue {
  const ctx = useContext(SettingsContext);
  if (!ctx) {
    throw new Error('useSettings must be used within a SettingsProvider');
  }
  return ctx;
}
