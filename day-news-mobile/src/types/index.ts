export interface DigestItem {
  topic: string;
  title: string | null;
  url: string | null;
  published_at: string | null;
  summary: string;
}

export interface DigestRecord {
  id?: number;
  date: string; // YYYY-MM-DD
  compiled: string; // HTML string
  items: DigestItem[];
  created_at: string;
}

export interface ParsedDigestItem {
  topic: string;
  emoji: string;
  headline: string;
  summary: string;
  url: string | null;
  title: string | null;
}

export type ThemeMode = 'light' | 'dark';

export interface ThemeColors {
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  accent: string;
  accentLight: string;
  accentHover: string;
  border: string;
  cardShadow: string;
  headerText: string;
  dateText: string;
  linkText: string;
  iconColor: string;
  buttonBg: string;
  buttonBgActive: string;
  buttonShadow: string;
}
