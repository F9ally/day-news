# Day2Day News — Mobile App

A production-grade React Native (Expo) mobile app for [Day2Day News](https://day2day.news).

## Features

- **Daily AI-curated news digests** — Same data as the website, powered by Supabase
- **Audio summaries** — Play/pause MP3 audio digests
- **Dark mode** — Persisted across sessions
- **Font size control** — Adjustable text size
- **Date navigation** — Browse past digests with back/forward controls
- **Pull to refresh** — Swipe down to reload the latest digest
- **Share** — Share the daily summary via native share sheet
- **About & Donate** — Support the project

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) ≥ 18
- [Expo CLI](https://docs.expo.dev/get-started/installation/) or `npx expo`
- Expo Go app on your phone (iOS/Android)

### Install

```bash
cd day-news-mobile
npm install
```

### Run

```bash
npx expo start
```

Scan the QR code with Expo Go (Android) or the Camera app (iOS).

### Build for production

```bash
npx expo build:android   # Android APK/AAB
npx expo build:ios       # iOS IPA
```

Or use EAS Build:

```bash
npx eas build --platform all
```

## Project Structure

```
day-news-mobile/
├── App.tsx                             # Entry point, navigation
├── app.json                            # Expo configuration
├── src/
│   ├── config/supabase.ts              # Supabase client
│   ├── constants/theme.ts              # Colors, emojis, config
│   ├── context/SettingsContext.tsx      # Dark mode + font size
│   ├── hooks/
│   │   ├── useDigest.ts                # Fetch & navigate digests
│   │   └── useAudio.ts                 # Audio playback
│   ├── components/
│   │   ├── ControlBar.tsx              # Sticky toolbar
│   │   ├── DigestContent.tsx           # Full digest card
│   │   ├── DigestTopic.tsx             # Individual topic section
│   │   └── AudioUnavailableModal.tsx   # Audio error modal
│   ├── screens/
│   │   ├── HomeScreen.tsx              # Main digest view
│   │   └── AboutScreen.tsx             # About & donate
│   ├── utils/
│   │   ├── dateUtils.ts                # UTC date helpers
│   │   └── digestParser.ts             # Parse digest items
│   └── types/index.ts                  # TypeScript types
```

## Backend

Uses the same Supabase backend as the website:
- Table: `daily_digests` (date, compiled HTML, items JSON)
- Storage: `news-audio` bucket for daily MP3 audio summaries
