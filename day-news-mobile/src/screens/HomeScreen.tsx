import React, { useCallback } from 'react';
import {
  View,
  Text,
  ScrollView,
  RefreshControl,
  Share,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useSettings } from '../context/SettingsContext';
import { useDigest } from '../hooks/useDigest';
import { useAudio } from '../hooks/useAudio';
import { ControlBar } from '../components/ControlBar';
import { DigestContent } from '../components/DigestContent';
import { AudioUnavailableModal } from '../components/AudioUnavailableModal';
import {
  isBeforeSixTwentyFiveUTC,
  formatDateForDisplay,
} from '../utils/dateUtils';

export function HomeScreen() {
  const { colors, toggleTheme, increaseFontSize, decreaseFontSize, fontSize } =
    useSettings();
  const insets = useSafeAreaInsets();
  const {
    currentDate,
    items,
    loading,
    error,
    isToday,
    goBack,
    goForward,
    goToday,
    canGoBack,
    canGoForward,
    refresh,
  } = useDigest();

  const audio = useAudio(currentDate);

  // Build the title matching website logic
  const title = isToday
    ? isBeforeSixTwentyFiveUTC()
      ? "Yesterday's Summary"
      : "Today's Summary"
    : formatDateForDisplay(currentDate);

  // Stop audio on navigation
  const handleGoBack = useCallback(() => {
    audio.stop();
    goBack();
  }, [audio, goBack]);

  const handleGoForward = useCallback(() => {
    audio.stop();
    goForward();
  }, [audio, goForward]);

  const handleGoToday = useCallback(() => {
    audio.stop();
    goToday();
  }, [audio, goToday]);

  // Share handler
  const handleShare = useCallback(async () => {
    const summaryText = items
      .map((item) => `${item.emoji} ${item.topic}: ${item.headline}\n${item.summary}`)
      .join('\n\n');

    const shareText = `${currentDate}\n\n${summaryText}\n\nhttps://day2day.news`;

    try {
      await Share.share({
        title: `Day2Day News: ${currentDate}`,
        message: shareText,
      });
    } catch {
      // User cancelled share
    }
  }, [items, currentDate]);

  // Refreshing state
  const [refreshing, setRefreshing] = React.useState(false);
  const onRefresh = useCallback(async () => {
    setRefreshing(true);
    await refresh();
    setRefreshing(false);
  }, [refresh]);

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      {/* App Title Header */}
      <View style={[styles.headerContainer, { paddingTop: insets.top, paddingHorizontal: 24, paddingVertical: 16 }]}>
        <Text style={[styles.headerTitle, { color: colors.text, fontSize: fontSize + 6 }]}>
          Day2Day News
        </Text>
        <Text style={[styles.headerSubtitle, { color: colors.textSecondary, fontSize }]}>
          Stay informed and save time!
        </Text>
      </View>

      {/* Sticky control bar */}
      <View>
        <ControlBar
          isPlaying={audio.isPlaying}
          isAudioLoading={audio.isLoading}
          onToggleAudio={audio.toggle}
          onToggleTheme={toggleTheme}
          onIncreaseFontSize={increaseFontSize}
          onDecreaseFontSize={decreaseFontSize}
          onShare={handleShare}
          onGoBack={handleGoBack}
          onGoForward={handleGoForward}
          onGoToday={handleGoToday}
          canGoBack={canGoBack}
          canGoForward={canGoForward}
          isToday={isToday}
          title={title}
        />
      </View>

      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[styles.scrollContent, { paddingBottom: insets.bottom + 24 }]}
        showsVerticalScrollIndicator={false}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor={colors.accent}
            colors={[colors.accent]}
          />
        }
      >
        {loading && !refreshing ? (
          <View style={styles.centerContainer}>
            <ActivityIndicator size="large" color={colors.accent} />
          </View>
        ) : error ? (
          <View style={styles.centerContainer}>
            <Text style={[styles.errorText, { color: colors.textSecondary, fontSize }]}>
              {error}
            </Text>
          </View>
        ) : (
          <DigestContent items={items} dateStr={currentDate} />
        )}

        {/* Footer */}
        <Text style={[styles.footer, { color: colors.textSecondary }]}>
          © 2025 Day2Day News Summaries. An Astronaut Website.{'\n'}
          AI can make mistakes. All rights reserved.{'\n'}
          Day2Day News does not endorse any ideas, political parties or products.
        </Text>
      </ScrollView>

      {/* Audio unavailable modal */}
      <AudioUnavailableModal
        visible={audio.audioUnavailable}
        onDismiss={audio.dismissUnavailable}
        onRetry={audio.retryPlay}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  root: {
    flex: 1,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
  },
  headerContainer: {
    alignItems: 'center',
    borderBottomWidth: 1,
    borderBottomColor: '#e0e0e0',
  },
  headerTitle: {
    fontWeight: '500',
    letterSpacing: 0.5,
  },
  headerSubtitle: {
    marginTop: 4,
    fontWeight: '400',
    letterSpacing: 0.2,
  },
  centerContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 80,
    paddingHorizontal: 24,
  },
  errorText: {
    textAlign: 'center',
    lineHeight: 24,
  },
  footer: {
    textAlign: 'center',
    fontSize: 11,
    lineHeight: 16,
    paddingHorizontal: 24,
    paddingTop: 20,
    paddingBottom: 8,
  },
});
