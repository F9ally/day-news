import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Linking,
} from 'react-native';
import { FontAwesome5 } from '@expo/vector-icons';
import { useSettings } from '../context/SettingsContext';

interface ControlBarProps {
  isPlaying: boolean;
  isAudioLoading: boolean;
  onToggleAudio: () => void;
  onToggleTheme: () => void;
  onIncreaseFontSize: () => void;
  onDecreaseFontSize: () => void;
  onShare: () => void;
  onGoBack: () => void;
  onGoForward: () => void;
  onGoToday: () => void;
  canGoBack: boolean;
  canGoForward: boolean;
  isToday: boolean;
  title: string;
  /** 'daily' shows the calendar icon to open Monthly Summaries; 'monthly' shows a back button to daily. */
  mode?: 'daily' | 'monthly';
  onOpenMonthly?: () => void;
  onBackToDaily?: () => void;
}

export function ControlBar({
  isPlaying,
  isAudioLoading,
  onToggleAudio,
  onToggleTheme,
  onIncreaseFontSize,
  onDecreaseFontSize,
  onShare,
  onGoBack,
  onGoForward,
  onGoToday,
  canGoBack,
  canGoForward,
  isToday,
  title,
  mode = 'daily',
  onOpenMonthly,
  onBackToDaily,
}: ControlBarProps) {
  const { colors, themeMode } = useSettings();
  const isDark = themeMode === 'dark';

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
      {mode === 'daily' && onOpenMonthly ? (
        <TouchableOpacity
          style={[styles.calendarButton, { backgroundColor: colors.buttonBg }]}
          onPress={onOpenMonthly}
          activeOpacity={0.7}
          accessibilityLabel="View Monthly Summaries"
        >
          <FontAwesome5 name="calendar-alt" size={14} color={colors.iconColor} />
        </TouchableOpacity>
      ) : null}

      <Text
        style={[
          styles.title,
          { color: colors.headerText },
        ]}
        numberOfLines={1}
      >
        {title}
      </Text>

      <View style={styles.buttonContainer}>
        {/* Left group: Audio, Theme, Font, Share */}
        <View style={styles.buttonGroup}>
          {mode === 'monthly' && onBackToDaily ? (
            <TouchableOpacity
              style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
              onPress={onBackToDaily}
              activeOpacity={0.7}
              accessibilityLabel="Back to Daily Summaries"
            >
              <FontAwesome5 name="arrow-left" size={14} color={colors.iconColor} />
            </TouchableOpacity>
          ) : null}

          <TouchableOpacity
            style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
            onPress={onToggleAudio}
            activeOpacity={0.7}
            accessibilityLabel={isPlaying ? 'Pause audio' : 'Play audio'}
          >
            {isAudioLoading ? (
              <ActivityIndicator size="small" color={colors.iconColor} />
            ) : (
              <FontAwesome5
                name={isPlaying ? 'pause' : 'play'}
                size={14}
                color={colors.iconColor}
              />
            )}
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
            onPress={onToggleTheme}
            activeOpacity={0.7}
            accessibilityLabel="Toggle dark mode"
          >
            <FontAwesome5
              name={isDark ? 'sun' : 'moon'}
              size={14}
              color={colors.iconColor}
            />
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
            onPress={onIncreaseFontSize}
            activeOpacity={0.7}
            accessibilityLabel="Increase font size"
          >
            <FontAwesome5 name="plus" size={12} color={colors.iconColor} />
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
            onPress={onDecreaseFontSize}
            activeOpacity={0.7}
            accessibilityLabel="Decrease font size"
          >
            <FontAwesome5 name="minus" size={12} color={colors.iconColor} />
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
            onPress={onShare}
            activeOpacity={0.7}
            accessibilityLabel="Share summary"
          >
            <FontAwesome5 name="share" size={14} color={colors.iconColor} />
          </TouchableOpacity>
        </View>

        {/* Right group: Navigation */}
        <View style={styles.buttonGroup}>
          {canGoBack && (
            <TouchableOpacity
              style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
              onPress={onGoBack}
              activeOpacity={0.7}
              accessibilityLabel={mode === 'monthly' ? 'Previous month' : 'Previous day'}
            >
              <FontAwesome5
                name="chevron-left"
                size={14}
                color={colors.iconColor}
              />
            </TouchableOpacity>
          )}

          {canGoForward && (
            <TouchableOpacity
              style={[styles.iconButton, { backgroundColor: colors.buttonBg }]}
              onPress={onGoForward}
              activeOpacity={0.7}
              accessibilityLabel={mode === 'monthly' ? 'Next month' : 'Next day'}
            >
              <FontAwesome5
                name="chevron-right"
                size={14}
                color={colors.iconColor}
              />
            </TouchableOpacity>
          )}

          {mode === 'daily' && !isToday && (
            <TouchableOpacity
              style={[
                styles.todayButton,
                { backgroundColor: colors.buttonBg },
              ]}
              onPress={onGoToday}
              activeOpacity={0.7}
              accessibilityLabel="Go to today"
            >
              <Text style={[styles.todayButtonText, { color: colors.accent }]}>
                Today
              </Text>
            </TouchableOpacity>
          )}
        </View>
      </View>

    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: 'rgba(0,0,0,0.1)',
  },
  title: {
    fontSize: 20,
    fontWeight: '600',
    textAlign: 'center',
    marginBottom: 8,
    letterSpacing: 0.2,
  },
  calendarButton: {
    position: 'absolute',
    top: 8,
    right: 16,
    width: 34,
    height: 34,
    borderRadius: 17,
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  buttonGroup: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
  },
  iconButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  todayButton: {
    paddingHorizontal: 12,
    height: 36,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
  },
  todayButtonText: {
    fontSize: 13,
    fontWeight: '600',
  },
});
