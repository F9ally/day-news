import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
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
}: ControlBarProps) {
  const { colors, themeMode } = useSettings();
  const isDark = themeMode === 'dark';

  return (
    <View style={[styles.container, { backgroundColor: colors.background }]}>
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
              accessibilityLabel="Previous day"
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
              accessibilityLabel="Next day"
            >
              <FontAwesome5
                name="chevron-right"
                size={14}
                color={colors.iconColor}
              />
            </TouchableOpacity>
          )}

          {!isToday && (
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
