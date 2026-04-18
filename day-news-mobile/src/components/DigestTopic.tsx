import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Linking,
} from 'react-native';
import { useSettings } from '../context/SettingsContext';
import { ParsedDigestItem } from '../types';

interface DigestTopicProps {
  item: ParsedDigestItem;
}

export function DigestTopic({ item }: DigestTopicProps) {
  const { colors, fontSize } = useSettings();

  const handlePress = () => {
    if (item.url) {
      Linking.openURL(item.url).catch(() => {});
    }
  };

  const headlineContent = (
    <Text
      style={[
        styles.headline,
        {
          color: item.url ? colors.linkText : colors.text,
          fontSize: fontSize - 1,
        },
      ]}
    >
      {item.headline}
    </Text>
  );

  return (
    <View style={[styles.container, { borderBottomColor: colors.border + '33' }]}>
      {/* Topic header with emoji */}
      <TouchableOpacity
        onPress={handlePress}
        disabled={!item.url}
        activeOpacity={item.url ? 0.6 : 1}
      >
        <Text style={[styles.topicHeader, { color: colors.headerText, fontSize: fontSize + 2 }]}>
          {item.emoji} {item.topic}
        </Text>
      </TouchableOpacity>

      {/* Headline */}
      {item.url ? (
        <TouchableOpacity onPress={handlePress} activeOpacity={0.6}>
          {headlineContent}
        </TouchableOpacity>
      ) : (
        headlineContent
      )}

      {/* Summary */}
      {item.summary ? (
        <Text
          style={[
            styles.summary,
            { color: colors.text, fontSize: fontSize - 2 },
          ]}
        >
          {item.summary}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 14,
    borderBottomWidth: StyleSheet.hairlineWidth,
  },
  topicHeader: {
    fontWeight: '600',
    marginBottom: 4,
    letterSpacing: 0.2,
  },
  headline: {
    fontWeight: '700',
    marginBottom: 6,
    lineHeight: 22,
  },
  summary: {
    lineHeight: 22,
  },
});
