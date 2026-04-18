import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ParsedDigestItem } from '../types';
import { DigestTopic } from './DigestTopic';
import { useSettings } from '../context/SettingsContext';

interface DigestContentProps {
  items: ParsedDigestItem[];
  dateStr: string;
}

export function DigestContent({ items, dateStr }: DigestContentProps) {
  const { colors, fontSize } = useSettings();

  if (items.length === 0) {
    return null;
  }

  return (
    <View style={[styles.card, { backgroundColor: colors.surface, shadowColor: colors.cardShadow }]}>
      <Text style={[styles.date, { color: colors.dateText, fontSize: fontSize - 2 }]}>
        {dateStr}
      </Text>
      {items.map((item, index) => (
        <DigestTopic key={`${item.topic}-${index}`} item={item} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 16,
    padding: 20,
    marginHorizontal: 16,
    marginTop: 12,
    marginBottom: 16,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 4,
  },
  date: {
    fontWeight: '700',
    textAlign: 'center',
    marginBottom: 12,
  },
});
