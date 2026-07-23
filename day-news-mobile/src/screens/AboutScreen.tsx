import React from 'react';
import {
  View,
  Text,
  ScrollView,
  TouchableOpacity,
  Linking,
  StyleSheet,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { useSettings } from '../context/SettingsContext';

export function AboutScreen() {
  const { colors, fontSize } = useSettings();
  const insets = useSafeAreaInsets();

  return (
    <View style={[styles.root, { backgroundColor: colors.background }]}>
      <ScrollView
        style={styles.scroll}
        contentContainerStyle={[
          styles.scrollContent,
          { paddingTop: insets.top + 16, paddingBottom: insets.bottom + 24 },
        ]}
        showsVerticalScrollIndicator={false}
      >
        {/* Header */}
        <View style={styles.header}>
          <Text style={[styles.headerTitle, { color: colors.headerText }]}>
            About & Donate
          </Text>
          <Text style={[styles.headerSubtitle, { color: colors.textSecondary }]}>
            Learn more and support the app
          </Text>
        </View>

        {/* About Card */}
        <View
          style={[
            styles.card,
            { backgroundColor: colors.surface, shadowColor: colors.cardShadow },
          ]}
        >
          <Text style={[styles.cardTitle, { color: colors.headerText, fontSize: fontSize + 2 }]}>
            About Day2Day News
          </Text>
          <Text style={[styles.cardText, { color: colors.text, fontSize }]}>
            Day2Day News is your go-to source for quick, reliable summaries of
            the most important events happening each day around the world. This
            is bite sized news without the bulk.
          </Text>
          <Text style={[styles.cardText, { color: colors.text, fontSize }]}>
            Curated daily by AI, our goal is to help you stay informed in under
            5 minutes a day. We go through testing to ensure information is
            accurate and trustworthy. However AI can make mistakes, so always
            verify the source. Links to the entire news feeds are always
            available, simply tap on the title to get redirected to the source.
          </Text>
          <Text style={[styles.cardText, { color: colors.text, fontSize }]}>
            Day2Day News is a project by Astronaut, an independent website and
            app developer.
          </Text>
        </View>

        {/* Support Card */}
        <View
          style={[
            styles.card,
            { backgroundColor: colors.surface, shadowColor: colors.cardShadow },
          ]}
        >
          <Text style={[styles.cardTitle, { color: colors.headerText, fontSize: fontSize + 2 }]}>
            Support Us
          </Text>
          <Text style={[styles.cardText, { color: colors.text, fontSize }]}>
            If you find Day2Day News helpful, please consider supporting us!
            Your donation helps keep this app running and completely free for
            everyone.
          </Text>
          <TouchableOpacity
            style={[styles.linkButton, { backgroundColor: colors.accentLight }]}
            onPress={() => Linking.openURL('https://revolut.me/loical')}
            activeOpacity={0.7}
          >
            <Text style={[styles.linkButtonText, { color: colors.accent }]}>
              Tip via Revolut →
            </Text>
          </TouchableOpacity>
        </View>

        {/* Contact Card */}
        <View
          style={[
            styles.card,
            { backgroundColor: colors.surface, shadowColor: colors.cardShadow },
          ]}
        >
          <Text style={[styles.cardTitle, { color: colors.headerText, fontSize: fontSize + 2 }]}>
            Contact
          </Text>
          <Text style={[styles.cardText, { color: colors.text, fontSize }]}>
            Have questions, feedback, or want to get in touch? We'd love to hear
            from you!
          </Text>
          <TouchableOpacity
            style={[styles.linkButton, { backgroundColor: colors.accentLight }]}
            onPress={() =>
              Linking.openURL('mailto:justwebsites.contact@gmail.com')
            }
            activeOpacity={0.7}
          >
            <Text style={[styles.linkButtonText, { color: colors.accent }]}>
              justwebsites.contact@gmail.com
            </Text>
          </TouchableOpacity>
        </View>

        {/* Visit Website Card */}
        <View
          style={[
            styles.card,
            { backgroundColor: colors.surface, shadowColor: colors.cardShadow },
          ]}
        >
          <Text style={[styles.cardTitle, { color: colors.headerText, fontSize: fontSize + 2 }]}>
            Website
          </Text>
          <Text style={[styles.cardText, { color: colors.text, fontSize }]}>
            Visit the full Day2Day News website for the same great content in
            your browser.
          </Text>
          <TouchableOpacity
            style={[styles.linkButton, { backgroundColor: colors.accentLight }]}
            onPress={() => Linking.openURL('https://day2day.news')}
            activeOpacity={0.7}
          >
            <Text style={[styles.linkButtonText, { color: colors.accent }]}>
              day2day.news →
            </Text>
          </TouchableOpacity>
        </View>

        {/* Footer */}
        <View style={styles.footerContainer}>
          <TouchableOpacity
            style={styles.footerLinkContainer}
            onPress={() =>
              Linking.openURL(
                'https://astronautapps.pages.dev/privacy_policy_day2daynews',
              )
            }
            activeOpacity={0.7}
            accessibilityRole="link"
            accessibilityLabel="Open privacy policy"
          >
            <Text style={[styles.footerLink, { color: colors.accent }]}>
              Privacy Policy →
            </Text>
          </TouchableOpacity>
          <Text style={[styles.footer, { color: colors.textSecondary }]}>
            © 2025 Day2Day News Summaries. An Astronaut Website.{'\n'}
            AI can make mistakes. All rights reserved.{'\n'}
            Day2Day News does not endorse any ideas, political parties or products.
          </Text>
        </View>
      </ScrollView>
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
    paddingHorizontal: 16,
  },
  header: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  headerTitle: {
    fontSize: 28,
    fontWeight: '500',
    marginBottom: 8,
  },
  headerSubtitle: {
    fontSize: 15,
    textAlign: 'center',
  },
  card: {
    borderRadius: 16,
    padding: 20,
    marginBottom: 12,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 1,
    shadowRadius: 8,
    elevation: 4,
  },
  cardTitle: {
    fontWeight: '600',
    marginBottom: 10,
  },
  cardText: {
    lineHeight: 24,
    marginBottom: 10,
  },
  linkButton: {
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 16,
    alignItems: 'center',
    marginTop: 4,
  },
  linkButtonText: {
    fontWeight: '600',
    fontSize: 15,
  },
  footerContainer: {
    paddingTop: 20,
    paddingBottom: 8,
  },
  footerLinkContainer: {
    alignItems: 'center',
    marginBottom: 12,
  },
  footerLink: {
    fontSize: 12,
    fontWeight: '600',
    textAlign: 'center',
  },
  footer: {
    textAlign: 'center',
    fontSize: 11,
    lineHeight: 16,
  },
});
