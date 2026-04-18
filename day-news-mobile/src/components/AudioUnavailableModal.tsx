import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Modal,
  StyleSheet,
  Pressable,
} from 'react-native';
import { useSettings } from '../context/SettingsContext';

interface AudioUnavailableModalProps {
  visible: boolean;
  onDismiss: () => void;
  onRetry: () => void;
}

export function AudioUnavailableModal({
  visible,
  onDismiss,
  onRetry,
}: AudioUnavailableModalProps) {
  const { colors } = useSettings();

  return (
    <Modal
      visible={visible}
      transparent
      animationType="fade"
      onRequestClose={onDismiss}
      statusBarTranslucent
    >
      <Pressable style={styles.overlay} onPress={onDismiss}>
        <View
          style={[styles.popup, { backgroundColor: colors.surface }]}
          onStartShouldSetResponder={() => true}
        >
          <Text style={[styles.title, { color: colors.text }]}>
            Audio not available yet
          </Text>

          <View style={styles.actions}>
            <TouchableOpacity
              style={[styles.button, styles.ghostButton]}
              onPress={onDismiss}
              activeOpacity={0.7}
            >
              <Text style={[styles.buttonText, { color: colors.textSecondary }]}>
                Dismiss
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[
                styles.button,
                styles.primaryButton,
                { backgroundColor: colors.accentLight },
              ]}
              onPress={onRetry}
              activeOpacity={0.7}
            >
              <Text style={[styles.buttonText, { color: colors.accent }]}>
                Try again
              </Text>
            </TouchableOpacity>
          </View>
        </View>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.4)',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 24,
  },
  popup: {
    borderRadius: 12,
    padding: 20,
    width: '100%',
    maxWidth: 420,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2,
    shadowRadius: 30,
    elevation: 10,
  },
  title: {
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 16,
  },
  actions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 8,
  },
  button: {
    borderRadius: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    minWidth: 100,
    alignItems: 'center',
  },
  ghostButton: {
    backgroundColor: 'transparent',
  },
  primaryButton: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 8,
    elevation: 2,
  },
  buttonText: {
    fontWeight: '600',
    fontSize: 15,
  },
});
