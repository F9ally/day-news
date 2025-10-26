import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_and_publish_digest import generate_audio_with_kokoro

print("Testing Kokoro TTS with short text...")
test_text = "Today's news summary includes stories from around the world."

audio_bytes = generate_audio_with_kokoro(test_text, voice="am_michael")

if audio_bytes:
    print(f"\nSUCCESS: Generated {len(audio_bytes)} bytes of audio (WAV)")
    
    # Convert to MP3
    try:
        from pydub import AudioSegment
        import io
        print(f"Converting WAV to MP3...")
        wav_io = io.BytesIO(audio_bytes)
        audio_segment = AudioSegment.from_wav(wav_io)
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="128k")
        mp3_bytes = mp3_io.getvalue()
        
        # Save MP3
        test_path = os.path.join(os.path.dirname(__file__), "../out/test_audio.mp3")
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        with open(test_path, 'wb') as f:
            f.write(mp3_bytes)
        
        compression_ratio = (1 - len(mp3_bytes) / len(audio_bytes)) * 100
        print(f"MP3 size: {len(mp3_bytes)} bytes (~{len(mp3_bytes)/(1024*1024):.2f}MB)")
        print(f"Compression: {compression_ratio:.1f}% smaller than WAV")
        print(f"Saved test audio to: {test_path}")
    except Exception as e:
        print(f"MP3 conversion failed: {e}")
        # Fallback to WAV
        test_path = os.path.join(os.path.dirname(__file__), "../out/test_audio.wav")
        os.makedirs(os.path.dirname(test_path), exist_ok=True)
        with open(test_path, 'wb') as f:
            f.write(audio_bytes)
        print(f"Saved WAV fallback to: {test_path}")
else:
    print("\nFAILED: No audio generated")
