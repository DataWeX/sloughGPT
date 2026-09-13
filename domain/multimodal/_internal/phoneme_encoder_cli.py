#!/usr/bin/env python3
"""
CLI tool for testing the phoneme encoder.

Usage:
    python -m domains.multimodal.phoneme_encoder_cli "hello world"
    python -m domains.multimodal.phoneme_encoder_cli --batch "hello" "world" "test"
    python -m domains.multimodal.phoneme_encoder_cli --lang it "ciao mondo"
    python -m domains.multimodal.phoneme_encoder_cli --lang pt "ola mundo"
    python -m domains.multimodal.phoneme_encoder_cli --detect "hello world"
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domains.multimodal.unified_phoneme_encoder import UnifiedPhonemeEncoder, detect_language


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m domains.multimodal.phoneme_encoder_cli <text>")
        print("       python -m domains.multimodal.phoneme_encoder_cli --batch <text1> <text2> ...")
        print("       python -m domains.multimodal.phoneme_encoder_cli --lang <lang> <text>")
        print("       python -m domains.multimodal.phoneme_encoder_cli --detect <text>")
        print()
        print("Languages: en, de, fr, es, it, pt")
        sys.exit(1)

    enc = UnifiedPhonemeEncoder()

    # Parse arguments
    language = None
    batch_mode = False
    detect_mode = False
    texts = []

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--batch":
            batch_mode = True
            i += 1
        elif sys.argv[i] == "--detect":
            detect_mode = True
            i += 1
        elif sys.argv[i] == "--lang" and i + 1 < len(sys.argv):
            language = sys.argv[i + 1]
            i += 2
        else:
            texts.append(sys.argv[i])
            i += 1

    if detect_mode:
        if not texts:
            print("Error: No text provided for language detection")
            sys.exit(1)

        text = " ".join(texts)
        detected = detect_language(text)
        print(f"Detected language: {detected}")
        print(f"Supported languages: {', '.join(enc.supported_languages)}")
    elif batch_mode:
        if not texts:
            print("Error: No texts provided for batch encoding")
            sys.exit(1)

        print(f"Batch encoding {len(texts)} texts:")
        print()

        for text in texts:
            print(enc.visualize(text, language=language))
            print()
    else:
        if not texts:
            print("Error: No text provided")
            sys.exit(1)

        text = " ".join(texts)
        print(enc.visualize(text, language=language))


if __name__ == "__main__":
    main()
