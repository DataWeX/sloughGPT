"""
Voice command group — text-to-speech and speech-to-text.
"""

from core.framework import click
from core.helpers import ns as _ns


def register(cli):
    """Register voice commands with the CLI group."""

    @cli.group(help="Text-to-speech and speech-to-text via the API")
    def voice():
        pass

    @voice.command("tts", help="Convert text to speech audio")
    @click.argument("text")
    @click.option("--output", "-o", help="Output file path (default: tts_output.wav)")
    @click.option("--play", is_flag=True, help="Play audio after generating")
    def voice_tts(text, output, play):
        from commands.voice import cmd_voice_tts
        cmd_voice_tts(_ns(text=text, output=output, play=play))

    @voice.command("stt", help="Transcribe an audio file to text")
    @click.argument("file")
    @click.option("--language", "-l", default="en", help="Audio language (default: en)")
    @click.option("--verbose", "-v", is_flag=True, help="Show confidence and language info")
    def voice_stt(file, language, verbose):
        from commands.voice import cmd_voice_stt
        cmd_voice_stt(_ns(file=file, language=language, verbose=verbose))

    return voice
