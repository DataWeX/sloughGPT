"""AML — Automatic Markup Language.

A structured data format designed for machine-to-machine and
human-to-machine communication.  Lighter than JSON, more precise
than YAML, self-describing with typed blocks.

MIME type: application/aml
File extension: .aml
"""

from aml.mime import detect_mime, is_aml, register_mime
from aml.parser import parse, parse_file
from aml.schema import AmlBlock, AmlDocument, AmlValue
from aml.serializer import dict_to_aml, serialize, serialize_file

__version__ = "0.1.0"
__all__ = [
    "parse",
    "parse_file",
    "serialize",
    "serialize_file",
    "dict_to_aml",
    "AmlDocument",
    "AmlBlock",
    "AmlValue",
    "register_mime",
    "detect_mime",
    "is_aml",
]
