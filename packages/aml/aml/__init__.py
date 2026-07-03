"""AML — Automatic Markup Language.

A structured data format designed for machine-to-machine and
human-to-machine communication.  Lighter than JSON, more precise
than YAML, self-describing with typed blocks.

MIME type: application/aml
File extension: .aml
"""

from aml.parser import parse, parse_file
from aml.serializer import serialize, serialize_file, dict_to_aml
from aml.schema import AmlDocument, AmlBlock, AmlValue
from aml.mime import register_mime, detect_mime, is_aml

__version__ = "0.1.0"
__all__ = ["parse", "parse_file", "serialize", "serialize_file",
           "dict_to_aml", "AmlDocument", "AmlBlock", "AmlValue",
           "register_mime", "detect_mime", "is_aml"]
