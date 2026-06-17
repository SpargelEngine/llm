"""Print the JSON Schema for TextPassList configuration."""

import json
import sys

from spargel_llm.text_pass import TextPassList

schema = TextPassList.model_json_schema()
json.dump(schema, sys.stdout, indent=2)
sys.stdout.write("\n")
