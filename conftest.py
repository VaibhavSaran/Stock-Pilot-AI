import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Patch pydantic-settings BEFORE any test imports chromadb.
# ChromaDB's Settings class reads .env but forbids unknown fields —
# this must be patched before any chromadb submodule is imported.
import pydantic_settings
pydantic_settings.BaseSettings.model_config["extra"] = "ignore"