import json
import os
from typing import Optional


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "prompts", "prompt_config.json")
_config_cache: Optional[dict] = None


def _load_config() -> dict:
    global _config_cache
    if _config_cache is None:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            _config_cache = json.load(f)
    return _config_cache


class PromptManager:
    """
    Manages language-aware prompts and terminology for the agent system.

    Usage:
        pm = PromptManager(language="pt-BR")
        system_messages = pm.build_system_prefix(current_time="2025-03-25T14:00:00")
        translated = pm.translate_term("Non-stop")  # -> "Voo Direto"

    The language is resolved in this order:
        1. Explicit `language` argument passed to __init__
        2. `AGENT_LANGUAGE` key in the config dict
        3. `default_language` from prompt_config.json
    """

    def __init__(self, language: Optional[str] = None, config: Optional[dict] = None):
        cfg = _load_config()
        lang = (
            language
            or (config or {}).get("AGENT_LANGUAGE")
            or cfg.get("default_language", "pt-BR")
        )
        if lang not in cfg["languages"]:
            lang = cfg.get("default_language", "pt-BR")
        self.language = lang
        self._lang_cfg = cfg["languages"][lang]

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------

    def get_directive(self) -> str:
        """Explicit instruction that forces the LLM to use the chosen language."""
        return self._lang_cfg["directive"]

    def get_meta_instruction(self, key: str) -> str:
        """Return a translated meta instruction string (opening_message, etc.)."""
        return self._lang_cfg["meta_instructions"].get(key, "")

    def format_meta_instruction(self, key: str, **kwargs) -> str:
        """
        Fetch a meta instruction by key and format it with kwargs.
        If the key doesn't exist, returns empty string.
        """
        template = self.get_meta_instruction(key)
        if not template:
            return ""
        try:
            return template.format(**kwargs)
        except Exception:
            # Avoid breaking the agent loop if formatting fails
            return template

    def get_tone(self) -> str:
        return self._lang_cfg.get("tone", "")

    def get_date_format(self) -> str:
        return self._lang_cfg.get("date_format", "DD/MM/YYYY")

    def get_time_format(self) -> str:
        return self._lang_cfg.get("time_format", "24h")

    def get_error_message(self, key: str) -> str:
        return self._lang_cfg.get("error_messages", {}).get(key, "")

    # ------------------------------------------------------------------
    # Terminology translation
    # ------------------------------------------------------------------

    def translate_term(self, term: str) -> str:
        """
        Translate a single API term to the target language.
        Falls back to the original term if no translation exists.

        Example:
            pm.translate_term("Non-stop")  ->  "Voo Direto"  (pt-BR)
            pm.translate_term("Non-stop")  ->  "Non-stop"    (en)
        """
        return self._lang_cfg.get("terminology", {}).get(term, term)

    def translate_text(self, text: str) -> str:
        """
        Replace all known API terms found in a text block with their translations.
        Useful for post-processing API responses before showing them to the LLM or user.
        """
        for source_term, translated_term in self._lang_cfg.get("terminology", {}).items():
            text = text.replace(source_term, translated_term)
        return text

    # ------------------------------------------------------------------
    # System message builders
    # ------------------------------------------------------------------

    def build_system_prefix(self, current_time: str = "") -> list:
        """
        Return the ordered list of system messages that should be prepended
        to every LLM call. This includes:
          [0] Language directive (highest priority)
          [1] Opening / persona message
          [2] Current time (if provided)
          [3] Tone and format rules

        These should be inserted BEFORE action-specific instructions.
        """
        messages = [
            {"role": "system", "content": self.get_directive()},
            {"role": "system", "content": self.get_meta_instruction("opening_message")},
        ]
        if current_time:
            if self.language == "pt-BR":
                messages.append({
                    "role": "system",
                    "content": f"A data e hora atual é: {current_time}. Use o formato de data {self.get_date_format()} e horário {self.get_time_format()} nas suas respostas."
                })
            else:
                messages.append({
                    "role": "system",
                    "content": f"The current date and time is: {current_time}. Use date format {self.get_date_format()} and {self.get_time_format()} time in your responses."
                })
        if self.get_tone():
            messages.append({"role": "system", "content": self.get_tone()})
        return messages

    def get_optimal_path_instruction(self) -> str:
        return self.get_meta_instruction("optimal_path")

    def get_answer_from_belief_instruction(self) -> str:
        return self.get_meta_instruction("answer_from_belief")
