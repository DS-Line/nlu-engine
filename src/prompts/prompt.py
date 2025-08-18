import pathlib

from .registry import _prompt_registry


class Prompt:
    """
    Provides access to registered prompt templates and allows variable substitution.
    """

    base_dir = pathlib.Path(__file__).resolve().parent / "assets"

    @staticmethod
    def get(name: str, vars: dict | None = None) -> str:
        """
        Retrieve a prompt by its registered name, substituting variables.

        Args:
            name (str): The name of the prompt (as defined in the registry).
            vars (dict[str, Any] | None): Key-value pairs for variable substitution.

        Returns:
            str: The rendered prompt string.

        Raises:
            ValueError: If the prompt name is not in the registry.
            RuntimeError: If the prompt file cannot be read.
        """
        if name not in _prompt_registry:
            raise ValueError(f"Prompt '{name}' not found in registry.")

        vars = vars or {}
        template_path = Prompt.base_dir / _prompt_registry[name]

        try:
            with template_path.open(encoding="utf-8") as f:
                template = f.read()

        except Exception as e:
            raise RuntimeError(f"Failed to load prompt file '{template_path}': {e}") from e

        for key, value in vars.items():
            template = template.replace(f"{{{key}}}", value)

        return template
