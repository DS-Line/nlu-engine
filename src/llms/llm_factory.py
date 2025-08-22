from importlib import import_module
from inspect import getmembers, isabstract, isclass

from decouple import config

from src.llms.base import BaseLLM


class LLMFactory:
    """
    Factory class for LLM
    """

    _package = "src.llms"

    @staticmethod
    def load_module(module_name: str) -> BaseLLM | None:
        """
        Dynamic import of LLM class

        :param module_name: name of the llm module
        :return : Distribution class
        :raise : Import Error
        """
        factory_module = import_module(f".{module_name}", package=LLMFactory._package)
        classes = getmembers(factory_module, lambda cls: isclass(cls) and not isabstract(cls))

        for _, _class in classes:
            if issubclass(_class, BaseLLM):
                return _class()
        raise ImportError


llm = LLMFactory.load_module(config("PROVIDER"))
