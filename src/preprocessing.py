from abc import ABC, abstractmethod
from typing import Callable


class PreprocessingOpBase(ABC):
    """
    Base class for all the preprocessing operations.
    """

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """
        Virtual method, applies pre processing.

        Note: This method has to be thread-safe.
        :param text: The text to be preprocessed.
        :return: Preprocessed text.
        """
        pass

    def __call__(self, text: str) -> str:
        """
        Overridden call operator, calls a preprocessing function on a piece of text.
        :param text: The text to be preprocessed.
        :return:
        """
        return self.preprocess(text)


class LambdaOp(PreprocessingOpBase):
    """
    Implementation of custom preprocessing operation specified as lambda expression or as a function pointer.
    """

    def __init__(self, func: Callable):
        """
        Constructor.
        :param func: Callable which will be called on a piece of text.
        """
        self._func = func

    def preprocess(self, text: str) -> str:
        """
        Implementation of the base class' virtual method.
        :param text: The text to be preprocessed.
        :return: Preprocessed text.
        """
        return self._func(text)


class ToLowercaseOp(PreprocessingOpBase):
    """
    Implementation of the 'lower()' preprocessing operation.
    """

    def preprocess(self, text: str) -> str:
        """
        Implementation of the base class' virtual method.
        :param text: The text to be preprocessed.
        :return: Preprocessed text.
        """
        return text.lower()


class RemoveSubstringOp(PreprocessingOpBase):
    """
    Removes given substring from a piece of text.
    """

    def __init__(self, substring: str):
        """
        Constructor.
        :param substring: Substring to be removed.
        """
        self._substring = substring

    def preprocess(self, text: str) -> str:
        """
        Implementation of the base class' virtual method.
        :param text: The text to be preprocessed.
        :return: Preprocessed text.
        """
        return text.replace(self._substring, '')
