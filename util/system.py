from abc import ABC, abstractmethod

class System(ABC):

    @property
    @abstractmethod
    def n(self):
        pass

    @property
    @abstractmethod
    def m(self):
        pass

    @abstractmethod
    def f(self, x, u):
        pass