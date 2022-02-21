from abc import ABC, abstractmethod

class System(ABC):

    @property
    @abstractmethod
    def n_states(self):
        pass

    @property
    @abstractmethod
    def m_inputs(self):
        pass

    @abstractmethod
    def f(self, x, u):
        pass