import numpy as np
from scipy.integrate import solve_ivp

class Sim():

    def __init__(self, system, x_0):

        self.system = system
        self.reset(x_0)
        
    def reset(self, x_0):

        self.xs = [x_0]
        self.us = [np.full((self.system.m_inputs,),np.nan).squeeze()]
        self.ts = [0]

        self.i = 0

    def step(self, u, dt):

        if callable(u):
            if hasattr(self.system, 'clip'):
                self.us[self.i] = self.system.clip(self.xs[self.i], u(self.xs[self.i], self.ts[self.i]), self.ts[self.i])[1]
                l = lambda t, x: self.system.f(*self.system.clip(x, np.atleast_1d(u(x, t)), t))
            else:
                self.us[self.i] = u(self.xs[self.i], self.ts[self.i])
                l = lambda t, x: self.system.f(x, np.atleast_1d(u(x, t)), t)
        else:
            if hasattr(self.system, 'clip'):
                self.us[self.i] = self.system.clip(self.xs[self.i], u, self.ts[self.i])[1]
                l = lambda t, x, u=np.atleast_1d(u): self.system.f(*self.system.clip(x, u, t))
            else:
                self.us[self.i] = u
                l = lambda t, x, u=np.atleast_1d(u): self.system.f(x, u, t)

        # integrate equations of motion from t to t+dt with constant input u
        res = solve_ivp(l, (self.ts[-1], self.ts[-1]+dt), np.atleast_1d(self.xs[-1]))

        self.xs.append(res['y'][:,-1])
        self.us.append(np.full((self.system.m_inputs,),np.nan).squeeze())
        self.ts.append(self.ts[-1]+dt)
        
        self.i += 1

    def solve(self, u, t):

        if callable(u):
            for dt in np.diff(t):
                self.step(u, dt)
        else:
            for i, dt in enumerate(np.diff(t)):
                self.step(u[i], dt)

    def get_result(self):

        return np.array(self.xs), np.array(self.us), np.array(self.ts)