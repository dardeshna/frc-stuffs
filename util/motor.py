class Motor():
    """Motor model

    V = I*R + k_b*w
    T = k_t*I - k_F*w

    V = input voltage
    I = motor current
    T = output torque
    w = motor speed

    R = motor resistance
    k_t = torque constant
    k_b = back-emf constant
    k_F = viscous friction constant

    R = V_nominal / I_stall
    k_b = (V_nominal - R * I_free) / w_free
    k_t = T_stall / I_stall
    k_F = k_t * I_free / w_free
    """

    def __init__(self, V_n, w_f, T_s, I_s, I_f, n_motors):

        self.V_n = V_n
        self.w_f = w_f
        self.T_s = T_s * n_motors
        self.I_s = I_s * n_motors
        self.I_f = I_f * n_motors

        self.R = self.V_n / self.I_s
        self.k_b = (self.V_n - self.R * self.I_f) / self.w_f
        self.k_t = self.T_s / self.I_s
        self.k_F = self.k_t * self.I_f / self.w_f

    def get_I(self, V, w):

        return (V - self.k_b * w) / self.R