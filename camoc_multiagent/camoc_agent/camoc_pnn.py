import pnn.learner as pln


class CAMOC_PNN(Object):
    def __init__(
        self,
        state_dim,
        ctrl_dim,
        latent_dim,
        inn_volume_preserving=False,
        inn_layers=3,
        inn_sublayers=2,
        inn_subwidth=30,
        inn_activation="sigmoid",
        symp_Elayers=3,
        symp_Ewidth=30,
        symp_activation="sigmoid",
    ):

        self.state_dim = state_dim
        self.ctrl_dim = ctrl_dim
        self.tot_dim = state_dim + ctrl_dim
        self.latent_dim = latent_dim
        self.inn = pln.nn.INN(
            self.tot_dim,
            self.latent_dim,
            inn_layers,
            inn_sublayers,
            inn_sublayers,
            inn_activation,
            volume_preserving=inn_volume_preserving,
        )
        self.sympnet = pln.nn.ESympNet(
            self.tot_dim, self.latent_dim, symp_Elayers, symp_Ewidth, symp_activation
        )
        self.pnn = pln.nn.PNN(self.inn, self.sympnet)
