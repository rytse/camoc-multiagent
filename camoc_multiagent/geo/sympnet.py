import torch
from torch import nn


class LinSympLayer(nn.Module):
    def __init__(self, d, ul):
        """
        Layer in a linear module for a symplectic network. Parameterizes a small
        subset of linear mappings that are valid symplectic forms. These layers
        are composed into modules which are composed with other "symplectic
        modules" to form full symplectic networks.

        Parameters:
        -----------
        d: dimension of the linear layer
        ul: 'up' or 'low', upper or lower module

        """

        super().__init__()

        self.d = d

        self.A = nn.Parameter(torch.randn(d, d))

        self.S = torch.zeros(d, d)  # A + A^T
        self.M = torch.eye(2 * d, 2 * d)  # [ I, 0/S_i ; S_i/0, I]

        if ul != "up" and ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pq):
        self.S = self.A + self.A.t()

        if self.ul == "up":
            self.M[self.d :, : self.d] = self.S
        elif self.ul == "low":
            self.M[: self.d, self.d :] = self.S

        return self.M @ pq


class LinSympMod(nn.Module):
    def __init__(self, d, width, ul):
        """
        Linear module for symplectic networks (SympNets). Parameterizes a set
        of linear mappings that are valid symplectic forms. These modules are
        composed with other "symplectic modules" to form full sympletic
        networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        width: module width, i.e. number of transformations to compose
        ul: 'up' or 'low', upper or lower module (which to multiply by first)

        """

        super().__init__()

        self.d = d
        self.ul = ul
        self.width = width

        self.b = nn.Parameter(torch.randn(2 * d))

        # Compose linear symplectic layers. Note that this isn't exactly
        # the same notion as a layer in a regular neural network, this is
        # a "wide" sequence of multiplications.
        self.layers = []
        ul_rep = ul
        for i in range(width):
            self.layers.append(LinSympLayer(d, ul_rep))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq):
        return self.M(self.pq) + self.b


class ActSympMod(nn.Module):
    def __init__(self, d, ul, act):
        """
        Activation module for symplectic networks (SympNets). Parameterizes the
        set of activation functions that preserve symplecticity when applied to
        linear symplectic modules. These modules are composed with other
        "symplectic modules" to form full sympletic networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """
        super().__init__()

        self.d = d

        self.ul = ul
        self.act = act

        self.a = nn.Parameter(torch.randn(d))

        if ul != "up" and ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pq):
        p = pq[: self.d]
        q = pq[self.d :]

        if self.ul == "up":
            return torch.vstack([p + self.a * self.act(q), q])
        elif self.ul == "low":
            return torch.vstack([p, q + self.a * self.act(p)])


class GradSympMod(nn.Module):
    def __init__(self, d, width, ul, act):
        """
        Gradient module for symplectic networks (SympNets). Parameterizes a
        subset of quadratic forms that are valid symplectic forms. This is a
        potentially faster alternative to linear modules. These modules are
        composed with other "symplectic modules" to form full sympletic
        networks.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        width: module width, number of multiplications before summing
        ul: 'up' or 'low', upper or lower module
        act: activation function

        """

        super().__init__()

        self.d = d
        self.width = width

        self.ul = ul
        self.act = act

        self.K = nn.Parameter(torch.randn(width, d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))

        if self.ul != "up" and self.ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pq):
        p = pq[: self.d]
        q = pq[self.d :]

        if self.ul == "up":
            z = self.K.t() @ (self.a * self.act(self.K @ q + self.b))
            return torch.vstack([p + z, q])
        elif self.ul == "low":
            z = self.K.t() @ (self.a * self.act(self.K @ p + self.b))
            return torch.vstack([p, q + z])


class ExtSympMod(nn.Module):
    def __init__(self, n, d, width, ul, act):
        """
        Extended module for symplectic networks (SympNets). Parameterizes a set
        of quadratic + affine forms that are valid symplectic forms. These
        modules are composed with other "symplectic modules" to form full
        sympletic networks.

        Parameters:
        -----------
        n: total input dimension
        d: reduction dimension
        width: module width, number of multiplications before summing
        ul: 'up' or 'low', upper or lower module
        act: activation function
        
        """

        super().__init__()

        self.n = n
        self.d = d
        self.width = width
        self.ul = ul
        self.act = act

        self.K1 = nn.Parameter(torch.randn(width, d))
        self.K2 = nn.Parameter(torch.randn(width, n - 2 * d))
        self.a = nn.Parameter(torch.randn(width))
        self.b = nn.Parameter(torch.randn(width))

        if self.ul != "up" and self.ul != "low":
            raise ValueError("ul must be 'up' or 'low'")

    def forward(self, pqc):
        p = pqc[: self.d]
        q = pqc[self.d : 2 * self.d]
        c = pqc[2 * self.d :]

        if self.ul == "up":
            z = self.K1.t() * (self.a * self.act(self.K1 @ q + self.K2 @ c + self.b))
            return torch.vstack([p + z, q, c])
        elif self.ul == "low":
            z = self.K1.t() * (self.a * self.act(self.K1 @ p + self.K2 @ c + self.b))
            return torch.vstack([p, q + z, c])


class LASympNet(nn.Module):
    def __init__(self, d, nlayers, subwidth, act=nn.ReLU):
        """
        Linear + activation module based symplectic network. Made of a set of
        linear symplectic modules and a set of activation modules. 

        TODO determine why in the paper they don't use an activation on the
        last layer.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        nlayers: number of layers of act(lin()) modules
        subwidth: width of each linear module

        """

        super().__init__()

        self.d = d
        self.nlayers = nlayers
        self.subwidth = subwidth

        ul_rep = "up"
        self.layers = []
        for i in nlayers:
            self.layers.append(LinSympMod(d, subwidth, ul_rep))
            self.layers.append(ActSympMod(d, ul_rep, act))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq):
        return self.M(pq)


class GSympNet(nn.Module):
    def __init__(self, d, nlayers, subwidth, act=nn.ReLU):
        """
        Gradient module based symplectic network. Made of a set of gradient
        symplectic modules.

        Parameters:
        -----------
        d: half the transformation dim (2d latent dim)
        nlayers: number of layers of act(lin()) modules
        subwidth: width of each linear module
        """

        super().__init__()

        self.d = d
        self.nlayers = nlayers
        self.subwidth = subwidth

        ul_rep = "up"
        self.layers = []
        for i in nlayers:
            self.layers.append(GradSympMod(d, subwidth, ul_rep, act))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pq):
        return self.M(pq)


class ESympNet(nn.Module):
    def __init__(self, n, d, nlayers, subwidth, act=nn.ReLU):
        """
        Extended module based symplectic network. Made of a set of extended
        symplectic modules.

        Parameters:
        -----------
        n: total input dimension
        d: reduction dimension
        nlayers: number of layers of act(lin()) modules
        subwidth: width of each linear module
        """

        super().__init__()

        self.n = n
        self.d = d
        self.nlayers = nlayers
        self.subwidth = subwidth

        ul_rep = "up"
        self.layers = []
        for i in nlayers:
            self.layers.append(ExtSympMod(n, d, subwidth, ul_rep, act))
            ul_rep = "low" if ul_rep == "up" else "up"
        self.M = nn.Sequential(*self.layers)

    def forward(self, pqc):
        return self.M(pqc)
