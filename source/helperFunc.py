import ufl 
import numpy as np

def all(x):
    """returns an array of value True with the size of the second dimension of x

        Parameters
        ----------
        x : array_like
            input array

        Returns
        -------
        val : ndarray
            array of True values with shape 
        """
    val = np.full(np.shape(x)[1], True)
    return val

def filterDict(dictAll, validKeys):
    """filters dict for all pairs where keys are part of validKeys

    Parameters
    ----------
    dictAll : dict
        ditctionary that shall be filtered
    validKeys : list
        list of valid keys for filtering

    Returns
    -------
    dict
        fitered dict
    """
    dictFiltered = dict()
    for key in validKeys:
        if key in dictAll:
            try:
                dictFiltered[key] = float(dictAll[key])
            except ValueError:
                dictFiltered[key] = dictAll[key]
    return dictFiltered

def degrad(s, eta):
    """creates the form for degradation function (s^2 + eta)

    Parameters
    ----------
    s : fem.form
        phase-field parameter
    eta : fem.constant
        residual stiffness (numerical parameter)

    Returns
    -------
    degrad : fem.form
        form for degradation function
    """
    degrad = s**2+eta
    return degrad

def positive(val):
    """sets negative values in form to 0

    Parameters
    ----------
    val : fem.form
        input value

    Returns
    -------
    pos : fem.form
        val if val > 0 else 0
    """
    return 0.5*(val+ufl.sign(val)*val)

def negative(val):
    """sets positive values in form to 0

    Parameters
    ----------
    val : fem.form
        input value

    Returns
    -------
    pos : fem.form
        val if val < 0 else 0
    """
    return 0.5*(val-ufl.sign(val)*val)

def eps(u):
    """creates the form for the infinitesimal strain tensor (eps = grad_sym(u))

    Parameters
    ----------
    u : fem.form
        displacement field

    Returns
    -------
    eps : fem.form
        infenitesimal strain tensor
    """
    eps = ufl.sym(ufl.grad(u))
    return eps

def sigma(u, lam, dim, mu):
    """creates the form for the stress tensor using an isotropic constitutive equation

    Parameters
    ----------
    u : fem.form
        displacement field
    lam : fem.constant
        lambda (lamé constant)
    dim : int
        dimension of space
    mu : fem.constant
        mu (lamé constant)

    Returns
    -------
    sigma : fem.form
        stress tensor
    """
    return lam * ufl.tr(eps(u)) * ufl.Identity(dim) + 2 * mu * eps(u)

def eigenstate2(A):
    """Eigenvalues of the 2x2 (real-valued) tensor A.

    Provides the spectral decomposition A = sum_{a=0}^{1} λ_a * E_a
    with (ordered) eigenvalues λ_a

    Note: Tensor A must not have complex eigenvalues!
    This function was adapted from https://github.com/fenics-dolfiny/dolfiny/blob/main/src/dolfiny/invariants.py#L72-L309
    Calculations unnecessary for the present usage were stripped from the code.
    """
    if ufl.shape(A) != (2, 2):
        raise RuntimeError(f"Tensor A of shape {ufl.shape(A)} != (2, 2) is not supported!")
    #
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    #
    A = ufl.variable(A)
    #
    # --- determine eigenvalues λ0, λ1
    #
    I1 = ufl.tr(A)
    #
    Δ = (A[0, 0] - A[1, 1]) ** 2 + 4 * A[0, 1] * A[1, 0]  # = I1**2 - 4 * I2
    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    Δ += eps**2
    # sorted eigenvalues: λ0 <= λ1
    λ = (I1 - ufl.sqrt(Δ)) / 2, (I1 + ufl.sqrt(Δ)) / 2

    return λ


def psielP_Lo(eps, lam, mu, nu, E):
    """creates the form for the part of the elastic energy density that should be degraded

    Parameters
    ----------
    eps : fem.form
        infinitesimal strain tensor
    lam : fem.constant
        lamé's first constant
    mu : fem.constant
        lamé's second constant
    nu : fem.constant
        Poisson's ratio
    E : fem.constant
        Young's modulus
    Returns
    -------
    psielP : fem.form
        form for the part of the elastic energy density that should be degraded
    """
    (eps_2, eps_1) = eigenstate2(eps)

    # implement  Lo's scheme (see https://doi.org/10.1016/j.eml.2021.101277)
    comb = positive((1 - nu) * eps_1 + nu * eps_2)                                 # 0 when not positive -> fulfill whole condition in paper 

    case1Condition = positive(((ufl.sign(eps_1)+ufl.sign(eps_2))/2))               # condition -> 1 if both positive, zero else
    case1Val = ((lam / 2) * (eps_1 + eps_2)**2 + mu * (eps_1**2 + eps_2**2))     

    case2Condition = positive(ufl.sign(eps_1)) * (-1.0*negative(ufl.sign(eps_2)))         # condition -> 1 if eps_1 positive and eps_2 negative, zero else        
    case2Val = (E * comb**2) / (2 * (1 - 2*nu) * (1 - nu**2))                      

    return case1Condition * case1Val + case2Condition*case2Val

def psielM_Lo(eps,lam,mu,nu,E):
    """creates the form for the part of the elastic energy density that should not be degraded

    Parameters
    ----------
    eps : fem.form
        infinitesimal strain tensor
    lam : fem.constant
        lamé's first constant
    mu : fem.constant
        lamé's second constant
    nu : fem.constant
        Poisson's ratio
    E : fem.constant
        Young's modulus
    Returns
    -------
    psielP : fem.form
        form for the part of the elastic energy density that should not be degraded
    """
    return ((lam/2)*ufl.tr(eps)**2 + mu * ufl.tr(eps*eps)) - psielP_Lo(eps, lam, mu, nu, E)


def psifrac(s, G_c, l_0):
    """creates the form for the fracture energy density

    Parameters
    ----------
    s : fem.form
        phase-field parameter
    G_c : fem.constant
        critical fracture energy release rate
    l_0 : fem.constant
        regularization parameter
    Returns
    -------
    psifrac : fem.form
        form for the fracture energy density
    """
    psifrac = G_c * (((1-s)**2)/(4*l_0)+l_0*(ufl.dot(ufl.grad(s), ufl.grad(s))))
    return psifrac

def psigrav(u,f):
    """creates the form for the gravitation energy density

    Parameters
    ----------
    u : fem.form
        displacement field
    f : fem.constant
        gravitation vector (rho*g*direction)
    Returns
    -------
    psigrav : fem.form
        form for the gravitation energy density
    """
    psigrav = ufl.dot(f,u)
    return psigrav

def psiwater(u,p):
    """creates the form for the water pressure energy density

    Parameters
    ----------
    u : fem.form
        displacement field
    p : fem.function
        gravitation vector (rho*g*direction)
    Returns
    -------
    psigrav : fem.form
        form for the gravitation energy density
    """
    psiwater = ufl.div(p*u)
    return psiwater

def stress(eps, s, eta, nu, E, lam,  mu):
    """creates the form for the stress tensor using the derivative of the strain energy density w.r.t. epsilon

    Parameters
    ----------
    eps : fem.form
        infinitesimal strain tensor
    s : fem.form
        phase-field parameter
    eta : fem.constant
        residual stiffness
    nu : fem.constant
        Poisson's ratio
    E : fem.constant
        Young's modulus
    lam : fem.constant
        lamé's first constant
    mu : fem.constant
        lamé's second constant
    Returns
    -------
    stress : fem.form
        form for the stress tensor
    """
    eps_var = ufl.variable(eps)
    s_var = ufl.variable(s)
    stress = ufl.diff(psielM_Lo(eps_var, lam, mu, nu, E)+ degrad(s_var,eta) * psielP_Lo(eps_var, lam, mu, nu, E), eps_var)
    return stress