def A(sigma, tau, nu, zeta):
return (nu*dot(sigma, tau) - zeta*trace(sigma)*trace(tau))*dx
def b(tau, w, eta):
return (div(tau[0])*w[0] + div(tau[1])*w[1] + skew(tau)*eta)*dx
nu = 0.5
zeta = 0.2475 r=2
S = FiniteElement("BDM", "triangle", r)
V = VectorElement("Discontinuous Lagrange", "triangle", r-1) Q = FiniteElement("Discontinuous Lagrange", "triangle", r-1) MX = MixedElement([S, S, V, Q])
(tau0, tau1, v, eta) = TestFunctions(MX) (sigma0, sigma1, u, gamma) = TrialFunctions(MX) sigma = [sigma0, sigma1]
tau = [tau0, tau1]
a = A(sigma, tau, nu, zeta) + b(tau, u, gamma) + b(sigma, v, eta) L = dot(v, f)*dx
