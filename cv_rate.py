#!/usr/bin/python3
import numpy as np
from numpy import pi, cos, sin
from discretizations.finite_difference import FiniteDifferences
from discretizations.finite_volumes import FiniteVolumes
import functools

LAMBDA_1_DEFAULT=0.0
LAMBDA_2_DEFAULT=0.0

A_DEFAULT=0.0
C_DEFAULT=1e-10
D1_DEFAULT=.6
D2_DEFAULT=.54

# 35 min with parameters T=1000dt, M1=400, M2=40, [lambda_]steps:200
TIME_WINDOW_LEN_DEFAULT=128
DT_DEFAULT=0.1

M1_DEFAULT= 8000
M2_DEFAULT= 800

SIZE_DOMAIN_1 = 200
SIZE_DOMAIN_2 = 20


def main():
    import sys

    if len(sys.argv) == 1:
        print("to launch tests, use \"python3 cv_rate.py test\"")
        print("Usage: cv_rate {test, graph, optimize, debug, analytic}")
    else:
        # defining discretization:

        finite_difference = FiniteDifferences(A_DEFAULT, C_DEFAULT,
                D1_DEFAULT, D2_DEFAULT,
                M1_DEFAULT, M2_DEFAULT, SIZE_DOMAIN_1, SIZE_DOMAIN_2,
                LAMBDA_1_DEFAULT, LAMBDA_2_DEFAULT, DT_DEFAULT)

        finite_volumes = FiniteVolumes(A_DEFAULT, C_DEFAULT,
                D1_DEFAULT, D2_DEFAULT,
                M1_DEFAULT, M2_DEFAULT, SIZE_DOMAIN_1, SIZE_DOMAIN_2,
                LAMBDA_1_DEFAULT, LAMBDA_2_DEFAULT, DT_DEFAULT)

        if sys.argv[1] == "test":
            import tests.test_linear_sys
            import tests.test_schwarz
            import tests.test_finite_volumes
            tests.test_linear_sys.launch_all_tests()
            tests.test_schwarz.launch_all_tests()
            tests.test_finite_volumes.launch_all_tests()
        elif sys.argv[1] == "graph":
            import matplotlib.pyplot as plt
            lambda_min = 1e-9
            lambda_max = 10
            steps = 30
            beauty_graph_finite(finite_difference, lambda_min, lambda_max, steps)
        elif sys.argv[1] == "optimize":
            from scipy.optimize import minimize_scalar
            print("rate finite volumes:", minimize_scalar(functools.partial(rate,
                                                                finite_volumes)))
            print("rate finite differences:", minimize_scalar(functools.partial(rate,
                                                                finite_difference)))

            D1 = D1_DEFAULT
            D2 = D2_DEFAULT
            def theory_star(wmin, wmax):
                a = (np.sqrt(D1) - np.sqrt(D2)) * (np.sqrt(wmin)+np.sqrt(wmax))
                return 1/(2*np.sqrt(2))*(a + np.sqrt(a*a + 8*np.sqrt(D1*D2)*np.sqrt(wmin*wmax)))
            print("theory:", theory_star(pi/DT_DEFAULT, pi/(DT_DEFAULT*TIME_WINDOW_LEN_DEFAULT)))
        elif  sys.argv[1] == "debug":
            #print(analytic_robin_robin_finite_differences(w=None, Lambda_1=-5, verbose=True))
            #rate_finite_differences2(-5)
            lambda_min = 1e-9
            lambda_max = 10
            steps = 0
            lambda_1 = np.linspace(lambda_min, lambda_max, steps)
            import matplotlib.pyplot as plt
            plt.plot(lambda_1, [analytic_robin_robin(finite_difference, w=0,Lambda_1=i) \
                               for i in lambda_1], "y")
        elif  sys.argv[1] == "ztransform":
            #print(analytic_robin_robin_finite_differences(w=None, Lambda_1=-5, verbose=True))
            #rate_finite_differences2(-5)


            lambda_min = 1e-9
            lambda_max = 10
            steps = 40
            lambda_1 = np.linspace(lambda_min, lambda_max, steps)
            import matplotlib.pyplot as plt
            plt.plot(lambda_1, [analytic_robin_robin(finite_difference, w=pi/DT_DEFAULT ,Lambda_1=i) \
                               for i in lambda_1], "y")
            plt.plot(lambda_1, [rate_by_z_transform(finite_difference, i) for i in lambda_1], "r")
            plt.show()
        elif sys.argv[1] == "analytic":
            print("TODO: lire rapport de Sophie et trouver quels tests faire !")
            import matplotlib.pyplot as plt
            lambda_min = 1e-9
            lambda_max = 10
            steps = 100
            beauty_graph_finite(finite_difference, lambda_min, lambda_max, True, steps)

def analytic_bounds(analytic_func, Lambda_1, wmax):
    from scipy.optimize import minimize_scalar
    cost_function_min = lambda t:analytic_func(w=t, Lambda_1=Lambda_1)
    cost_function_max = lambda t:-analytic_func(w=t, Lambda_1=Lambda_1)
    ret_min = minimize_scalar(method='bounded', fun=cost_function_min, bounds=(-wmax, wmax)).fun
    ret_max = -minimize_scalar(method='bounded', fun=cost_function_max, bounds=(-wmax, wmax)).fun
    assert ret_max >= -cost_function_max(0.) 
    return ret_min, ret_max


PARALLEL = True
def beauty_graph_finite(discretization, lambda_min, lambda_max, is_differences, steps=100):
    rate_func = functools.partial(rate, discretization)
    rate_func_normL2 = functools.partial(rate, discretization,
            function_to_use=np.linalg.norm)

    import matplotlib.pyplot as plt
    from scipy.optimize import minimize_scalar, minimize

    lambda_1 = np.linspace(lambda_min, np.sqrt(lambda_max), steps)**2
    dt = DT_DEFAULT
    T = dt*TIME_WINDOW_LEN_DEFAULT
    print("> Starting frequency analysis.")
    rho = []
    for t in np.arange( dt, T, dt/20):
        rho += [[analytic_robin_robin(discretization, w=pi/t,Lambda_1=i) \
               for i in lambda_1]]
        rho += [[analytic_robin_robin(discretization, w=-pi/t,Lambda_1=i) \
               for i in lambda_1]]

    def analytic_rate_to_minimize(lam):
        # let's maximize over the frequency:
        to_minimize = lambda t:-analytic_robin_robin(discretization, w=t, Lambda_1=lam)
        ret_ = minimize_scalar(method="bounded", bounds=(pi/T, pi/dt), fun=to_minimize)
        ret2_ = minimize_scalar(method="bounded", bounds=(-pi/dt, pi/T), fun=to_minimize)
        return max(-ret_.fun, -ret2_.fun)

    sqD1 = np.sqrt(D1_DEFAULT)
    sqD2 = np.sqrt(D2_DEFAULT)
    sqw1 = np.sqrt(pi/T)
    sqw2 = np.sqrt(pi/dt)
    continuous_best_lam = 1/(2*np.sqrt(2)) * ((sqD2-sqD1)*(sqw1+sqw2) + np.sqrt((sqD2-sqD1)**2 * (sqw1 + sqw2)**2 + 8*sqD1*sqD2*sqw1*sqw2))
    print("In continuous framework: best $\\Lambda$ is", continuous_best_lam)

    min_rho, max_rho = np.min(np.array(rho), axis=0), np.max(np.array(rho),axis=0)
    best_analytic = lambda_1[np.argmin(max_rho)]

    plt.fill_between(lambda_1, min_rho, max_rho, facecolor="green", label="pi/T < |w| < pi/dt")
    plt.vlines(best_analytic, 0,1, "g", 'dashed', label='best $\\Lambda$ with frequency analysis' )
    plt.plot(lambda_1, [analytic_robin_robin(discretization, w=0,Lambda_1=i) \
                       for i in lambda_1], "y")
    plt.vlines(continuous_best_lam, 0,1, "k", 'dashed', label='best $\\Lambda$ in continuous case' )
    plt.plot(lambda_1, [rate_by_z_transform(discretization, i) for i in lambda_1], "m")
    plt.plot(lambda_1, [rate_by_z_transform(discretization, i) for i in lambda_1], "m", label="Z TRANSFORMED ANALYTIC RATE")
    rho = []
    for logt in np.arange(0 , 25):
        t = dt * 2.**logt
        rho += [[analytic_robin_robin(discretization, w=pi/t,Lambda_1=i) \
               for i in lambda_1]]
        rho += [[analytic_robin_robin(discretization, w=-pi/t,Lambda_1=i) \
               for i in lambda_1]]
    plt.fill_between(lambda_1, np.min(np.array(rho), axis=0), np.max(np.array(rho), axis=0), facecolor="grey", label="|w| < pi/dt", alpha=0.4)

    print("> Starting simulations")

    x = np.linspace(lambda_min, lambda_max, steps)
    if PARALLEL:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor() as executor:
            rate_f = executor.map(rate_func, x)
            rate_f_L2 = executor.map(rate_func_normL2, x)
    else:
        rate_f = map(rate_func, x)
        rate_f_L2 = map(rate_func_normL2, x)

    print("> Starting minimization in infinite norm.")

    best_linf_norm = minimize_scalar(method="bounded", bounds=(lambda_min, lambda_max), fun=rate_func).x
    print("> Starting minimization in L2 norm.")
    best_L2_norm = minimize_scalar(method="bounded", bounds=(lambda_min, lambda_max), fun=rate_func_normL2).x
    plt.plot(x, list(rate_f_L2), "b", label=discretization.name() + ", $L^2$ norm")
    plt.vlines(best_L2_norm, 0, 1, "b", 'dashed', label='best $\\Lambda$ for $L^2$' )
    plt.plot(x, list(rate_f), "r", label=discretization.name() + ", $L^\\infty$ norm")
    plt.vlines(best_linf_norm, 0,1, "r", 'dashed', label='best $\\Lambda$ for $L^\\infty$' )

    plt.xlabel("$\\Lambda^1$")
    plt.ylabel("$\\rho$")

    #plt.plot(lambda_1, [rate_finite_differences_by_z_transform(i) for i in lambda_1], "r--")
    #plt.plot(lambda_1, 
    #       [analytic_robin_robin_finite_volumes(Lambda_1=i) \
    #               for i in lambda_1], "k--", label="Theorical rate")
    plt.title("Global in time: T="+str(TIME_WINDOW_LEN_DEFAULT)+ \
            "dt\n $D_1$="+str(D1_DEFAULT)+", $D_2$="+str(D2_DEFAULT)+", " + \
            "$0=a\\approx c$, $H_1$=-"+str(SIZE_DOMAIN_1)+", $H_2$="+str(SIZE_DOMAIN_2)+"\n" + \
            "$\\Lambda^2$="+str(LAMBDA_2_DEFAULT)+",  dt="+str(DT_DEFAULT)+", h=0.5")

    plt.legend()
    plt.show()

def rate_by_z_transform(discretization, Lambda_1):
    NUMBER_SAMPLES = TIME_WINDOW_LEN_DEFAULT
    all_points = np.linspace(0, 2*pi, NUMBER_SAMPLES, endpoint=False)
    dt=DT_DEFAULT
    z_transformed = lambda z:discretization.analytic_robin_robin(s=1./dt*(z-1)/z,
            Lambda_1=Lambda_1)
    r = 1.001
    samples = [z_transformed(r*np.exp(p*1j)) for p in all_points]
    ret_ifft = np.fft.ifft(np.array(samples))
    rate_each_time = [r**n * l for n, l in enumerate(ret_ifft)]
    return np.max(np.abs(rate_each_time))


def analytic_robin_robin(discretization, w=None, Lambda_1=LAMBDA_1_DEFAULT,
        Lambda_2=LAMBDA_2_DEFAULT, a=A_DEFAULT, 
        c=C_DEFAULT, dt=DT_DEFAULT, M1=M1_DEFAULT, M2=M2_DEFAULT,
        D1=D1_DEFAULT, D2=D2_DEFAULT, verbose=False):
    if w is None:
        s = 1./dt
    else:
        s = w*1j
    return discretization.analytic_robin_robin(s=s, Lambda_1=Lambda_1,
            Lambda_2=Lambda_2, a=a, c=c, dt=dt, M1=M1, M2=M2,
            D1=D1, D2=D2, verbose=verbose)


def rate(discretization, Lambda_1=LAMBDA_1_DEFAULT, Lambda_2=LAMBDA_2_DEFAULT,
        a=A_DEFAULT, c=C_DEFAULT, time_window_len=TIME_WINDOW_LEN_DEFAULT,
                dt=DT_DEFAULT, M1=M1_DEFAULT, M2=M2_DEFAULT, function_to_use=max):

    h1, h2 = discretization.get_h(SIZE_DOMAIN_1, SIZE_DOMAIN_2, M1, M2)
    D1, D2 = discretization.get_D(h1, h2)

    f1 = np.zeros(M1)
    f2 = np.zeros(M2)
    neumann = 0
    dirichlet = 0

    precomputed_Y1 = discretization.precompute_Y(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_1, upper_domain=False)

    precomputed_Y2 = discretization.precompute_Y(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_2, upper_domain=True)

    # random false initialization:
    u1_0 = np.zeros(M1)
    u2_0 = np.zeros(M2)
    error = []
    np.random.seed(9380)
    all_u1_interface = 2*(np.random.rand(time_window_len) - 0.5)
    all_phi1_interface = 2*(np.random.rand(time_window_len) - 0.5)
    # Beginning of schwarz iterations:
    for k in range(2):
        all_u2_interface = []
        all_phi2_interface = []
        all_u2 =  [u2_0]
        # Time iteration:
        for i in range(time_window_len):
            u_interface = all_u1_interface[i]
            phi_interface = all_phi1_interface[i]

            u2_ret, u_interface, phi_interface = discretization.integrate_one_step(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_2, u_nm1=all_u2[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=True, Y=precomputed_Y2)
            all_u2 += [u2_ret]
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = []
        all_phi1_interface = []
        all_u1 = [u1_0]

        for i in range(time_window_len):

            u_interface = all_u2_interface[i]
            phi_interface = all_phi2_interface[i]

            u1_ret, u_interface, phi_interface = discretization.integrate_one_step(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_1, u_nm1=all_u1[-1],
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=False, Y=precomputed_Y1)
            all_u1 += [u1_ret]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

        error += [function_to_use([abs(e) for e in all_phi1_interface])]

    return error[1] / error[0]

if __name__ == "__main__":
    main()
