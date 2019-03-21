import numpy as np
"""
import matplotlib
import matplotlib.pyplot as plt
"""
from numpy import cos, sin, pi
from numpy.random import random
from utils_numeric import solve_linear
from finite_volumes import get_Y_star, integrate_one_step_star, integrate_one_step

"""
    Tests the function "integrate_one_step" of finite_volumes.
    h and D are constant, a and c are != 0.
    We don't care about interface here.
    We compute full solution and compare it to half solutions
    when the bd condition is exact
"""
def test_integrate_half_domain():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0

    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 0.0000001
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.4

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f should be an average and not a local approximation
    f = c * u_theoric + a*partial_xu
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = u_theoric # np.zeros_like(u_theoric)
    u_n, u_interface, phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u0=u0)

    u_np1, real_u_interface, real_phi_interface = integrate_one_step_star(M1=M1, \
            M2=M2, h1=h1, h2=h2, D1=D1,
            D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
            neumann=neumann, dirichlet=dirichlet, u0=u_n)

    real_u_interface, real_phi_interface = u_interface, phi_interface
    u1_n = u_n[:M1]
    u2_n = u_n[M1:]
    u1_np1 = u_np1[:M1]
    u2_np1 = u_np1[M1:]

    # Schwarz:
    value_interface = 0.0

    Lambda_1 = 0.0
    Lambda_2 = 1.0

    u1_n = u1_n[::-1]
    u1_np1 = u1_np1[::-1]

    """
    # Schwarz:
    for i in range(200):
        print("robin condition:", Lambda_1*u_interface + phi_interface)
        print("interface was (", u_interface, phi_interface,"): ")
        u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                bd_cond=neumann, Lambda=Lambda_1, u_nm1=u2_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=True)
        print(u_interface, phi_interface)
        print("robin condition:", Lambda_1*u_interface + phi_interface)
        print("robin condition:", Lambda_2*u_interface + phi_interface)
        u2_plot = np.concatenate(([u_interface],u2_ret))
        u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                bd_cond=dirichlet, Lambda=Lambda_2, u_nm1=u1_n,
                u_interface=u_interface, phi_interface=phi_interface,
                upper_domain=False)
        print("robin condition:", Lambda_2*u_interface + phi_interface)
    """

    u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
            h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
            bd_cond=neumann, Lambda=Lambda_1, u_nm1=u2_n,
            u_interface=u_interface, phi_interface=phi_interface,
            upper_domain=True)
    u2_plot = np.concatenate(([u_interface],u2_ret))
    u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
            h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
            bd_cond=dirichlet, Lambda=Lambda_2, u_nm1=u1_n,
            u_interface=u_interface, phi_interface=phi_interface,
            upper_domain=False)


    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    fig, ax = plt.subplots()
    ax.plot(x, u_np1, "g")
    u1_plot = np.concatenate(([u_interface], u1_ret))

    x1_plot = np.concatenate(([0],x1[::-1]))
    x2_plot = np.concatenate(([0], x2))
    line_1, = ax.plot(x1_plot, u1_plot, "r")
    line_2, = ax.plot(x2_plot, u2_plot, "b")
    u_inter = u_interface
    phi_inter = phi_interface
    def animate(k):
        u_interface=u_inter
        phi_interface=phi_inter
        for i in range(k):
            u2_ret, u_interface, phi_interface = integrate_one_step(M=M2,
                    h=h2, D=D2, a=a, c=c, dt=dt, f=f2,
                    bd_cond=neumann, Lambda=Lambda_1, u_nm1=u2_n,
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=True)
            u2_plot = np.concatenate(([u_interface],u2_ret))
            u1_ret, u_interface, phi_interface = integrate_one_step(M=M1,
                    h=h1, D=D1, a=a, c=c, dt=dt, f=f1,
                    bd_cond=dirichlet, Lambda=Lambda_2, u_nm1=u1_n,
                    u_interface=u_interface, phi_interface=phi_interface,
                    upper_domain=False)
            u1_plot = np.concatenate(([u_interface], u1_ret))
        line_1.set_ydata(u1_plot)  # update the data
        line_2.set_ydata(u2_plot)  # update the data
        return line_1, line_2

    # Init only required for blitting to give a clean slate.
    def init():
        line_1.set_ydata(np.ma.array(x1_plot, mask=True))
        line_2.set_ydata(np.ma.array(x2_plot, mask=True))
        return line_1, line_2

    ani = animation.FuncAnimation(fig, animate, np.arange(1, 700), init_func=init,
                                  interval=1, blit=True)
    plt.show()


    return "ok"

"""
    Tests the function "integrate_one_step_star" of finite_volumes.
    h and D are constant, a and c are != 0.
    D is discontinuous over the interface.
    We make multi steps to reach convergence rather
    than one big step
"""
def test_integrate_multi_step_with_a_c():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 1
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.1

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    f = c * u_theoric + a*partial_xu
    # Note: f should be an average and not a local approximation
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u = np.zeros_like(u_theoric)
    for i in range(13):
        u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                    D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                    neumann=neumann, dirichlet=dirichlet, u0=u)[0]

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"


"""
    Tests the function "integrate_one_step_star" of finite_volumes.
    h and D are constant, a and c are != 0.
    D is discontinuous over the interface.
"""
def test_integrate_one_step_with_a_c():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 10, 10
    h1, h2 = 1/M1, 1/M2
    dt = 10000000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))

    D1 = 2.2
    D2 = 1.2
    a = 1.1
    c = 0.4

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    partial_xu = pi * np.concatenate((np.ones_like(x1) * D2/D1,
        np.ones_like(x2)))
    neumann = pi
    dirichlet = -pi
    # Note: f should be an average and not a local approximation
    f = c * u_theoric + a*partial_xu
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = np.zeros_like(u_theoric)
    u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                neumann=neumann, dirichlet=dirichlet, u0=u0)[0]

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"


"""
    Tests the function "integrate_one_step_star" of finite_volumes.
    h and D are constant, a and c are zeros.
    D is discontinuous over the interface.
    The functions used are extremely simple because it's not trivial
    enough to have an analytical solution
"""
def test_integrate_one_step_simplest():
    # Our domain is [0,1]
    # first function : 
    # u = -pi(1 - D2/D1) + pi*x         if x>0
    # u = -pi(1 - D2/D1) + pi*x*D2/D1   if x<0
    M1, M2 = 20, 20
    h1, h2 = 1/M1, 1/M2
    dt = 100000
    # Center of the volumes are x, sizes are h
    x1 = np.linspace(-1 + h1/2, -h1 / 2, M1)
    x2 = np.linspace(h2/2, 1 - h2 / 2, M2)
    x = np.concatenate((x1, x2))


    D1 = 2.4
    D2 = 1.4
    a = 0.0
    c = 0.0

    u_theoric = -pi*(1-D2/D1) + np.concatenate((pi*x1*D2/D1, pi*x2))
    neumann = pi
    dirichlet = -pi
    f = np.zeros_like(x)
    f1 = f[:M1]
    f1 = f1[::-1]
    f2 = f[M1:]

    u0 = np.zeros_like(x)
    u = integrate_one_step_star(M1=M1, M2=M2, h1=h1, h2=h2, D1=D1,
                D2=D2, a=a, c=c, dt=dt, f1=f1, f2=f2,
                neumann=neumann, dirichlet=dirichlet, u0=u0)[0]

    u1 = u[:M1]
    u2 = u[M1+1:]

    assert abs(-pi*(1-D2/D1) + pi*(1-h2/2) - u[-1]) < 1e-6
    assert np.linalg.norm(u_theoric - u) < 2e-5
    assert np.linalg.norm(np.diff(u1) - pi*h1*D2/D1) < 1e-6
    assert np.linalg.norm(np.diff(u2) - pi*h2) < 1e-6
    return "ok"

def test_integrate_one_step():
    assert "ok" == test_integrate_one_step_simplest()
    assert "ok" == test_integrate_one_step_with_a_c()
    assert "ok" == test_integrate_multi_step_with_a_c()
    assert "ok" == test_integrate_half_domain()
    return "ok"

def launch_all_tests():
    print("Test of compact scheme.")
    print("Test integration with finite volumes:", test_integrate_one_step())

