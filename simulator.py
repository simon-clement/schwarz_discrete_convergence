#!/usr/bin/python3
"""
    This module provides functions to observe real convergence factors.
"""
import time
import functools
import numpy as np
from numpy import pi

def linear_regression_cplx(x, output_param):
    """ to perform a linear regression in the complex domain with scikit fctions,
    we remark that the real part of the outputs are of the form
    Sum( Re(model_i)*Re(x_i) - Im(model_i)*Im(x_i)).
    So we replace (x1, x2, ...) by (Re(x1), -Im(x1), Re(x2), -Im(x2), ...)
    and we do similar reasoning for imaginary part.
    """
    from sklearn.linear_model import LinearRegression
    x_not_complex = np.zeros((2*x.shape[0], 2*x.shape[1]))
    for kth_sample in range(x.shape[0]):
        for ith_coord in range(x.shape[1]):
            # (x1, x2, ...) -> (Re(x1), -Im(x1), Re(x2), -Im(x2), ...)
            x_not_complex[kth_sample, 2*ith_coord] = np.real(x[kth_sample, ith_coord])
            x_not_complex[kth_sample, 2*ith_coord+1] = -np.imag(x[kth_sample, ith_coord])

    for kth_sample in range(x.shape[0]):
        for ith_coord in range(x.shape[1]):
            x_not_complex[x.shape[0] + kth_sample, 2*ith_coord] = np.imag(x[kth_sample, ith_coord])
            x_not_complex[x.shape[0] + kth_sample, 2*ith_coord+1] = np.real(x[kth_sample, ith_coord])
    output_total = np.concatenate((np.real(output_param), np.imag(output_param)))

    model_reals = LinearRegression(n_jobs=-1).fit(x_not_complex, output_total).coef_
    # Actual linear regression in imaginary numbers:
    model_cplx = 1j*np.ones(model_reals.shape[0] // 2)
    for ith_coord in range(model_cplx.shape[0]):
        model_cplx[ith_coord] = model_reals[2*ith_coord] + 1j*model_reals[2*ith_coord+1]
    return model_cplx

def linear_regression_1D(errors):
    """
    that relies the data in X[:,1] to the data in X[:,2].
    the tuple is made of several arrays of size X.shape[-1].
    X[:,k] is the k-th input, and Y[k] is the k-th output.
    """
    # 1D:
    res = np.zeros_like(errors[0,1])
    for i in range(errors.shape[2]):
        input_params = errors[:,1,i]
        x = np.array([(input_param,) for input_param in input_params])
        output_params = errors[:,2,i]
        res[i] = linear_regression_cplx(x, output_params)[0]
    return res

def linear_regression_2D(errors_u, errors_phi):
    """
    gives a 2x2 matrix
    that relies the data in (errors_u,errors_phi)[:,1] to the data in (errors_u, errrors_phi)[:,2].
    the tuple is made of several arrays of size X.shape[-1].
    X[:,k] is the k-th input, and Y[k] is the k-th output.
    """
    # 1D:
    from sklearn.linear_model import LinearRegression
    transition_11 = np.zeros_like(errors_u[0,1])
    transition_12 = np.zeros_like(errors_u[0,1])
    transition_21 = np.zeros_like(errors_u[0,1])
    transition_22 = np.zeros_like(errors_u[0,1])
    for i in range(errors_phi.shape[2]):
        x = np.array([(errors_u[k,1,i], errors_phi[k,1,i]) for k in range(errors_phi.shape[0])])
        output_param_1 = errors_u[:,2,i]
        output_param_2 = errors_phi[:,2,i]
        transition_11[i], transition_12[i] = linear_regression_cplx(x, output_param_1)
        transition_21[i], transition_22[i] = linear_regression_cplx(x, output_param_2)
    return [[transition_11, transition_12], [transition_21, transition_22]]


def eigenvalues_matrixlinear_frequency_simulation(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(interface_errors, discretization, N, **kwargs)
    print(number_samples, "samples")

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        errors_u = []
        errors_phi = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors_u += [result[0]]
            errors_phi += [result[1]]
    freq_err_u = fftshift(fft(np.array(errors_u), axis=-1), axes=(-1, ))
    freq_err_phi = fftshift(fft(np.array(errors_phi), axis=-1), axes=(-1, ))

    def det_uj_over_u1(k, j, u, phi):
        """
            for j=2 or 3, returns the ration of the determinants
            of U_k^j over U_k^1.
            with D1, D2 the eigenvalues of the transition matrix,
            the result of this function:
            -for j=1, is D1+D2
            - for j=2, is D2^2 + D1^2 + D1*D2
        """
        return (u[:, k] * phi[:, k+j] - phi[:, k] * u[:,k+j]) / (u[:,k] * phi[:,k+1] - phi[:,k] * u[:,k+1])
    
    R3 = det_uj_over_u1(1, 2, freq_err_u, freq_err_phi)
    R3 = det_uj_over_u1(1, 3, freq_err_u, freq_err_phi)
    assert not np.isnan(R2).any()
    assert not np.isnan(R3).any()
    from numpy.lib import scimath
    root = scimath.sqrt(4 * R3 - 3*R2*R2)
    D1 = np.copy(R2)
    D2 = np.copy(R2)
    np.add(R2, root, where=(np.abs(R2+root)>np.abs(R2-root)), out=D1)
    np.add(R2, -root, where=(np.abs(R2+root)<=np.abs(R2-root)), out=D1)
    np.add(R2, -root, where=(np.abs(R2+root)>np.abs(R2-root)), out=D2)
    np.add(R2, root, where=(np.abs(R2+root)<=np.abs(R2-root)), out=D2)

    return D1, D2

def matrixlinear_frequency_simulation(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(interface_errors, discretization, N, **kwargs)
    print(number_samples, "samples")

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        errors_u = []
        errors_phi = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors_u += [result[0]]
            errors_phi += [result[1]]
    freq_err_u = fftshift(fft(np.array(errors_u), axis=-1), axes=(-1, ))
    freq_err_phi = fftshift(fft(np.array(errors_phi), axis=-1), axes=(-1, ))
    return np.std(freq_err_u, axis=0), np.std(freq_err_phi, axis=0)

def simulation_firstlevels(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(firstlevels_errors, discretization, N, **kwargs)

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        errors_phi = []
        errors_baru = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors_phi += [result[0]]
            errors_baru += [result[1]]
    freq_err_u = fftshift(fft(np.array(errors_baru), axis=-1), axes=(-1, ))
    freq_err_phi = fftshift(fft(np.array(errors_phi), axis=-1), axes=(-1, ))
    return freq_err_phi, freq_err_u


def simulation_cv_rate_linear(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(interface_errors, discretization, N, **kwargs)
    print(number_samples, "samples")

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        errors_phi = []
        errors_u = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors_u += [result[0]]
            errors_phi += [result[1]]
    freq_err_u = fftshift(fft(np.array(errors_u), axis=-1), axes=(-1, ))
    freq_err_phi = fftshift(fft(np.array(errors_phi), axis=-1), axes=(-1, ))
    return linear_regression_1D(freq_err_u * discretization.LAMBDA_2 + freq_err_phi)

def simulation_cv_rate_matrixlinear(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(interface_errors, discretization, N, **kwargs)
    print(number_samples, "samples")

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        errors_phi = []
        errors_u = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors_u += [result[0]]
            errors_phi += [result[1]]
    freq_err_u = fftshift(fft(np.array(errors_u), axis=-1), axes=(-1, ))
    freq_err_phi = fftshift(fft(np.array(errors_phi), axis=-1), axes=(-1, ))
    return linear_regression_2D(freq_err_u, freq_err_phi)

def frequency_simulation(discretization, N, number_samples=100, **kwargs):
    """
        Simulate and returns directly errors in frequencial domain.
        number_samples simulations are done to have
        an average on all possible first guess.
        Every argument should be given in discretization.
        N is the number of time steps.
        kwargs can contain any argument of interface_errors, like NUMBER_IT
    """
    import concurrent.futures
    from numpy.fft import fft, fftshift
    to_map = functools.partial(interface_errors, discretization, N, **kwargs)
    print(number_samples, "samples")

    from progressbar import ProgressBar
    progressbar = ProgressBar(maxval=number_samples)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        errors_phi = []
        errors_u = []
        for result in progressbar(executor.map(to_map, range(number_samples))):
            errors_u += [result[0]]
            errors_phi += [result[1]]
    freq_err_u = fftshift(fft(np.array(errors_u), axis=-1), axes=(-1, ))
    freq_err_phi = fftshift(fft(np.array(errors_phi), axis=-1), axes=(-1, ))
    return np.std(freq_err_u * discretization.LAMBDA_2 + freq_err_phi, axis=0)


def interface_errors(discretization,
                     time_window_len,
                     seed=9380,
                     NUMBER_IT=2):
    """
        returns errors at interface from beginning (first guess) until the end.
    """
    def f1(t):
        return np.zeros(discretization.size_f(upper_domain=False))
    def f2(t):
        return np.zeros(discretization.size_f(upper_domain=True))
    def neumann(t):
        return 0.
    def dirichlet(t):
        return 0.

    precomputed_Y1 = discretization.precompute_Y(upper_domain=False)
    precomputed_Y2 = discretization.precompute_Y(upper_domain=True)

    u1_0 = np.zeros(discretization.size_prognostic(upper_domain=False))
    u2_0 = np.zeros(discretization.size_prognostic(upper_domain=True))
    # random false initialization:
    np.random.seed(seed)
    all_u1_interface = np.concatenate(([0], 2 * (np.random.rand(time_window_len) - 0.5)))
    all_phi1_interface = np.concatenate(([0], 2 * (np.random.rand(time_window_len) - 0.5)))
    #ret = [discretization.LAMBDA_2 * all_u1_interface + all_phi1_interface]
    ret_u = [all_u1_interface]
    ret_phi = [all_phi1_interface]
    # Beginning of schwarz iterations:
    from scipy.interpolate import interp1d
    for k in range(NUMBER_IT):
        all_u2_interface = [0]
        all_phi2_interface = [0]
        last_u2 = u2_0
        additional = []
        # Time iteration:
        interpolator_u1 = interp1d(x=np.array(range(time_window_len+1)), y=all_u1_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))
        interpolator_phi1 = interp1d(x=np.array(range(time_window_len+1)), y=all_phi1_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))

        for i in range(1, time_window_len+1):
            u_interface = lambda t : interpolator_u1(i-1+t)
            phi_interface = lambda t : interpolator_phi1(i-1+t)

            u2_ret, u_interface, phi_interface, *additional = \
                    discretization.integrate_one_step(
                f=f2,
                bd_cond=neumann,
                u_nm1=last_u2,
                u_interface=u_interface,
                phi_interface=phi_interface,
                additional=additional,
                upper_domain=True,
                Y=precomputed_Y2)
            last_u2 = u2_ret
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = [0]
        all_phi1_interface = [0]
        last_u1 = u1_0
        additional = []

        interpolator_u2 = interp1d(x=np.array(range(time_window_len+1)), y=all_u2_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))
        interpolator_phi2 = interp1d(x=np.array(range(time_window_len+1)), y=all_phi2_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))
        for i in range(1, time_window_len+1):
            u_interface = lambda t : interpolator_u2(i-1+t)
            phi_interface = lambda t : interpolator_phi2(i-1+t)

            u1_ret, u_interface, phi_interface, *additional = \
                    discretization.integrate_one_step(
                f=f1,
                bd_cond=dirichlet,
                u_nm1=last_u1,
                u_interface=u_interface,
                phi_interface=phi_interface,
                additional=additional,
                upper_domain=False,
                Y=precomputed_Y1)
            last_u1 = u1_ret
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

        #ret += [discretization.LAMBDA_2 * np.array(all_u1_interface) + np.array(all_phi1_interface)]
        ret_u += [np.array(all_u1_interface)]
        ret_phi += [np.array(all_phi1_interface)]

    return ret_u, ret_phi


def firstlevels_errors(discretization,
                     time_window_len,
                     seed=9380,
                     NUMBER_IT=1):
    """
        returns errors at interface from beginning (first guess) until the end.
    """
    def f1(t):
        return np.zeros(discretization.size_f(upper_domain=False))
    def f2(t):
        return np.zeros(discretization.size_f(upper_domain=True))
    def neumann(t):
        return 0.
    def dirichlet(t):
        return 0.

    precomputed_Y1 = discretization.precompute_Y(upper_domain=False)
    precomputed_Y2 = discretization.precompute_Y(upper_domain=True)
    firstmlevels = 3
    u1_0 = np.zeros(discretization.size_prognostic(upper_domain=False))
    u2_0 = np.zeros(discretization.size_prognostic(upper_domain=True))
    # random false initialization:
    np.random.seed(seed)
    all_u1_interface = np.concatenate(([0], 2 * (np.random.rand(time_window_len) - 0.5)))
    all_phi1_interface = np.concatenate(([0], 2 * (np.random.rand(time_window_len) - 0.5)))
    # Beginning of schwarz iterations:
    from scipy.interpolate import interp1d
    for k in range(NUMBER_IT):
        all_u2_interface = [0]
        all_phi2_interface = [0]
        last_u2 = u2_0
        additional = []
        # Time iteration:
        interpolator_u1 = interp1d(x=np.array(range(time_window_len+1)), y=all_u1_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))
        interpolator_phi1 = interp1d(x=np.array(range(time_window_len+1)), y=all_phi1_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))

        for i in range(1, time_window_len+1):
            u_interface = lambda t : interpolator_u1(i-1+t)
            phi_interface = lambda t : interpolator_phi1(i-1+t)

            u2_ret, u_interface, phi_interface, *additional = \
                    discretization.integrate_one_step(
                f=f2,
                bd_cond=neumann,
                u_nm1=last_u2,
                u_interface=u_interface,
                phi_interface=phi_interface,
                additional=additional,
                upper_domain=True,
                Y=precomputed_Y2)
            last_u2 = u2_ret
            all_u2_interface += [u_interface]
            all_phi2_interface += [phi_interface]

        all_u1_interface = [0]
        all_phi1_interface = [0]
        last_u1 = u1_0
        additional = []

        interpolator_u2 = interp1d(x=np.array(range(time_window_len+1)), y=all_u2_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))
        interpolator_phi2 = interp1d(x=np.array(range(time_window_len+1)), y=all_phi2_interface,
                kind='cubic', bounds_error=False, fill_value=(0., 0.))

        ret_phi = [np.zeros(firstmlevels)]
        ret_additional = [np.zeros(firstmlevels)]
        for i in range(1, time_window_len+1):
            u_interface = lambda t : interpolator_u2(i-1+t)
            phi_interface = lambda t : interpolator_phi2(i-1+t)

            u1_ret, u_interface, phi_interface, *additional = \
                    discretization.integrate_one_step(
                f=f1,
                bd_cond=dirichlet,
                u_nm1=last_u1,
                u_interface=u_interface,
                phi_interface=phi_interface,
                additional=additional,
                upper_domain=False,
                Y=precomputed_Y1)
            last_u1 = u1_ret
            ret_phi += [u1_ret[:firstmlevels]]
            ret_additional += [additional[0][:firstmlevels]]
            all_u1_interface += [u_interface]
            all_phi1_interface += [phi_interface]

    return ret_phi, ret_additional

if __name__ == "__main__":
    import main
    main.main()
