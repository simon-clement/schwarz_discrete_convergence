from cffi import FFI
import numpy as np
import discretizations.finite_difference as finite_diff
import os
os.system('cd rust_tbc_parab_schwarz;cargo build --release; cd ..;')

def clean():
    os.system('cd rust_tbc_parab_schwarz;cargo clean; cd ..')

ffi = FFI()
# TODO remove or adapt
ffi.cdef("""
    double rate(unsigned long time_window_len, double Lambda_1, double Lambda_2, double a, double c, double dt, unsigned long M1, unsigned long M2,
        const double* h1, const double* h2, double* D1, double* D2, int is_finite_differences, unsigned long number_samples);
    const double* interface_err(unsigned long time_window_len, double Lambda_1, double Lambda_2, double a, double c, double dt, unsigned long M1, unsigned long M2,
        const double* h1, const double* h2, double* D1, double* D2, int is_finite_differences, unsigned long number_samples);
    const double* full_interface_err(unsigned long time_window_len, double Lambda_1, double Lambda_2, double a, double c, double dt, unsigned long M1, unsigned long M2,
        const double* h1, const double* h2, double* D1, double* D2, int is_finite_differences, unsigned long number_samples);
    """)


def _as_f64_array(array):
    return ffi.cast('const double *', array.ctypes.data)


def _as_f64(num):
    """ Cast np.float64 for Rust."""
    return ffi.cast("double", num)


def _as_u64(num):
    """ Cast `num` to Rust `usize`."""
    return ffi.cast("unsigned long", num)


def bool_as_i32(num):
    """ Cast `num` to Rust `usize`."""
    if num:
        x = 1
    else:
        x = 0
    return ffi.cast("int", x)


# Go get the Rust library.
# TODO must compute rate = mean(errors[2])/mean(errors[1])
# right now it is mean(errors[2]/errors[1])
lib = ffi.dlopen(
    "rust_tbc_parab_schwarz/target/release/librust_rate_constant.so")


def rate(discretization,
         N,
         Lambda_1=None,
         Lambda_2=None,
         a=None,
         c=None,
         dt=None,
         M1=None,
         M2=None,
         function_to_use=lambda x: max(np.abs(x)),
         number_seeds=10,
         function_D1=None,
         function_D2=None):
    if M1 is None:
        M1 = discretization.M1_DEFAULT
    if M2 is None:
        M2 = discretization.M2_DEFAULT
    if Lambda_1 is None:
        Lambda_1 = discretization.LAMBDA_1_DEFAULT
    if Lambda_2 is None:
        Lambda_2 = discretization.LAMBDA_2_DEFAULT
    h1, h2 = discretization.get_h(M1=M1, M2=M2)
    D1, D2 = discretization.get_D(h1=h1,
                                  h2=h2,
                                  function_D1=function_D1,
                                  function_D2=function_D2)
    a, c, dt = discretization.get_a_c_dt(a, c, dt)
    time_window_len = _as_u64(N)
    Lambda_1arg = _as_f64(Lambda_1)
    Lambda_2arg = _as_f64(Lambda_2)
    aarg = _as_f64(a)
    carg = _as_f64(c)
    dtarg = _as_f64(dt)
    M1arg = _as_u64(M1)
    M2arg = _as_u64(M2)
    is_finite_differences = bool_as_i32(
        discretization.name() == finite_diff.FiniteDifferences().name())
    number_samples = _as_u64(number_seeds)

    h1arg, h2arg, D1arg, D2arg = _as_f64_array(h1), _as_f64_array(h2), \
        _as_f64_array(D1), _as_f64_array(D2)

    ptr = lib.interface_err(time_window_len, Lambda_1arg, Lambda_2arg, aarg,
                            carg, dtarg, M1arg, M2arg, h1arg, h2arg, D1arg,
                            D2arg, is_finite_differences, number_samples)
    buf_ret = np.reshape(
        np.frombuffer(ffi.buffer(ptr, 8 * N * 3), dtype=np.float64), (3, N))

    # Warning: keep an eye on the lifetime of what you send inside _as_*
    # print(lib.rate(time_window_len, Lambda_1, Lambda_2,
    #           a, c, dt,
    #           M1, M2, h1arg, h2arg, D1arg, D2arg,
    #           is_finite_differences,
    #           number_samples))
    return function_to_use(buf_ret[2]) / function_to_use(buf_ret[1])


def errors(discretization,
           N,
           Lambda_1=None,
           Lambda_2=None,
           a=None,
           c=None,
           dt=None,
           M1=None,
           M2=None,
           number_seeds=10,
           function_D1=None,
           function_D2=None):
    if M1 is None:
        M1 = discretization.M1_DEFAULT
    if M2 is None:
        M2 = discretization.M2_DEFAULT
    if Lambda_1 is None:
        Lambda_1 = discretization.LAMBDA_1_DEFAULT
    if Lambda_2 is None:
        Lambda_2 = discretization.LAMBDA_2_DEFAULT
    h1, h2 = discretization.get_h(M1=M1, M2=M2)
    D1, D2 = discretization.get_D(h1=h1,
                                  h2=h2,
                                  function_D1=function_D1,
                                  function_D2=function_D2)
    a, c, dt = discretization.get_a_c_dt(a, c, dt)
    time_window_len = _as_u64(N)
    Lambda_1arg = _as_f64(Lambda_1)
    Lambda_2arg = _as_f64(Lambda_2)
    aarg = _as_f64(a)
    carg = _as_f64(c)
    dtarg = _as_f64(dt)
    M1arg = _as_u64(M1)
    M2arg = _as_u64(M2)
    is_finite_differences = bool_as_i32(
        discretization.name() == finite_diff.FiniteDifferences().name())
    number_samples = _as_u64(number_seeds)

    h1arg, h2arg, D1arg, D2arg = _as_f64_array(h1), _as_f64_array(h2), \
        _as_f64_array(D1), _as_f64_array(D2)

    ptr = lib.interface_err(time_window_len, Lambda_1arg, Lambda_2arg, aarg,
                            carg, dtarg, M1arg, M2arg, h1arg, h2arg, D1arg,
                            D2arg, is_finite_differences, number_samples)
    buf_ret = np.reshape(
        np.frombuffer(ffi.buffer(ptr, 8 * N * 3), dtype=np.float64), (3, N))
    return buf_ret


def errors_raw(discretization,
               N,
               Lambda_1=None,
               Lambda_2=None,
               a=None,
               c=None,
               dt=None,
               M1=None,
               M2=None,
               number_seeds=10,
               function_D1=None,
               function_D2=None):
    if M1 is None:
        M1 = discretization.M1_DEFAULT
    if M2 is None:
        M2 = discretization.M2_DEFAULT
    if Lambda_1 is None:
        Lambda_1 = discretization.LAMBDA_1_DEFAULT
    if Lambda_2 is None:
        Lambda_2 = discretization.LAMBDA_2_DEFAULT
    h1, h2 = discretization.get_h(M1=M1, M2=M2)
    D1, D2 = discretization.get_D(h1=h1,
                                  h2=h2,
                                  function_D1=function_D1,
                                  function_D2=function_D2)
    a, c, dt = discretization.get_a_c_dt(a, c, dt)
    time_window_len = _as_u64(N)
    Lambda_1arg = _as_f64(Lambda_1)
    Lambda_2arg = _as_f64(Lambda_2)
    aarg = _as_f64(a)
    carg = _as_f64(c)
    dtarg = _as_f64(dt)
    M1arg = _as_u64(M1)
    M2arg = _as_u64(M2)
    is_finite_differences = bool_as_i32(
        discretization.name() == finite_diff.FiniteDifferences().name())
    number_samples = _as_u64(number_seeds)

    h1arg, h2arg, D1arg, D2arg = _as_f64_array(h1), _as_f64_array(h2), \
        _as_f64_array(D1), _as_f64_array(D2)

    ptr = lib.full_interface_err(time_window_len, Lambda_1arg, Lambda_2arg,
                                 aarg, carg, dtarg, M1arg, M2arg, h1arg, h2arg,
                                 D1arg, D2arg, is_finite_differences,
                                 number_samples)
    buf_ret = np.reshape(
        np.frombuffer(ffi.buffer(ptr, 8 * N * 3 * number_seeds),
                      dtype=np.float64), (number_seeds, 3, N))
    return buf_ret
