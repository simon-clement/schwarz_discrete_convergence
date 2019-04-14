"""
    This module is a very simple and effective module
    to perform memoisation on the heavy computations.
    The only problem is the limited possible types of arguments.
    The only possible parameters are litterals...
    If you want to use an other parameter,
    just implement its __hash__ and __eq__ function.
"""
import numpy as np

# everything is stored in the following folder.
# To know its size, "du -h <MEMOISATION_FOLDER_NAME>"
MEMOISATION_FOLDER_NAME = "cache_npy"
INDEX_NAME = MEMOISATION_FOLDER_NAME + "/index"
KEY_FOR_UNIQUE_ITEM = "key"


def clean():
    """
        Clean the cache. you can also just use:
        rm -rf <MEMOISATION_FOLDER_NAME>
        Don't forget to call this function if you change
        the computation (e.g. if you touch to cv_rate.py) !
    """
    import os
    import glob
    for filename in glob.iglob(MEMOISATION_FOLDER_NAME + "/*",
                               recursive=True):
        try:
            os.remove(filename)
        except OSError:
            for hashed_file in glob.iglob(filename + "/*",
                                       recursive=True):
                try:
                    os.remove(hashed_file)
                except OSError:
                    print("Error : recursion of more than one folder")
            try:
                os.removedirs(filename)
            except OSError:
                print("please clean manually folder " +
                      MEMOISATION_FOLDER_NAME)


def memoised(func_to_memoise, *args_mem, **kwargs_mem):
    """
        memoise function "fun" with arguments given.
        just replace your call:
        fun(my_arg1, my_arg2, my_kwarg1=my_val1)
        by: memoised(fun, my_arg1, my_arg2, my_kwarg1=my_val1)
    """
    fun = func_to_memoise
    import os
    filename_dict = INDEX_NAME + "_" + fun.__name__ + ".npy"
    try:
        dic = np.load(filename_dict)[()]
    except IOError:  # there is no index yet !
        dic = {}

    directory = MEMOISATION_FOLDER_NAME + "/" + fun.__name__
    os.makedirs(directory, exist_ok=True)
    key_dic = (*args_mem, tuple((key, val)
                                for key, val in sorted(kwargs_mem.items())))

    if key_dic in dic:
        # dic[key_dic] is the name of the file we're interested in.
        try:
            res = np.load(dic[key_dic])[()][KEY_FOR_UNIQUE_ITEM]
            if res is None:
                print("That is strange, we have a None result... " +
                      "Let's compute it again.")
            else:
                print("Found value for " + fun.__name__ + " in cache. " +
                      "If you changed anything in the function, " +
                      "run \"./main.py clean\"")
                return res
        except IOError:
            print("A file in the memoisation index doesn't exist.")

    filename_res = directory + "/" + repr(hash(key_dic)) + ".npy"
    # Store the filename in the dictionnary...
    while filename_res in dic.values():
        print("warning: collision in hashes or bug; very rare event")
        filename_res += str(int(np.random.rand()*10))

    dic[key_dic] = filename_res
    np.save(filename_dict, dic)
    # Finally, we can compute and store our result.
    res = fun(*args_mem, **kwargs_mem)
    # We use a dictionnary to store because we don't know type(res)
    to_store = {KEY_FOR_UNIQUE_ITEM: res}
    np.save(filename_res, to_store)
    return res


class FunMem():
    """
        Memoisable function.
        after the declaration of your function foo, use:
        foo = FunMem(foo)
        the function can then be passed as a parameter in
        memoised: for example,
        memoised(minimize_scalar, foo)

        Have a unsafe __repr__, which is the
        name of the function.
    """

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

    def __repr__(self):
        return self.fun.__name__

    def __eq__(self, other):
        return self.fun.__name__ == self.fun.__name__

    def __hash__(self):
        return hash(self.fun.__name__)
