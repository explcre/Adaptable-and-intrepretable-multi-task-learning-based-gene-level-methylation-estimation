import pytorch_lightning as pl


## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


from collections import defaultdict
def extract_weight_method_parameters_from_args(args):
    weight_methods_parameters = defaultdict(dict)
    weight_methods_parameters.update(
        dict(
            nashmtl=dict(
                update_weights_every=args.update_weights_every,
                optim_niter=args.nashmtl_optim_niter,
            ),
            stl=dict(main_task=args.main_task),
            cagrad=dict(c=args.c),
            dwa=dict(temp=args.dwa_temp),
        )
    )
    return weight_methods_parameters
