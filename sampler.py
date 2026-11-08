import numpy as np

def sampler_function(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    samplers = extra_options.pop('samplers')
    splits = extra_options.pop('splits')
    splits_l = np.array([0] + splits + [len(sigmas)])
    sigmas_l = [sigmas[start:end] for start, end in list(zip(splits_l, splits_l[1:]+1))]
    for sampler, sigmas in zip(samplers, sigmas_l):
        extras = dict(**extra_options, **sampler.extra_options)
        noise = sampler.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=callback, disable=disable, **extras)
    return noise