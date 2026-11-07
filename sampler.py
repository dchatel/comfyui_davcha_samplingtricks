def sampler_function(model_k, noise, sigmas, extra_args=None, callback=None, disable=None, **extra_options):
    samplers = extra_options.pop('samplers')
    splits = extra_options.pop('splits')
    if any(a >= b for a, b in zip(splits, splits[1:])):
        raise ValueError(f"[Davcha Scheduled Sampler] Splits must be strictly increasing, got {splits}.")
    # Splits past the last step leave the remaining samplers with nothing to do
    total_steps = len(sigmas) - 1
    bounds = [0] + [min(s, total_steps) for s in splits] + [total_steps]
    for sampler, start, end in zip(samplers, bounds, bounds[1:]):
        if start >= end:
            continue
        extras = dict(**extra_options, **sampler.extra_options)
        noise = sampler.sampler_function(model_k, noise, sigmas[start:end + 1], extra_args=extra_args, callback=callback, disable=disable, **extras)
    return noise
