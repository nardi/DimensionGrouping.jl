## Synthetic trajectory data generation.

"""
    Generates 'amount' 1-d trajectories which start at 'start' and
    move with a constant velocity 'slope' for 'length' time units.
"""
linear_walk(length, amount=1; start=0, slope=0.01) =
    repeat((slope .* (0:length-1) .+ start)' |> collect, amount, 1)
"""
    Generates 'amount' 1-d trajectories which start at 'start' and
    randomly take a step according to a Laplace ditribution with
    spread parameter 'stepsize', for 'length' steps in total.
""" 
random_walk(length, amount=1; start=0, stepsize=0.01) =
    cumsum(rand(Laplace(0, stepsize), amount, length), dims=2) .+ start
"""
    Creates a group of trajectories, where they all 'walk linearly'
    in the same way (according to 'slope') and each induvidual
    performs a random walk around the linear trajectory (deviating
    according to 'stepsize').
"""
group_walk(length, size; start=0, stepsize=0.01, slope=0.01) =
    linear_walk(length, size, start=start, slope=slope) +
    random_walk(length, size, stepsize=stepsize)
"""
    Smoothens a set of trajectories to make them more realistic
    (e.g. remove the discontinuous changes in velocity). Filters
    using a Gaussian kernel of width 'scale' (wider is smoother),
    and allows for subsampling (selecting each n-th sample only)
    using 'subsample'.
"""
smooth_walk(data; scale::Real=4, subsample::Int=4) =
    imfilter(data, Kernel.gaussian((0, scale)))[:, 1:subsample:T]