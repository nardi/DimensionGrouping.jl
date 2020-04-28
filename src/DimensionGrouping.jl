module DimensionGrouping

export dg_init_random, dg_energy, dg_train!
DG = DimensionGrouping; export DG # Shorthand for package.

include.(
    ["distances", "arrayutils", "convolution",
     "testutils"]
.|> (f -> f * ".jl"))

using Statistics, Distributions
using LinearAlgebra, FillArrays, BlockArrays
import ImageFiltering
import Zygote; using Zygote: gradient, Buffer, dropgrad
using Flux.Optimise: update!, ADAM

import ProgressMeter; using ProgressMeter: Progress

abstract type Parametrization end
abstract type Nonparametric <: Parametrization end
const NonParametric = Nonparametric # To limit typing errors :P

struct Grouping{P <: Parametrization, R <: Real}
    k::Int; # The number of representatives.
    T::Int; # The number of time steps in the trajectory data.
    reps::AbstractArray{R}; # The data or parameters for the representatives.
    dist::Function; # A metric to use to calculate the distance between entities and representatives.
    dist_pow::Int; # The power used for the distance in the cost function.
	reg::Dict{Symbol, Real}; # A dictionary containing the regularizations applied.
end

const DEFAULT_REG = Dict{Symbol, Real}();

default_init(R::Type) = Uniform(zero(R), one(R))
function default_init(X::AbstractArray{R}, param::Type) where {R<:Real}
    if param == Nonparametric
        Uniform(extrema(X)...)
    else
        default_init(R)
    end
end

function dg_init_random(k::Int, T::Int; entity_dim::Int=1,
                        eltype::Type=Float32, dist::Function=euclidean,
                        init::Sampleable=default_init(eltype), dist_pow::Int=2)
    Grouping{Nonparametric, eltype}(
		k, T, rand(init, k * entity_dim, T), dist, dist_pow, DEFAULT_REG
	)
end

function dg_init_random(k::Int, X::AbstractArray{R}, entity_dim::Int=1;
                        dist::Function=euclidean,
                        init::Sampleable=default_init(X, param),
                        dist_pow::Int=2) where {R <: Real}
    N, T = size(X)[1] ÷ entity_dim, size(X)[2]
    kw = (entity_dim=entity_dim, eltype=R, dist=dist,
          init=init, dist_pow=dist_pow)
    
    dg_init_random(k, T; kw...)
end

"""
    Turns the parameter data for the representatives into their
    trajectory data. In the nonparametric case, this is a no-op.
"""
construct_representatives(dg::Grouping{Nonparametric}, X) = dg.reps

"""
    Calculates the sum of regularization terms in the given configuration.
    Returns zero in the nonparametric case.
"""
regularization(dg::Grouping{Nonparametric, R}) where {R<:Real} = zero(R)


"""
    Calculates the "energy" of a given grouping, in the context
    of the data set X.
"""
function dg_energy(dg::Grouping{P, R}, X, sigma=zero(R)) where {P, R}
    # Set up values for use. Drop gradients for irrelevant values to
    # save on computation.
    k, T = (dg.k, dg.T) .|> dropgrad
    dist, p = (dg.dist, dg.dist_pow) |> dropgrad
    N, _ = size(X)
    kernel = gaussian((0, 0, sigma))
    
    S = construct_representatives(dg, X)
    
    # Create a 3d buffer array (one that Zygote knows how to handle),
    # of size k*N*T.
    D = Buffer(S, k, N, T)

    # Store distances for each entity/representative pair.
    for (t, (s, x)) in enumerate(zip(eachcol(S), eachcol(X)))
        D[:, :, t] = pwdist(s, x, dist).^p
    end
    
    # Convolve over the time axis (the last axis).
    Dc = conv(D |> copy, kernel, ImageFiltering.Fill(zero(R)))
    # To calculate the energy, pick the minimum distance for each entity.
    # Then sum all distances and average over time.
    E = mean(sum(minimum(Dc, dims=1), dims=2))
	
    # Add any regularization penalties.
	E + regularization(dg)
end

"""
    Adjusts a grouping to a data set X. Can use any optimizer from
    Flux. Applies as many optimization steps as given by 'rounds',
    and for any step i uses the value of sigma[i] to determine the
    width of the convolution kernel. Stops early if a step changes
    less (in the negative direction) than either 'reltol' (relatively)
    or 'abstol' (absolutely). Returns the grouping and some statistics,
    which may be ignored.
"""
function dg_train!(dg::Grouping, X, rounds, sigma::AA; opt = ADAM(0.1),
                   reltol=1e-3, abstol=1e-5, verbose=true, prog=nothing)
    loss = (dg, sigma) -> dg_energy(dg, X, sigma |> dropgrad)
    
    # If we are giving verbose output, make sure we have a progress bar.
    if (verbose && prog == nothing) prog = Progress(rounds) end
    
    prev_loss = loss(dg, sigma[end])
    
    rounds_needed = 1
    let new_loss, abs_loss, rel_loss
        for i in 1:rounds
            grad = gradient(loss, dg, sigma[i])
            update!(opt, dg.reps, grad[1].reps)

            # Calculate loss relative to the final σ value, for consistency.
            new_loss = loss(dg, sigma[end])
            abs_loss = new_loss - prev_loss
            rel_loss = abs_loss / prev_loss

            if verbose 
                ProgressMeter.next!(prog;
                    showvalues = [("loss", new_loss),
                        ("absolute change", abs_loss),
                        ("relative change", rel_loss)
                ])
            end

            if -abstol < abs_loss < 0 || -reltol < rel_loss < 0
                # Print if we stop early.
                if verbose println("Stopping after $i/$rounds rounds.") end
                rounds_needed = i // rounds
                break
            end
            
            prev_loss = new_loss
        end
    
        dg, (rounds_needed=rounds_needed, loss=new_loss, abs=abs_loss, rel=rel_loss)
    end
end

"""
    Creates a sequence of convolution widths starting
    at 'start' and decaying as 1/t, such that at time
    T/2 the width equals start/2.
"""
function dg_σ_decay(start::AbstractFloat, T::Int)
    @. start / (1 + (0:(T-1)) / ((T-1)/2))
end

# Automatically constructs a sequence of convolution widths.
function dg_train!(dg::Grouping{P, R}, X::AbstractArray, rounds;
                   sigma=dg.T/10, kw...) where {P, R}
    dg_train!(dg, X, rounds, dg_σ_decay(convert(R, sigma), rounds); kw...)
end

"""
    Quantizes each trajectory in X to its group representative.
"""
function dg_quantize(dg::Grouping, X)
    k, T = dg.k, dg.T
    S = construct_representatives(dg, X)
    dist = dg.dist
    N, _ = size(X)
    
    # Project each entity onto its closest representative.
    hcat([begin
            d = pwdist(s, x, dist)
            winners = s[eachcol(d) .|> argmin]
    end for (s, x) in zip(eachcol(S), eachcol(X))]...)
end
    
"""
    Calculates the error in quantizing each trajectory in X
    to its group representative.
"""
function dg_quantization_err(dg::Grouping, X)
    norm(X .- dg_quantize(dg, X)) / size(X)[2]
end

end # module
