using LinearAlgebra, FillArrays, BlockArrays
import Base, Zygote

# Tensor product of vectors, because the normal "product"
# is not differentiable.
prod(a, b) = hcat([[(x, y) for x in a] for y in b]...);

AA = AbstractArray
"""
    Construct a matrix containing the distances between each pair of points.
"""
pwdist(s::AA, x::AA, dist::Function=euclidean) = prod(s, x) .|> (p -> dist(p[1], p[2]));
pwdist(t::Tuple{N, N} where N<:AA, dist=euclidean) = pwdist(t..., dist);
pwdist(dist::Function=euclidean) = (t::Tuple{N, N} where N<:AA) -> pwdist(t..., dist);

"""
    Normalizes a data set so that the min/max values in each row are 0 and 1.
    Returns the original range as well, so it can be inverted later.
"""
function data_normalize(data, drange=data_range(data))
    data_normal = @. (data - drange[1]) / (drange[2] - drange[1])
    return data_normal, drange
end
data_range(data) = (minimum(data), maximum(data))
"""
    Scales a data set so that the values in each row are in the ranges
    indicated by data_range.
"""
function data_denormalize(data, data_range)
    @. data * (data_range[2] - data_range[1]) + data_range[1]
end

# Curried versions of dropdims.
Base.dropdims(dim::Int) = x -> dropdims(x, dims=dim)
Base.dropdims(dims::Tuple{Int}) = x -> dropdims(x, dims=dims)

"""
    hglue: hcat all members of a collection.
"""
hglue(x) = hcat(x...);

"""
    Normalize M so that the p-norm of each column equals 1.
"""
col_normalize!(M, p=2) = begin eachcol(M) .|> (x -> normalize!(x, p)); M end

"""
    Create a normalized version of M where the p-norm of each column equals 1.
"""
col_normalize(M, p=2) = col_normalize!(copy(M), p);

"""
    Normalize M so that the p-norm of each row equals 1.
"""
row_normalize!(M, p=2) = begin eachrow(M) .|> (x -> normalize!(x, p)); M end

"""
    Create a normalized version of M where the p-norm of each row equals 1.
"""
row_normalize(M, p=2) = row_normalize!(copy(M), p);

"""
    Converts an k x n matrix into a mk x mn matrix where each entry is
    'stretched' to a m x m scaling matrix.
"""
stretch_matrix(C::AbstractArray, m) = C .|> (x -> Diagonal(Fill(x, m))) |> mortar

# Gets a properly sized array for gradient output.
Zygote.grad_mut(arr::AbstractArray) = similar(arr)

# Manual definition of the derivative for stretch_matrix
# because it uses BlockArrays which are not supported.
Zygote.@adjoint! function stretch_matrix(C::AbstractArray, m)
    out = stretch_matrix(C, m)
    out, (Δout) -> begin
        s = size(C)
        ΔC = Zygote.grad_mut(__context__, C)
        Δout_bl = BlockArray(Δout, Fill(m, s[1]), Fill(m, s[2]))

        for i in 1:s[1], j in 1:s[2]
            ΔC[i, j] = (@view Δout_bl[Block(i, j)]) |> diag |> sum
        end
        
        #     (ΔC, Δm)
        res = (ΔC, nothing)
        res
    end
end