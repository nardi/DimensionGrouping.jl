using ImageFiltering: imfilter!, default_resource, alg_defaults,
    filter_algorithm, borderinstance, allocate_output, reflect
using ImageFiltering.KernelFactors: gaussian

import Zygote

# Custom wrapper around imfilter, because it cannot be differentiated easily.
function conv!(out, img, kernel, border)
    alg = filter_algorithm(out, img, kernel)
    imfilter!(default_resource(alg_defaults(alg, out, kernel)),
        out, img, kernel, borderinstance(border))
end

# Utility function to produce an appropriate array for convolution results.
function conv_buffer(img, kernel, border)
    allocate_output(img, kernel, border)
end

# Allocating version of conv!.
function conv(img, kernel, border)
    out = conv_buffer(img, kernel, border)
    conv!(out, img, kernel, border)
end

# This is so we can obtain a 'buffer' array when calculating derivatives.
Zygote.grad_mut(arr::Array) = similar(arr)

# The derivative of conv! for Zygote.
Zygote.@adjoint! function conv!(out, img, kernel, border)
    conv!(out, img, kernel, border)
    out, (Δconv) -> begin
        # Some Zygote magic going on here. Basically, this way if we perform
        # multiple convolutions we can reuse a buffer array, I think...
        Δimg = Zygote.grad_mut(__context__, img)
        
        rk = broadcast(*, kernel...) |> reflect
        conv!(Δimg, Δconv, rk, border)
        #     (Δout, Δimg, Δkernel, Δborder)
        res = (nothing, Δimg, nothing, nothing)
        res
    end
end