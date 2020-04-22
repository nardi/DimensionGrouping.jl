export euclidean, periodic_dist, circle_dist

using LinearAlgebra

Num = Number
Vec = Vector

# TODO: this probably exists in Julia base, or something similar :)
apply(t::Tuple{Function}) = t[1](tail(t)...)

# Euclidean distance.
euclidean(x::Vec, y::Vec) = norm(x .- y)
euclidean(x::Num, y::Num) = abs(x - y)

# Periodic and circle distance.
periodic_dist(p::Num) = (a::Num, b::Num) -> min(abs(a - b), p - abs(a - b));
periodic_dist(ps::Vec) = (as::Vec, bs::Vec) -> (zip(periodic_dist.(ps), as, bs) .|> apply);
periodic_dist(a::Num, b::Num, p::Num=1) = periodic_dist(p)(a, b);
periodic_dist(as::Vec, bs::Vec, ps::Vec=ones(length(as))) = periodic_dist(ps)(as, bs);
periodic_dist(as::Vec, bs::Vec, ps::Num=1) = periodic_dist(as, bs, fill(p, length(as)));
circle_dist(a::Num, b::Num) = periodic_dist(a, b, 2π);
circle_dist(as::Vec, bs::Vec) = periodic_dist(as, bs, 2π);