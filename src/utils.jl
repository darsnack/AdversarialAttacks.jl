"""
    rand_init(x::Array; range = (0, 1))    

Generate a random initialization of the same size and type as `x`.
If `x` is an integer array, then `range` specifies a set of integers.
Otherwise, `range` specifies the set of real numbers.
"""
rand_init(x::AbstractArray{<:Integer}; range = (0, 255)) = rand(range[1]:range[2], size(x))
rand_init(x::AbstractArray; range = (0, 1)) = (range[2] - range[1]) .* rand(size(x)) .+ range[1]

"""
    proj_lball!(xadv, δ; ϵ, ϵnorm)

Project the perturbed sample (`xadv`) onto an `ϵnorm`-ball of size `ϵ` around `x`.

# Arguments
- `xadv`: the adversarial (perturbed) sample/batch
- `δ`: the pertubations
- `ϵ`: the size of the L-norm ball
- `ϵnorm`: the type of L-norm used
"""
function proj_lball!(xadv, δ; ϵ, ϵnorm)
    if isinf(ϵnorm)
        xadv .= max.(min.(xadv, x .+ ϵ), x .- ϵ)
    else
        map!((xi, δi) -> xi .+ δi * (ϵ / max.(norm(reshape(δi, :), ϵnorm), ϵ)),
             xadv, xadv, eachslice(δ; dims = 4))
    end
            
    return xadv
end