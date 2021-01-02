using LinearAlgebra: norm

function _computepgdstep!(grads; α, αnorm) 
    if isinf(αnorm)
        grads .= sign.(grads) .* α
    else
        map!(x -> x * α / norm(reshape(x, :), αnorm), eachslice(grads; dims = ndims(grads)))
    end
        
    return grads
end
    
"""
    pgd!(x, y, model; loss, nsteps, target = nothing,
         ϵ = 0.5, α = ϵ / nsteps, ϵnorm = 2, αnorm = ϵnorm,
         clamprange = (0, 1), initrange = clamprange, project = true, mcsamples = 1)

Perturb `x` *in-place* using a `nsteps` PGD attack.
When `target` is set, `loss(model(x), target)` is minimized.
If `target` is `nothing`, then `loss(model(x), y)` is maximized (i.e. an untargeted attack).

# Arguments:
- `x`: the input sample to perturb *in-place*
- `y`: the correct output label
- `model`: the model to attack
- `loss`: the loss function
- `nsteps`: the number of PGD iterations
- `ϵ`: the size of the pertubation ball
- `α`: the step size for each perturbation step
- `ϵnorm`: the type of L-norm for the ϵ-ball
- `αnorm`: the type of L-norm used for normalizing the steps by sample per batch
- `clamprange`: the range to clamp the final perturbed sample
- `initrange`: the range to used when randomly initializing the perturbation
- `project`: set to `true` to project the perturbation onto the ϵ-ball at each step
- `mcsamples`: the number of Monte-Carlo samples used to estimate the gradient at each step
               (this is helpful `pgd!` is paired with a stochastic/obfuscated defense mechanism)
"""
function pgd!(x, y, model; loss, nsteps, target = nothing,
                           ϵ = 0.5, α = ϵ / nsteps, ϵnorm = 2, αnorm = ϵnorm,
                           clamprange = (0, 1), initrange = clamprange,
                           project = true, mcsamples = 1)
    # choose targeted or untargeted label
    ytarget = isnothing(target) ? y : target

    # randomly initialize perturbation
    δ = rand_init(x; range = initrange)
    
    # perform attack iterations
    for i in 1:nsteps
        # take gradient
        grads = zero(x)
        for j in 1:mcsamples # estimate gradient with samples
            grads .+= gradient(x -> loss(model(x), ytarget), x .+ δ)[1]
        end
        grads ./= mcsamples
        
        # compute gradient step
        _computepgdstep!(grads; α = α, αnorm = αnorm)
            
        # gradient descent towards target or gradient ascent away from y
        δ .+= isnothing(target) ? grads : -grads
        
        if project
            # project back onto l-ball
            proj_lball!(x, δ; ϵ = ϵ, ϵnorm = ϵnorm)
        else
            # just add gradient step
            x .+= δ
        end
    end
           
    # clamp output
    x .= clamp.(x, clamprange[1], clamprange[2])

    return x
end
        
"""
    pgd(x, y, model; loss, nsteps, target = nothing,
        ϵ = 0.5, α = ϵ / 5, ϵnorm = 2, αnorm = 2,
        clamprange = (0, 1), project = true, mcsamples = 1)

A non-mutating version of [`pgd!`](@ref).
"""
pgd(x, y, model; kwargs...) = pgd!(copy(x), y, model; kwargs...)

"""
    fgsm!(x, y, model; loss, target = nothing, ϵ = 0.5,
                       clamprange = (0, 1), mcsamples = 1)

Perturb `x` *in-place* using a FGSM attack.
When `target` is set, `loss(model(x), target)` is minimized.
If `target` is `nothing`, then `loss(model(x), y)` is maximized (i.e. an untargeted attack).

*Note: FGSM is implemented as 1-step PGD with L∞-norm.*

# Arguments:
- `x`: the input sample to perturb *in-place*
- `y`: the correct output label
- `model`: the model to attack
- `loss`: the loss function
- `ϵ`: the size of the pertubation ball
- `clamprange`: the range to clamp the final perturbed sample
- `initrange`: the range to used when randomly initializing the perturbation
- `mcsamples`: the number of Monte-Carlo samples used to estimate the gradient at each step
                (this is helpful `pgd!` is paired with a stochastic/obfuscated defense mechanism)
"""
fgsm!(x, y, model; loss, target = nothing, ϵ = 0.5, clamprange = (0, 1), initrange = clamprange, mcsamples = 1) =
    pgd!(x, y, model; loss = loss, target = target,
                      ϵ = ϵ, clamprange = clamprange, initrange = initrange, 
                      mcsamples = mcsamples,
                      nsteps = 1, ϵnorm = Inf)

"""
    fgsm(x, y, model; loss, target = nothing, ϵ = 0.5,
                      clamprange = (0, 1), mcsamples = 1)

A non-mutating version of [`fgsm!`](@ref).
"""
fgsm(x, y, model; kwargs...) = fgsm!(copy(x), y, model; kwargs...)