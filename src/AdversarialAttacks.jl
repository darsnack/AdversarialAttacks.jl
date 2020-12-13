module AdversarialAttacks

using Zygote
using Adapt
using LinearAlgebra: norm

export fgsm, fgsm!, pgd, pgd!

include("utils.jl")
include("attacks.jl")

end