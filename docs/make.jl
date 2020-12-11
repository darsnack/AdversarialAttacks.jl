using AdversarialAttacks
using Documenter

makedocs(;
    modules=[AdversarialAttacks],
    authors="Kyle Daruwalla",
    repo="https://github.com/darsnack/AdversarialAttacks.jl/blob/{commit}{path}#L{line}",
    sitename="AdversarialAttacks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://darsnack.github.io/AdversarialAttacks.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/darsnack/AdversarialAttacks.jl",
)
