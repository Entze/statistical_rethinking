using Distributions: pdf, Binomial
using Plots: plot

"""
Suppose the globe tossing data (Chapter 2) had turned out to be 4 water
and 11 land. Construct the posterior distribution, using grid approximation.
Use the same flat prior as in the book.
"""
function problem1(grid_length=20)

    # define grid
    p_grid = [n for n in range(0,1,grid_length)]
    @debug "p_grid" p_grid

    # define prior
    prior = fill(1.0, grid_length)
    @debug "prior" prior

    # Distribution
    binomials =  Binomial.(15, p_grid)
    @debug "binomials" binomials

    # compute likelihood at each value in grid
    likelihood = pdf.(binomials, 4)
    @debug "likelihood" likelihood

    # compute product of likelihood and prior
    unstd_posterior = likelihood .* prior
    @debug "unstd_posterior" unstd_posterior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior ./ sum(unstd_posterior)
    @debug "posterior" posterior

    return posterior
end

function problem1_plot(grid_length=20)
    xs = Vector(range(0,1,grid_length))
    data = problem1(grid_length)
    title = "HW1 Problem 1 (Gridsize=" * string(grid_length) * ")"
    xlabel="probability of water"
    ylabel="posterior probability"
    labels="y"
    plot(xs, data, title=title, labels=labels, xlabel=xlabel, ylabel=ylabel)
end

function problem2(grid_length=20)
     # define grid
    p_grid = [n for n in range(0,1,grid_length)]
    @debug "p_grid" p_grid

    # define prior
    prior = [g < 0.5 ? 0. : g for g in p_grid]
    @debug "prior" prior

    # Distribution
    binomials =  Binomial.(6, p_grid)
    @debug "binomials" binomials

    # compute likelihood at each value in grid
    likelihood = pdf.(binomials, 4)
    @debug "likelihood" likelihood

    # compute product of likelihood and prior
    unstd_posterior = likelihood .* prior
    @debug "unstd_posterior" unstd_posterior

    # standardize the posterior, so it sums to 1
    posterior = unstd_posterior ./ sum(unstd_posterior)
    @debug "posterior" posterior

    return posterior
end

function problem2_plot(grid_length=20)
    xs = Vector(range(0,1,grid_length))
    data = problem2(grid_length)
    title = "HW1 Problem 2 (Gridsize=" * string(grid_length) * ")"
    xlabel="probability of water"
    ylabel="posterior probability"
    labels="y"
    plot(xs, data, title=title, labels=labels, xlabel=xlabel, ylabel=ylabel)
end


function problem3()
end

function problem4()
end
