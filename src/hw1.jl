using Distributions: pdf, Binomial, rand
import Plots
using Plots: plot, plot!, scatter!, histogram
using StatisticalRethinking: hpdi, PI
using Gen: @gen, binom, (~)

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

"""
Now suppose the data are 4 water and 2 land. Compute the posterior
again, but this time use a prior that is zero below p = 0.5 and a constant
above p = 0.5. This corresponds to prior information that a majority of the
Earth’s surface is water.
"""
function problem1_plot(grid_length=20)
    xs = range(0,1,length=grid_length)
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
    prior = [g < 0.5 ? 0. : 1. for g in p_grid]
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
    xs = range(0,1,length=grid_length)
    data = problem2(grid_length)
    title = "HW1 Problem 2 (Gridsize=" * string(grid_length) * ")"
    xlabel="probability of water"
    ylabel="posterior probability"
    labels="y"
    plot(xs, data, title=title, labels=labels, xlabel=xlabel, ylabel=ylabel)
end

"""
For the posterior distribution from 2, compute 89% percentile and HPDI
intervals. Compare the widths of these intervals. Which is wider? Why? If
you had only the information in the interval, what might you misunderstand
about the shape of the posterior distribution
"""
function problem3(grid_length=20)
    posterior = problem2(grid_length)
    return (posterior, PI(posterior;perc_prob=0.89), hpdi(posterior;alpha=0.11))
end

function problem3_plot(grid_length=20)
    posterior, percentile, highprobabilitydensityinterval = problem3(grid_length)
    xs = Vector(range(0,1,grid_length))
    plot(xs, posterior, layout=2)
    scatter!(percentile, posterior, fillrange=0., fillalpha=0.3, fillcolor=:blue)
end

"""
Suppose there is bias in sampling so that Land
is more likely than Water to be recorded. Specifically, assume that 1-in-5
(20%) of Water samples are accidentally recorded instead as ”Land”. First,
write a generative simulation of this sampling process. Assuming the true
proportion of Water is 0.70, what proportion does your simulation tend to
produce instead? Second, using a simulated sample of 20 tosses, compute
the unbiased posterior distribution of the true proportion of water.
"""
function problem4(samples=100, tosses=20, proportion_water=0.7, bias=0.8)
    biased_sample = zeros(tosses+1)
    biased_true = pdf.(Binomial(tosses, proportion_water * bias), 0:tosses)
    unbiased = pdf.(Binomial(tosses, proportion_water), 0:tosses)
    for _ in 1:samples
        sample = biased_water(tosses, proportion_water, bias)
        biased_sample[sample] += 1
    end

    biased_sample = biased_sample ./ samples

    return (biased_sample, biased_true, unbiased)
end

@gen function biased_water(tosses=20, proportion_water=0.7, bias=0.8)

    true_water ~ binom(tosses, proportion_water)
    recorded_water ~ binom(true_water, bias)

    return recorded_water
end

function problem4_plot(samples=100, tosses=20, proportion_water=0.7, bias=0.8)
    xs = range(0, tosses, step=1)
    @debug xs

    biased_sample, biased_true, unbiased = problem4(samples, tosses, proportion_water, bias)

    labels = "biased sample"
    xlabel="probability of water"
    ylabel="posterior probability"
    seriestype=:bar

    biased_sample_plot = plot(xs, biased_sample, seriestype=seriestype, color=:blue, title="Biased (" * string(samples) * " samples)", labels=labels, xlabel=xlabel, ylabel=ylabel)
    @debug biased_sample_plot

    biased_true_plot = plot(xs, biased_true, seriestype=seriestype, color=:green, title="Biased", labels=labels, xlabel=xlabel, ylabel=ylabel)
    @debug biased_true_plot

    unbiased_plot = plot(xs, unbiased, seriestype=seriestype, color=:red, title="Unbiased", labels=labels, xlabel=xlabel, ylabel=ylabel)
    @debug unbiased_plot

    overlay_plot = plot(xs, [biased_sample biased_true unbiased], seriestype=seriestype, seriesalpha=[0.33 0.33 0.33], color=[:blue :green :red], labels=["biased (sample)" "biased" "unbiased"], xlabel=xlabel, ylabel=ylabel)
    @debug overlay_plot

    plot(biased_sample_plot, biased_true_plot, unbiased_plot, overlay_plot, layout=4, legend=false)
end
