using StatisticalRethinking: sr_datadir, PI
using StatisticalRethinkingCommon
using Statistics: mean, median
using StatsBase: sample
using Plots: plot, display, scatter!, plot!
using DataFrames: DataFrame
using Gen: @gen, (~), normal, gamma, get_args, get_retval
import Gen
import CSV

@info "Reading Howell dataset"
@time howell1 = CSV.read(sr_datadir("Howell1.csv"), DataFrame)

howell1_adults = howell1[howell1.age .>= 18, :]
howell1_children = howell1[howell1.age .< 13, :]
howell1_boys = howell1_children[howell1_children.male .== 1, :]
howell1_girls = howell1_children[howell1_children.male .== 0, :]

function plot_howell1()

    height_weight_plot = plot(howell1.height, howell1.weight, seriestype=:scatter, color=:blue, title="Height/Weight", xlabel="height (cm)", ylabel="weight (kg)", labels="data")
    return height_weight_plot

end

function plot_howell1_adults()

    height_weight_plot = plot(howell1_adults.height, howell1_adults.weight, seriestype=:scatter, color=:blue, title="Height/Weight Adults", xlabel="height (cm)", ylabel="weight (kg)", labels="data")
    return height_weight_plot

end

function plot_howell1_children()

    height_weight_plot = plot(howell1_children.height, howell1_children.weight, seriestype=:scatter, color=:blue, title="Height/Weight Children", xlabel="height (cm)", ylabel="weight (kg)", labels="data")
    age_weight_plot = plot(howell1_children.age, howell1_children.weight, seriestype=:scatter, color=:blue, title="Age/Weight Children", xlabel="age (years)", ylabel="weight (kg)", labels="data")
    return plot(height_weight_plot, age_weight_plot, layout=2)

end

function plot_howell1_age_weight_boys_girls()
    age_weight_plot = plot([howell1_boys.age, howell1_girls.age],
                           [howell1_boys.weight, howell1_girls.weight],
                           seriestype=:scatter,
                           color=[:blue :red],
                           title="Age/Weight Children",
                           xlabel="age (years)",
                           ylabel="weight (kg)",
                           labels=["Boys" "Girls"])
    return age_weight_plot
end


function plot_howell1_height_weight_boys_girls()
    height_weight_plot = plot([howell1_boys.height, howell1_girls.height],
                              [howell1_boys.weight, howell1_girls.weight],
                              seriestype=:scatter,
                              color=[:blue :red],
                              title="Height/Weight Children",
                              xlabel="height (cm)",
                              ylabel="weight (kg)",
                              labels=["Boys" "Girls"])
    return height_weight_plot
end

function plot_howell1_boys_girls()
    return plot(plot_howell1_height_weight_boys_girls(), plot_howell1_boys_girls(), layout=2)

end




@gen function pos_line_model(xs,
                             alpha_mu=0.,
                             alpha_sigma=1.,
                             beta_shape=1.,
                             beta_scale=1.,
                             sigma_shape=1.,
                             sigma_scale=1.)

    alpha ~ normal(alpha_mu, alpha_sigma)
    beta ~ gamma(beta_shape, beta_scale)
    sigma ~ gamma(sigma_shape, sigma_scale)

    function f(x)
        return alpha .+ beta .* x
    end

    for (i, x) in enumerate(xs)
        ({(:y, i)} ~ normal(f(x), sigma))
    end

    return f

end

@gen function problem3_model(xs, male)
    alpha0  ~ normal(5., 1.0)
    beta0   ~  gamma(1., 1.5)
    sigma0  ~  gamma(1., 1.0)
    alpha1  ~ normal(5., 1.0)
    beta1   ~  gamma(1., 1.5)
    sigma1  ~  gamma(1., 1.0)

    function f(x, m)
        return (alpha0 * (1 - m) + alpha1 * m) + (beta0  * (1 - m) + beta1 * m) * x
    end


    for (i, x) in enumerate(xs)
        m = male[i]
        {(:y,  i)} ~ normal(f(x,m), sigma0 * (1-m) + sigma1 * m)
    end

    return f
end


function render_linear_trace(trace; show_data=true)

    # Pull out xs from the trace
    xs, = get_args(trace)

    xmin = minimum(xs)
    xmax = maximum(xs)

    # Pull out the return value, useful for plotting
    y = get_retval(trace)


    # Draw the line
    test_xs = collect(range(xmin - 5, stop=xmax + 5, length=1000))
    test_ys = map(y, test_xs)

    ymin = minimum(test_ys)
    ymax = maximum(test_ys)

    fig = plot(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                xlim=(xmin-0.5, xmax+0.5), ylim=(ymin-0.5, ymax+0.5))

    if show_data
        ys = [trace[(:y, i)] for i=1:length(xs)]

        # Plot the data set
        scatter!(xs, ys, c="black", label=nothing)
    end

    return fig
end

function do_importance_inference(model, xs, ys, steps, modelargs...)

    # Create a choice map that maps model addresses (:y, i)
    # to observed values ys[i]. We leave :slope and :intercept
    # unconstrained, because we want them to be inferred.
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Call importance_resampling to obtain a likely trace consistent
    # with our observations.
    (trace, _) = Gen.importance_resampling(model, (xs, modelargs...), observations, steps)
    return trace
end

function overlay(renderer, traces; same_data=true, args...)
    fig = renderer(traces[1], show_data=true, args...)

    xs, = get_args(traces[1])
    xmin = minimum(xs)
    xmax = maximum(xs)

    # Draw the line



    for i=2:length(traces)
        y = get_retval(traces[i])
        test_xs = collect(range(xmin - 5, stop=xmax + 5, length=1000))
        test_ys = map(y, test_xs)
        ymin = minimum(test_ys)
        ymax = maximum(test_ys)
        fig = plot!(test_xs, map(y, test_xs), color="black", alpha=0.5, label=nothing,
                    xlim=(xmin - 0.5 , xmax + 0.5), ylim=(ymin - 0.5, ymax + 0.5))
    end
    return fig
end

function predict_new_data(model, trace, new_xs::Vector{Float64}, param_addrs)

    # Copy parameter values from the inferred trace (`trace`)
    # into a fresh set of constraints.
    constraints = Gen.choicemap()
    for addr in param_addrs
        constraints[addr] = trace[addr]
    end

    # Run the model with new x coordinates, and with parameters
    # fixed to be the inferred values.
    (new_trace, _) = Gen.generate(model, (new_xs,), constraints)

    # Pull out the y-values and return them.
    ys = [new_trace[(:y, i)] for i=1:length(new_xs)]
    return ys
end

function infer_and_predict(model, xs, ys, new_xs, param_addrs, num_traces, steps, modelargs...)
    pred_ys = []
    for _ in 1:num_traces
        trace = do_importance_inference(model, xs, ys, steps, modelargs...)
        push!(pred_ys, predict_new_data(model, trace, new_xs, param_addrs))
    end
    return pred_ys
end


"""
Construct a linear regression of weight as predicted by height, using the
adults (age 18 or greater) from the Howell1 dataset. The heights listed below
were recorded in the !Kung census, but weights were not recorded for these
individuals. Provide predicted weights and 89% compatibility intervals for
each of these individuals. That is, fill in the table below, using model-based
predictions.

"""
function problem1(samples=100)

    alpha_mu = 50.
    alpha_sigma= 10.

    beta_shape=1.
    beta_scale=5.

    sigma_shape=1.
    sigma_scale=5.

    xbar = mean(howell1_adults.height)
    xs = howell1_adults.height .- xbar
    ys = howell1_adults.weight

    @info "Inferring:"
    @time all_predictions = infer_and_predict(pos_line_model, xs, ys, [140, 160, 175] .- xbar, (:alpha, :beta, :sigma), samples, 1000, alpha_mu, alpha_sigma, beta_shape, beta_scale, sigma_shape, sigma_scale)
    all_predictions = vectorlist_to_matrix(all_predictions)
    @info "Calculating intervals:"
    @time intervals = [PI(all_predictions[:,i]) for i in (1,2,3)]
    return mean(all_predictions, dims=1), intervals
end



function problem1_support()

    alpha_mu = 50.
    alpha_sigma= 10.

    beta_shape=1.
    beta_scale=5.

    sigma_shape=1.
    sigma_scale=5.

    fs = [pos_line_model((), alpha_mu, alpha_sigma, beta_shape, beta_scale, sigma_shape, sigma_scale) for _ in 1:15]

    prior_plot = plot(xs, [f(xs) for f in fs], legend=false, title="Priors")

    traces = [do_importance_inference(pos_line_model, xs, ys, 100, alpha_mu, alpha_sigma, beta_shape, beta_scale, sigma_shape, sigma_scale) for _ in 1:10]

    posterior = overlay(render_linear_trace, traces)

    problem1_plot = plot(prior_plot, posterior, layout=2)

    display(problem1_plot)

end



function problem2(steps=100, samples=100)

    alpha_mu = 5.
    alpha_sigma = 1.

    beta_shape = 1.
    beta_scale = 3.

    sigma_shape = 1.
    sigma_scale = 1.

    priors = [pos_line_model((), alpha_mu, alpha_sigma, beta_shape, beta_scale, sigma_shape, sigma_scale) for _ in 1:15]

    xs_test = Vector(range(minimum(howell1_children.age) - 0.5, maximum(howell1_children.age) + 0.5, length=1000))

    prior_plot = plot(xs_test, [f(xs_test) for f in priors], legend=false, title="Priors")

    xs = howell1_children.age
    ys = howell1_children.weight

    traces = [do_importance_inference(pos_line_model, xs, ys, steps, alpha_mu, alpha_sigma, beta_shape, beta_scale, sigma_shape, sigma_scale) for _ in 1:samples]

    posteriors = overlay(render_linear_trace, sample(traces, 15, replace=false))
    plot!(posteriors, title="Posteriors")

    problem2_plot = plot(prior_plot, posteriors, layout=2)
    display(problem2_plot)

end

function problem3(steps=100, samples=100)

    priors_boys = [problem3_model((), ()) for _ in 1:15]
    priors_girls = [problem3_model((), ()) for _ in 1:15]
    xs_test = Vector(range(minimum(howell1_children.age) - 0.5, maximum(howell1_children.age) + 0.5, length=1000))

    prior_plot = plot(xs_test, [f.(xs_test, ones(1000)) for f in priors_boys], color=:blue, labels=nothing)
    plot!(prior_plot, xs_test, [f.(xs_test, zeros(1000)) for f in priors_girls], color=:red, labels=nothing)
    plot!(prior_plot, title="Priors")

    xs = howell1_children.age

    ys = howell1_children.weight

    traces = [do_importance_inference(problem3_model, xs, ys, steps, howell1_children.male) for _ in 1:samples]
    posterior_funs = Gen.get_retval.(traces)


    posterior_plot = plot_howell1_age_weight_boys_girls()
    plot!(posterior_plot, xs_test, [f.(xs_test, ones(1000)) for f in sample(posterior_funs, 7, replace=false)], alpha=1/2, color=:red, legend=false)
    plot!(posterior_plot, xs_test, [f.(xs_test, zeros(1000)) for f in sample(posterior_funs, 8, replace=false)], alpha=1/2, color=:blue, legend=false)
    plot!(posterior_plot, title="Posteriors")

    problem3_plot = plot(prior_plot, posterior_plot, layout=2)
    display(problem3_plot)

end
