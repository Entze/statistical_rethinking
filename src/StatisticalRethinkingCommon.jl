module StatisticalRethinkingCommon

using Gen
using Statistics
import StatsBase
using StatsBase: fit, ZScoreTransform
using ProgressMeter

export vectorlist_to_matrix, vector_to_columnmatrix, vector_to_rowmatrix, standardize_columns, standardize_columns!, standardize_column, standardize_column!, sequential_sampling, sampling, mcmc, infer

function vectorlist_to_matrix(list_of_vectors)
    dim1 = length(list_of_vectors)
    dim2 = length(list_of_vectors[1])
    matrix_type = typeof(list_of_vectors[1][1])
    matrix = zeros(matrix_type, dim1, dim2)
    for i in 1:dim1
        for j in 1:dim2
            matrix[i,j] = list_of_vectors[i][j]
        end
    end
    return matrix
end

function vector_to_columnmatrix(vector)
    return reshape(vector, (length(vector), 1))
end

function vector_to_rowmatrix(vector)
    return reshape(vector, (1, length(vector)))
end

function standardize_columns(dataframe, columnsels; center=true, scale=true)
    new_dataframe = copy(dataframe)
    dts = standardize_columns!(new_dataframe, columnsels, center=center, scale=scale)
    return new_dataframe, dts
end

function standardize_columns!(dataframe, columnsels; center=true, scale=true)
    for columnsel in columnsels
        dts = standardize_column!(dataframe, columnsel, center=center, scale=scale)
    end
end

function standardize_column(dataframe, columnsel; center=true, scale=true)
    column = dataframe[:, columnsel]
    dt = fit(ZScoreTransform, column, dims=1, center=center, scale=scale)
    return Statsbase.transfrom(dt, column), dt
end

function standardize_column!(dataframe, columnsel; center=true, scale=true)
    column = dataframe[!, columnsel]
    dt = fit(ZScoreTransform, column, dims=1, center=center, scale=scale)
    dataframe[!, columnsel] = StatsBase.transform(dt, column)
    return dt
end

function sequential_sampling(model, modelargs=(), observations=Gen.EmptyChoiceMap(); steps=100, samples=1)
    return [Gen.importance_resampling(model, modelargs, observations, samples)[1] for _ in 1:steps]
end

function sampling(model, modelargs=(), observations=Gen.EmptyChoiceMap(); steps=100, samples=1, progress=nothing)
    if !isnothing(progress) && progress isa Bool && progress
        progress = Progress(steps * samples, desc="Sampling: " , showspeed=true)
    end
    lk = ReentrantLock()
    traces = []
    lml_ests = []
    Threads.@threads for _ in 1:steps
        trace,lml_est = Gen.importance_resampling(model, modelargs, observations, samples)
        lock(lk) do
            push!(traces, trace)
            push!(lml_ests, lml_est)
            if !isnothing(progress) && progress isa Progress
                update!(progress, samples)
            end
        end
    end

    if !isnothing(progress) && progress isa Progress
        finish!(progress)
    end
    return traces, lml_ests
end

function mcmc(model, modelargs=(), observations=Gen.EmptyChoiceMap(); selection=nothing, warmup=0, steps=100, kernel=Gen.mh, check=false, progress=nothing)
    trace, = Gen.generate(model, modelargs, observations)

    if !isnothing(progress) && progress isa Bool && progress
        progress = Progress(warmup + steps, desc="Prewarmup: " , showspeed=true)
    end

    if isnothing(selection)
        obs = []
        for (addr,) in Gen.get_values_shallow(observations)
            push!(obs, addr)
        end
        selection = Gen.complement(Gen.select(obs...))
    end

    if !isnothing(progress) && progress isa Progress
        update!(progress, 0, desc="Warmup:    ")
    end

    warmup_num_accepted = 0
    warmup_traces = []
    for _ in 1:warmup
        trace,accepted = kernel(trace, selection, observations=observations, check=check)
        push!(warmup_traces, trace)
        warmup_num_accepted += Int(accepted)
        if !isnothing(progress) && progress isa Progress
            next!(progress)
        end
    end

    if !isnothing(progress) && progress isa Progress
        update!(progress, 0, desc="MCMC:      ")
    end

    num_accepted = 0
    traces = []
    for _ in 1:steps
        trace,accepted = kernel(trace, selection, observations=observations, check=check)
        push!(traces, trace)
        num_accepted += Int(accepted)
        if !isnothing(progress) && progress isa Progress
            next!(progress)
        end
    end

    if !isnothing(progress) && progress isa Progress
        finish!(progress)
    end


    return traces, num_accepted, warmup_traces, warmup_num_accepted
end

function infer(traces, model, modelargs=(), parameter_addresses=(); combine=(p -> mean(p, dims=1)))
    choicemaps = Gen.get_choices.(traces)
    parameter_collection = []
    for (i, address) in enumerate(parameter_addresses)
        if i == 1
            parameter_collection = vector_to_columnmatrix(Gen.get_value.(choicemaps, address))
            continue
        end
        parameter_collection = hcat(parameter_collection, vector_to_columnmatrix(Gen.get_value.(choicemaps, address)))
    end
    parameters = combine(parameter_collection)
    constraints = Gen.choicemap()
    for (i,address) in enumerate(parameter_addresses)
        constraints[address] = parameters[i]
    end
    new_trace = Gen.generate(model, modelargs, constraints)
    return new_trace
end

end
