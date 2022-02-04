module StatisticalRethinkingCommon

import StatsBase
using StatsBase: fit, ZScoreTransform

export vectorlist_to_matrix, standardize_columns, standardize_columns!, standardize_column, standardize_column!

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

function mcmc(model, modelargs, observations=Gen.EmptyChoiceMap(), warmup=0, steps=100, kernel=Gen.mh())
    trace, = Gen.generate(model, modelargs, observations)

    warmup_num_accepted = 0
    warmup_traces = []
    for i in 1:warmup
        trace,accepted = kernel(trace, observations=observations)
        append!(warmup_traces, trace)
        warmup_num_accepted += Int64(accepted)
    end

    traces = []
    num_accepted = 0
    for i in 1:steps
        trace,accepted = kernel(trace, observations=observations)
        append!(traces, trace)
        accepted += Int64(accepted)
    end

    return traces, num_accepted, warmup_traces, warmup_num_accepted
end

end
