using DataFrames: DataFrame
import CSV

using StatisticalRethinking: sr_datadir, PI
using StatisticalRethinkingCommon



@info "Reading foxes dataset"
@time foxes = CSV.read(sr_datadir("foxes.csv"), DataFrame)

"""
The first two problems are based on the same data. The data in data(foxes)
are 116 foxes from 30 different urban groups in England. These fox groups
are like street gangs. Group size (groupsize) varies from 2 to 8 individuals.
Each group maintains its own (almost exclusive) urban territory. Some ter-
ritories are larger than others. The area variable encodes this information.
Some territories also have more avgfood than others. And food influences
the weight of each fox. Assume this DAG:

A -> F -> G
     |   /
     |  /
     v v
      W

where F is avgfood, G is groupsize, A is area, and W is weight.
Use the backdoor criterion and estimate the total causal influence of A on
F. What effect would increasing the area of a territory have on the amount
of food inside it?
"""
function problem1()
    data = standardize_columns(foxes, (:area, :avgfood))

end

@gen function problem1_model(areas)

    a ~ normal(0, 1)
    b ~ normal(0, 1)


end
