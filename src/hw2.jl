using StatisticalRethinking: sr_datadir
import CSV
using DataFrames: DataFrame

howell1 = CSV.read(sr_datadir("Howell1.csv"), DataFrame)

"""
Construct a linear regression of weight as predicted by height, using the
adults (age 18 or greater) from the Howell1 dataset. The heights listed below
were recorded in the !Kung census, but weights were not recorded for these
individuals. Provide predicted weights and 89% compatibility intervals for
each of these individuals. That is, fill in the table below, using model-based
predictions.
"""
function problem1()
end
