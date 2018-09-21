# Bash script to run monte carlo and the statistics for given filename
# Execute with bash statistics_4datasets.sh <name of the file>

pwd
echo $1 $2
time python mc.py $1 $2
time python statistics_mc.py $1 $2
