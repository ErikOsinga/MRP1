# Bash script to run monte carlo and the statistics for all filename
pwd
echo $1
TMP='biggest_selection'
time python mc_full_sample_correct.py $TMP $1
time python statistics_mc_full_sample_correct.py $TMP $1
TMP='value_added_selection'
time python mc_full_sample_correct.py $TMP $1
time python statistics_mc_full_sample_correct.py $TMP $1
TMP='value_added_selection_MG'
time python mc_full_sample_correct.py $TMP $1
time python statistics_mc_full_sample_correct.py $TMP $1
TMP='value_added_selection_NN'
time python mc_full_sample_correct.py $TMP $1
time python statistics_mc_full_sample_correct.py $TMP $1
TMP='value_added_compmatch'
time python mc_full_sample_correct.py $TMP $1
time python statistics_mc_full_sample_correct.py $TMP $1
