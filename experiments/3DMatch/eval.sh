if [ "$3" = "test" ]; then
    python test.py --cfg 3dmatch.yaml --test_epoch=$1 --benchmark=$2 --note_name $3
fi
python eval.py --cfg 3dmatch.yaml --test_epoch=$1 --benchmark=$2 --method=lgr --note_name $3
# for n in 250 500 1000 2500; do
#     python eval.py --test_epoch=$1 --num_corr=$n --run_matching --run_registration --benchmark=$2
# done
