python test.py --cfg config.yaml --snapshot=../../output/3DMatch_release/snapshots/snapshot.pth.tar --benchmark=3DLoMatch --note_name release
python eval.py --cfg config.yaml --benchmark=3DLoMatch --method=lgr --note_name release
