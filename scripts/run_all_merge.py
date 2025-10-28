mode=$1

python3 merge_outputs.py ${mode} 1 5percent
python3 merge_outputs.py ${mode} 5 5percent
python3 merge_outputs.py ${mode} 10 5percent
python3 merge_outputs.py ${mode} 15 5percent
python3 merge_outputs.py ${mode} 25 5percent
python3 merge_outputs.py ${mode} 1 nodiff
python3 merge_outputs.py ${mode} 5 nodiff
python3 merge_outputs.py ${mode} 10 nodiff
python3 merge_outputs.py ${mode} 15 nodiff
python3 merge_outputs.py ${mode} 25 nodiff
python3 merge_outputs.py ${mode} 1 0.05percent
python3 merge_outputs.py ${mode} 5 0.05percent
python3 merge_outputs.py ${mode} 10 0.05percent
python3 merge_outputs.py ${mode} 15 0.05percent
python3 merge_outputs.py ${mode} 25 0.05percent
python3 merge_outputs.py ${mode} 1 0.0percent
python3 merge_outputs.py ${mode} 1 0.1percent
python3 merge_outputs.py ${mode} 1 0.25percent
