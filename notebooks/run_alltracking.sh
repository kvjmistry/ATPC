

python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_1bar_Efilt_0.0percent_smear_2638.h5 1 0.0percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_1bar_Efilt_0.1percent_smear_2638.h5 1 0.1percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_1bar_Efilt_0.25percent_smear_2638.h5 1 0.25percent 1;



python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_1bar_Efilt_5.0percent_smear_2638.h5 1 5.0percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_5bar_Efilt_5.0percent_smear_2638.h5 5 5.0percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_10bar_Efilt_5.0percent_smear_2638.h5 10 5.0percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_15bar_Efilt_5.0percent_smear_2638.h5 15 5.0percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_25bar_Efilt_5.0percent_smear_2638.h5 25 5.0percent 1;

python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_1bar_Efilt_0.05percent_smear_2638.h5 1 0.05percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_5bar_Efilt_0.05percent_smear_2638.h5 5 0.05percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_10bar_Efilt_0.05percent_smear_2638.h5 10 0.05percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_15bar_Efilt_0.05percent_smear_2638.h5 15 0.05percent 1;
python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_25bar_Efilt_0.05percent_smear_2638.h5 25 0.05percent 1;

# python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_1bar_Efilt_smear_2638.h5 1 nodiff 1;
# python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_5bar_Efilt_smear_2638.h5 5 nodiff 1;
# python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_10bar_Efilt_smear_2638.h5 10 nodiff 1;
# python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_15bar_Efilt_smear_2638.h5 15 nodiff 1;
# python TrackReconstruction.py ../data/ATPC_0nubb/ATPC_0nubb_25bar_Efilt_smear_2638.h5 25 nodiff 1;

rm *.pkl
rm *reco*.h5
