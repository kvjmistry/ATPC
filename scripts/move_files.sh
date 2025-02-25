
dir=ATPC_Tl

cd $dir
mkdir -p 1bar 5bar 10bar 15bar
mv *_1bar_* 1bar; mv *_5bar_* 5bar; mv *_10bar_* 10bar;mv *_15bar_* 15bar;

cd 1bar
mkdir -p 5percent 0.5percent 0.25percent 0.1percent 0.05percent nodiff nexus
mv *_0.1p*.h5 0.1percent/; mv *_0.25p*.h5 0.25percent/; mv *_5.0p*.h5 5percent/; mv *_0.5p*.h5 0.5percent/; mv *_0.05p*.h5 0.05percent/; mv *_1bar_smear*h5 nodiff/; mv *nexus*h5 nexus/

cd ..
cd 5bar
mkdir -p 5percent nodiff nexus
mv *_5.0p*.h5 5percent/; mv *bar_smear*h5 nodiff/; mv *nexus*h5 nexus/

cd ..
cd 10bar
mkdir -p 5percent nodiff nexus
mv *_5.0p*.h5 5percent/; mv *bar_smear*h5 nodiff/; mv *nexus*h5 nexus/

cd ..
cd 15bar
mkdir -p 5percent nodiff nexus
mv *_5.0p*.h5 5percent/; mv *bar_smear*h5 nodiff/; mv *nexus*h5 nexus/

cd ../../