folder=ATPC_0nubb

mkdir -p $folder/1bar/nodiff
mkdir -p $folder/1bar/5percent
mkdir -p $folder/1bar/0.25percent
mkdir -p $folder/1bar/0.5percent
mkdir -p $folder/1bar/0.1percent
mkdir -p $folder/5bar/nodiff
mkdir -p $folder/10bar/nodiff
mkdir -p $folder/15bar/nodiff

cd ${folder}/15bar/nodiff/

ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl


cd ../../
cd 10bar/nodiff
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl


cd ../../
cd 5bar/nodiff
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl

cd ../../
cd 1bar/nodiff
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl

cd ../../
cd 1bar/5percent
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl


cd ../../
cd 1bar/0.25percent
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl


cd ../../
cd 1bar/0.5percent
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl

cd ../../
cd 1bar/0.1percent
ls ./*.tar | xargs -n 1 -P 8 -I {} bash -c '
    file={}
    if tar -xvf "$file"; then
        echo "Extracted $file successfully."
        rm ${file}
    else
        echo "Failed to extract $file."
    fi
'
mkdir -p reco pkl
mv *.h5 reco
mv *.pkl pkl


cd ../../../