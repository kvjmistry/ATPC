#!/bin/bash

MODE=Bi_ion

# 1 bar enriched
Enrichment=enr; Pressure=1; Thickness=4; CubeSize=6.182
sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

Enrichment=enr; Pressure=1; Thickness=6; CubeSize=6.182
sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

Enrichment=enr; Pressure=1; Thickness=8; CubeSize=6.182
sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

Enrichment=enr; Pressure=1; Thickness=10; CubeSize=6.182
sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

Enrichment=enr; Pressure=1; Thickness=12; CubeSize=6.182
sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

Enrichment=enr; Pressure=1; Thickness=18; CubeSize=6.182
sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize


# 5 bar enriched
# Enrichment=enr; Pressure=5; Thickness=4; CubeSize=3.615
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=5; Thickness=6; CubeSize=3.615
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=5; Thickness=8; CubeSize=3.615
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=5; Thickness=10; CubeSize=3.615
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=5; Thickness=12; CubeSize=3.615
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=5; Thickness=18; CubeSize=3.615
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # 10 bar enriched
# Enrichment=enr; Pressure=10; Thickness=4; CubeSize=2.870
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=10; Thickness=6; CubeSize=2.870
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=10; Thickness=8; CubeSize=2.870
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=10; Thickness=10; CubeSize=2.870
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=10; Thickness=12; CubeSize=2.870
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=10; Thickness=18; CubeSize=2.870
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # 15 bar enriched
# Enrichment=enr; Pressure=15; Thickness=4; CubeSize=2.507
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=15; Thickness=6; CubeSize=2.507
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=15; Thickness=8; CubeSize=2.507
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=15; Thickness=10; CubeSize=2.507
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=15; Thickness=12; CubeSize=2.507
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=15; Thickness=18; CubeSize=2.507
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # 25 bar enriched
# Enrichment=enr; Pressure=25; Thickness=4; CubeSize=2.114
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=25; Thickness=6; CubeSize=2.114
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=25; Thickness=8; CubeSize=2.114
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=25; Thickness=10; CubeSize=2.114
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=25; Thickness=12; CubeSize=2.114
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=25; Thickness=18; CubeSize=2.114
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # =====================================================================================


# # 1 bar natural
# Enrichment=nat; Pressure=1; Thickness=4; CubeSize=13.319
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=1; Thickness=6; CubeSize=13.319
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=1; Thickness=8; CubeSize=13.319
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=1; Thickness=10; CubeSize=13.319
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=1; Thickness=12; CubeSize=13.319
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=1; Thickness=18; CubeSize=13.319
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize


# # 5 bar natural
# Enrichment=nat; Pressure=5; Thickness=4; CubeSize=7.789
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=5; Thickness=6; CubeSize=7.789
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=5; Thickness=8; CubeSize=7.789
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=5; Thickness=10; CubeSize=7.789
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=5; Thickness=12; CubeSize=7.789
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=5; Thickness=18; CubeSize=7.789
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # 10 bar natural
# Enrichment=nat; Pressure=10; Thickness=4; CubeSize=6.182
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=10; Thickness=6; CubeSize=6.182
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=10; Thickness=8; CubeSize=6.182
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=10; Thickness=10; CubeSize=6.182
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=10; Thickness=12; CubeSize=6.182
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=enr; Pressure=10; Thickness=18; CubeSize=6.182
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # 15 bar natural
# Enrichment=nat; Pressure=15; Thickness=4; CubeSize=5.301
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=15; Thickness=6; CubeSize=5.301
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=15; Thickness=8; CubeSize=5.301
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=15; Thickness=10; CubeSize=5.301
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=15; Thickness=12; CubeSize=5.301
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=15; Thickness=18; CubeSize=5.301
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# # 25 bar natural
# Enrichment=nat; Pressure=25; Thickness=4; CubeSize=4.556
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=25; Thickness=6; CubeSize=4.556
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=25; Thickness=8; CubeSize=4.556
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=25; Thickness=10; CubeSize=4.556
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=25; Thickness=12; CubeSize=4.556
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize

# Enrichment=nat; Pressure=25; Thickness=18; CubeSize=4.556
# sbatch --array=1-10 RunGammaGeneration.sh $Enrichment $Pressure $Thickness $MODE $CubeSize