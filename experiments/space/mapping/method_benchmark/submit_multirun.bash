#!/bin/bash

#SBATCH -o "slurm_%j.out"
#SBATCH -e "slurm_%j.err"
#SBATCH -J cpu_sub
#SBATCH --qos cpu_normal
#SBATCH -p cpu_p
#SBATCH -t 1:00:00
#SBATCH --mem 20G

source ~/.bashrc
conda activate moscot

# unset SLURM_CPU_BIND
# python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.1 method.alpha=0.1 method.rep=x method.cost=cosine method.tau=1.0 method.quad=pca_spatial &
python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.1,0.05 method.alpha=0.1,0.3,0.5 method.rep=x,pca method.cost=cosine,sq_euclidean,geodesic method.tau=0.99,0.9,0.5 method.quad=spatial,pca_spatial &
# python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.1,0.05 method.alpha=0.1,0.3,0.5 method.rep=x,pca method.cost=cosine,sq_euclidean,geodesic method.tau=1.0,0.995 method.quad=spatial,pca_spatial &

# last run
# python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.01,0.001 method.alpha=0.1,0.2,0.5 method.rep=x,pca method.cost=cosine,sq_euclidean,geodesic method.tau=1.0,0.95 &

# python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.01 method.alpha=0.1 method.rep=x method.cost=cosine method.tau=1.0 &
# python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.1 method.alpha=0.1 method.rep=x method.cost=geodesic method.tau=1.0 &
# python experiment.py --multirun launcher=icb method.name=MOSCOT method.epsilon=0.001,0.0001 method.alpha=0.1,0.2,0.5 method.rep=x,pca method.cost=cosine,sq_euclidean &
# python experiment.py --multirun launcher=icb method.name=TANGRAM method.num_epochs=500,1000,2000,3000 method.learning_rate=0.1,0.01 &
# python experiment.py --multirun launcher=icb method.name=GIMVI method.epochs=200,400,6000,1000 method.n_latent=10,20,50 &
wait
