export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate em
python run_inversion.py 100 0 10 10