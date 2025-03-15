# NOT THE ORIGINAL REPO PLEASE CITE ALEXANDR and SHANNON

This is a forked version of this code. I have added modern logging, python3 syntax, and some more robust runtime features that will help the user debug. There is now an external dependency, `pymatgen`, but I assume this is only a minor issue to computational chemists at this point. 

## Global variables

Please set `VASP_RAMAN_PARAMS` and `VASP_RAMAN_RUN` in your submission script.

  - `VASP_RAMAN_PARAMS` is defined as `FIRST-MODE_LAST-MODE_NDERIV_STEPSIZE` where:
      - `FIRST_MODE` - integer, first mode for which derivative of the polarizability is computed
      - `LAST-MODE`  - integer, last mode for which derivative of the polarizability is computed
      - `NDERIV`     - integer, scheme for finite difference, **currently** only value `2` is supported
      - `STEPSIZE`   - float, step-size for finite difference, in Angstroms
        
    Example: `VASP_RAMAN_PARAMS=01_10_2_0.01`

  - `VASP_RAMAN_RUN` the command to execute VASP (can contain MPI call):  

An example of SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=Example
#SBATCH --output=job.out
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --partition=debug

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited  # remove limit on stack size

export VASP_RAMAN_RUN='srun /u/afonari/vasp.5.3.2/vasp.5.3/vasp'
export VASP_RAMAN_PARAMS='01_10_2_0.01'

python3 vasp_raman.py > vasp_raman.out
```


## How to cite

Use [Bibtext](https://raw.githubusercontent.com/raman-sc/VASP/master/vasp_raman_py.bib) or [RIS](https://raw.githubusercontent.com/raman-sc/VASP/master/vasp_raman_py.ris) file for citation.

## Contributors

Alexandr Fonari (Georgia Tech, PIs: J.-L. Bredas/V. Coropceanu): [Email](mailto:alexandr.fonari[nospam]gatech.edu)  
Shannon Stauffer (UT Austin, PI: G. Henkelman): [Email](mailto:stauffers[nospam]utexas.edu).
