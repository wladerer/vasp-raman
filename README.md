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


## Calculation Prep

You must do some work beforehand to have this script do the rest. In the directory in which you would like to run `vasp_raman.py` please include the following files:

- INCAR        - should contain `NWRITE =3`, `LEPSILON=.TRUE`, and `IBRION = {5,6,7,8}` 
- OUTCAR.phon  - should contain 'Eigenvectors after division by SQRT(mass)' 
- POSCAR.phon  
- POTCAR       
- KPOINTS      

Where the `.phon` extension indicates that this file has come from either a DFPT or finite-differences calculation.

Here is an example INCAR that you can use
```
ISTART = 0
NWRITE = 3
LPLANE = .FALSE.
KPAR = 8

ENCUT = 400.0
PREC = Accurate
EDIFF = 1.0E-8
ISMEAR = 0
SIGMA = 0.05
LEPSILON=.TRUE.

LREAL = Automatic
ADDGRID = .TRUE.
LCHARG = .FALSE.
```

Note that this is not the only configuration. You can have multistep options as well (see the examples in the original repository). 


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
