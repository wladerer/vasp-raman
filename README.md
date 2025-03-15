# NOT THE ORIGINAL REPO PLEASE CITE ALEXANDR and SHANNON

This is a forked version of this code. I have added modern logging, python3 syntax, and some more robust runtime features that will help the user debug. There are now external dependencies (`pymatgen` and `numpy`) but I assume this is only a minor issue to computational chemists at this point. 

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

You must do some work beforehand to have this script do the rest. 



Note that this is not the only configuration. You can have multistep options as well (see the examples in the original repository). I am currently working on making this compatible with slurm job arrays so everything can be done in parallel.  

Now that we have run the DFPT or finite-differences calculation, we can move onto running the script. In the directory in which you would like to run `vasp_raman.py` please include the following files:

- INCAR        - should contain `NWRITE =3`, `LEPSILON=.TRUE`, and `IBRION = {5,6,7,8}` 
- OUTCAR.phon  - should contain 'Eigenvectors after division by SQRT(mass)' 
- POSCAR.phon  
- POTCAR       
- KPOINTS      

Where the `.phon` extension indicates that this file has come from either a DFPT or finite-differences calculation. `POTCAR` and `KPOINTS` are standard files and nothing special needs to be done to prepare these. 

Here is an example INCAR that you can use
```
ISTART = 0
NWRITE = 3
LPLANE = False
KPAR = 8

ENCUT = 400.0
PREC = Accurate
EDIFF = 1.0E-8
ISMEAR = 0
SIGMA = 0.05
LEPSILON = False

LREAL = False
ADDGRID = False
LCHARG = False
```

Now we can move onto submitting this job. 

An example of SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=Example
#SBATCH --output=job.out
#SBATCH --ntasks=32
#SBATCH --time=01:00:00
#SBATCH --partition=standard

cd $SLURM_SUBMIT_DIR

ulimit -s unlimited  # remove limit on stack size

export VASP_RAMAN_RUN='mpirun -n 32 path/to/vasp'
export VASP_RAMAN_PARAMS='01_10_2_0.01'

python3 vasp_raman.py 
```

After a few minutes you should start to see several OUTCAR files. These will be processed by the script and written to `vasp_raman.dat` which has the extracted information required to reproduce the Raman spectrum. You can plot the results using the provided script `plot_raman.py` using the command line. 

```
python3 plot_raman.py vasp_raman.dat

## How to cite

Use [Bibtext](https://raw.githubusercontent.com/raman-sc/VASP/master/vasp_raman_py.bib) or [RIS](https://raw.githubusercontent.com/raman-sc/VASP/master/vasp_raman_py.ris) file for citation.

## Contributors

Alexandr Fonari (Georgia Tech, PIs: J.-L. Bredas/V. Coropceanu): [Email](mailto:alexandr.fonari[nospam]gatech.edu)  
Shannon Stauffer (UT Austin, PI: G. Henkelman): [Email](mailto:stauffers[nospam]utexas.edu).
