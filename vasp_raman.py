#!/usr/bin/env python3

import os
import sys
import logging
from shutil import move
import argparse


from pymatgen.io.vasp import Vasprun, Outcar, Poscar
import numpy as np

def funclog(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Running function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

#eventually convert to numpy
def MAT_m_VEC(m, v):
    p = [0.0 for i in range(len(v))]
    for i in range(len(m)):
        assert len(v) == len(m[i]), 'Length of the matrix row is not equal to the length of the vector'
        p[i] = sum([m[i][j] * v[j] for j in range(len(v))])
    return p

#why is this even a function
def T(m):
    return [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]


#convert to pymatgen
@funclog
def parse_poscar(filename: str):
    poscar = Poscar.from_file(filename)
    structure = poscar.structure

    n_atoms = len(structure)
    volume = structure.volume
    b = structure.lattice.matrix.tolist()
    positions = structure.cart_coords.tolist()
    poscar_header = poscar.comment

    return n_atoms, volume, b, positions, poscar_header

@funclog
def parse_env_params(params):
    tmp = params.strip().split('_')
    if len(tmp) != 4:
        logging.error("There should be exactly four parameters")
        sys.exit(1)
    first, last, nderiv, step_size = int(tmp[0]), int(tmp[1]), int(tmp[2]), float(tmp[3])
    logging.debug(f"First mode: {first}, last mode: {last}, nderiv: {nderiv}, step size: {step_size}")
    return first, last, nderiv, step_size


@funclog
def get_modes_from_vasprun(filename: str):
    vasprun = Vasprun(filename, parse_dos=False, parse_projected_eigen=False)
    eigvals = vasprun.normalmode_eigenvals
    eigvecs = vasprun.normalmode_eigenvecs
    norms = [np.sqrt(sum(abs(x)**2 for x in eigvec.flatten())) for eigvec in eigvecs]
    
    logging.debug(f"Eigenvalues: {eigvals}")
    logging.debug(f"Eigenvectors: {eigvecs}")
    logging.debug(f"Norms: {norms}")
    
    return eigvals, eigvecs, norms


@funclog
def get_epsilon_from_vasprun(filename: str):
    try:
        vasprun = Vasprun(filename, parse_dos=False, parse_projected_eigen=False) 
        epsilon = vasprun.epsilon_static
        logging.debug(f"Dielectric tensor: {epsilon}")
        return epsilon
    except Exception as e:
        logging.error(f"Error parsing dielectric tensor from vasprun.xml: {e}")
        raise RuntimeError("Couldn't find dielectric tensor in vasprun.xml")
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Raman off-resonant activity calculator using VASP as a back-end.")
    parser.add_argument('-g', '--gen', help='Generate POSCAR only', action='store_true')
    parser.add_argument('-u', '--use_poscar', help='Use provided POSCAR in the folder, USE WITH CAUTION!!', action='store_true')
    parser.add_argument('-o','--output', help='Output file name', default='raman.dat')
    args = vars(parser.parse_args())
    
    VASP_RAMAN_RUN = os.environ.get('VASP_RAMAN_RUN')
    if VASP_RAMAN_RUN is None:
        logging.error("ERROR Set environment variable 'VASP_RAMAN_RUN'")
        sys.exit(1)
    logging.info(f"User Setting: VASP_RAMAN_RUN='{VASP_RAMAN_RUN}'")
    
    VASP_RAMAN_PARAMS = os.environ.get('VASP_RAMAN_PARAMS')
    if VASP_RAMAN_PARAMS is None:
        logging.error("Environment variable 'VASP_RAMAN_PARAMS' is not set, exiting...")
        sys.exit(1)
    logging.info(f"User Setting: VASP_RAMAN_PARAMS= {VASP_RAMAN_PARAMS}")
    first, last, nderiv, step_size = parse_env_params(VASP_RAMAN_PARAMS)
    assert first >= 1, 'First mode should be equal or larger than 1'
    assert last >= first, 'Last mode should be equal or larger than first mode'
    if args['gen']:
        assert last == first, "-gen' mode -> only generation for the one mode makes sense"

    assert nderiv == 2, 'Only nderiv = 2 is supported'
    disps = [-1, 1]      # hardcoded for
    coeffs = [-0.5, 0.5] # three point stencil (nderiv=2)
    
    if os.path.isfile('vasprun.xml.phon'):
        eigvals, eigvecs, norms = get_modes_from_vasprun('vasprun.xml.phon')
        logging.debug(f"Eigenvalues: {eigvals}")
        logging.debug(f"Eigenvectors: {eigvecs}")
        logging.debug(f"Norms: {norms}")
    
    else:
        logging.error("Couldn't find 'vasprun.xml.phon', exiting...")
        sys.exit(1)

    with open(args['output'], 'w') as outfile:
        outfile.write(f"{'mode':>4}\t{'freq(cm-1)':>12}\t{'alpha':>10}\t{'beta2':>10}\t{'activity':>10}\n")
        for i in range(first-1, last):
            eigval = eigvals[i]
            eigvec = eigvecs[i]
            norm = norms[i]
            
            logging.info(f"Mode #{i+1}:\tfrequency {eigval:.7f} cm-1;\tnorm: {norm:.7f}")
           
            ra = [[0.0 for x in range(3)] for y in range(3)]
            for j in range(len(disps)):
                disp_filename = f'vasprun.xml.{i+1:04d}.{disps[j]:+d}.out'
                
                try:
                    vasprun = Vasprun(disp_filename, parse_dos=False, parse_projected_eigen=False)
                    logging.info(f"File {disp_filename} exists, parsing...")

                except IOError:
                    if args['use_poscar']:
                        logging.info(f"File {disp_filename} not found, preparing displaced POSCAR")
                        structure = vasprun.final_structure
                        n_atoms = len(structure)
                        structure = structure.copy()
                        for k in range(n_atoms):
                            displacement = eigvec[k] * step_size * disps[j] / norm
                            structure.translate_sites(k, displacement, frac_coords=False)
                        
                        poscar = Poscar(structure)
                        poscar.write_file('POSCAR')
                    else:
                        logging.info("Using provided POSCAR")
                    
                    if args['gen']: # only generate POSCARs
                        poscar_fn = 'POSCAR.%+d.out' % disps[j]
                        move('POSCAR', poscar_fn)
                        logging.info("'-gen' mode -> {poscar_fn} with displaced atoms have been generated")
                        
                        if j+1 == len(disps): # last iteration for the current displacements list
                            logging.info("'-gen' mode -> POSCAR files with displaced atoms have been generated, exiting now")
                            sys.exit(0)
                    else: 
                        logging.info("Running VASP")
                        os.system(VASP_RAMAN_RUN)
                        #check if converged 
                        try:
                            vasprun = Vasprun('vasprun.xml', parse_dos=False, parse_projected_eigen=False, parse_eigen=False) 
                            if not vasprun.converged:
                                logging.error("VASP didn't converge, exiting...")
                                sys.exit(1)
                        except Exception as e:
                            logging.error(f"Error parsing vasprun.xml: {e}")
                            sys.exit(1)

                        try:
                            move('vasprun.xml', disp_filename)
                        except IOError:
                            logging.error("ERROR Couldn't find vasprun file, exiting...")
                            sys.exit(1)
                
                try:
                    eps = get_epsilon_from_vasprun(disp_filename)
                    
                except Exception as err:
                    logging.error(f"{err}")
                    logging.error(f"Moving {disp_filename} back to 'vasprun.xml' and exiting...")
                    move(disp_filename, 'vasprun.xml')
                    sys.exit(1)
                
                for m in range(3):
                    for n in range(3):
                        volume = vasprun.final_structure.volume
                        ra[m][n]   += eps[m][n] * coeffs[j]/step_size * norm * volume/(4.0*np.pi)
                #units: A^2/amu^1/2 =         dimless   * 1/A         * 1/amu^1/2  * A^3
            
            alpha = (ra[0][0] + ra[1][1] + ra[2][2])/3.0
            beta2 = ( (ra[0][0] - ra[1][1])**2 + (ra[0][0] - ra[2][2])**2 + (ra[1][1] - ra[2][2])**2 + 6.0 * (ra[0][1]**2 + ra[0][2]**2 + ra[1][2]**2) )/2.0
            logging.info(f"! {i+1:4d}  freq: {eigval:10.5f}  alpha: {alpha:10.7f}  beta2: {beta2:10.7f}  activity: {45.0*alpha**2 + 7.0*beta2:10.7f}")

            outfile.write(f"{i+1:03d}  {eigval:10.5f}  {alpha:10.7f}  {beta2:10.7f}  {45.0*alpha**2 + 7.0*beta2:10.7f}\n")
    