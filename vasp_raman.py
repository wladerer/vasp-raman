#!/usr/bin/env python3

import os
import sys
import logging
from math import sqrt, pi
from shutil import move
import argparse

import numpy as np
from pymatgen.io.vasp import Vasprun, Poscar

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def MAT_m_VEC(m, v):
    m = np.array(m)
    v = np.array(v)
    return np.dot(m, v).tolist()

def T(m):
    return [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]

def read_vasprun(vasprun_file):
    try:
        with open(vasprun_file, 'r') as f:
            return Vasprun(f, parse_dos=False)
    except IOError as e:
        logging.error(f"Failed to open {vasprun_file}: {e}")
        sys.exit(1)

def parse_and_validate_params(param_string):
    try:
        first, last, nderiv, step_size = map(int, param_string.split('_')[:3]) + [float(param_string.split('_')[3])]
        if nderiv != 2:
            logging.error('Only nderiv=2 is supported')
            sys.exit(1)
        return first, last, nderiv, step_size
    except ValueError as e:
        logging.error(f"Invalid parameter string format: {e}")
        sys.exit(1)


def parse_poscar(poscar_fh):
    poscar = Poscar.from_file(poscar_fh.name)
    structure = poscar.structure
    nat = len(structure)
    vol = structure.volume
    b = structure.lattice.matrix.tolist()
    positions = structure.cart_coords.tolist()
    poscar_header = poscar.comment
    return nat, vol, b, positions, poscar_header


def parse_env_params(params):
    tmp = params.strip().split('_')
    if len(tmp) != 4:
        logging.error("[parse_env_params]: ERROR there should be exactly four parameters")
        sys.exit(1)
    first, last, nderiv, step_size = int(tmp[0]), int(tmp[1]), int(tmp[2]), float(tmp[3])
    return first, last, nderiv, step_size

def parse_freqdat(freqdat_fh, nat):
    freqdat_fh.seek(0) # just in case
    eigvals = [ 0.0 for i in range(nat*3) ]
    for i in range(nat*3): # all frequencies should be supplied, regardless of requested to calculate
        tmp = freqdat_fh.readline().split()
        eigvals[i] = float(tmp[0])
    return eigvals

def parse_modesdat(modesdat_fh, nat):
    modesdat_fh.seek(0) # just in case

    eigvecs = [ 0.0 for i in range(nat*3) ]
    norms =   [ 0.0 for i in range(nat*3) ]

    for i in range(nat*3): # all frequencies should be supplied, regardless of requested to calculate
        eigvec = []
        for j in range(nat):
            tmp = modesdat_fh.readline().split()
            eigvec.append([ float(tmp[x]) for x in range(3) ])

        modesdat_fh.readline().split() # empty line
        eigvecs[i] = eigvec
        norms[i] = sqrt( sum( [abs(x)**2 for sublist in eigvec for x in sublist] ) )

    return eigvecs, norms

def get_modes_from_vasprun(vasprun_fh, nat):
    vasprun = Vasprun(vasprun_fh)
    eigvals = vasprun.normalmode_eigenvals
    eigvecs = vasprun.normalmode_eigenvecs
    norms = [sqrt(sum(abs(x)**2 for sublist in eigvec for x in sublist)) for eigvec in eigvecs]
    return eigvals.tolist(), eigvecs.tolist(), norms

def get_epsilon_from_vasprun(vasprun_fh):
    vasprun = Vasprun(vasprun_fh)
    epsilon = vasprun.epsilon_static
    return epsilon.tolist()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Raman off-resonant activity calculator using VASP as a back-end.")
    parser.add_argument('-g', '--gen', help='Generate POSCAR only', action='store_true')
    parser.add_argument('-u', '--use_poscar', help='Use provided POSCAR in the folder, USE WITH CAUTION!!', action='store_true')
    args = vars(parser.parse_args())
    #
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
    
    try:
        poscar_fh = open('POSCAR.phon', 'r')
    except IOError:
        logging.error("Couldn't open input file POSCAR.phon, exiting...")
        sys.exit(1)
    
    nat, vol, b, pos, poscar_header = parse_poscar(poscar_fh)
    logging.info(f"POSCAR info: {pos}")
    
    # either use modes from vtst tools or VASP
    if os.path.isfile('freq.dat') and os.path.isfile('modes_sqrt_amu.dat'):
        try:
            freqdat_fh = open('freq.dat', 'r')
        except IOError:
            logging.error("Couldn't open freq.dat, exiting...")
            sys.exit(1)
        #
        eigvals = parse_freqdat(freqdat_fh, nat)
        freqdat_fh.close()
        #
        try: 
            modes_fh = open('modes_sqrt_amu.dat' , 'r')
        except IOError:
            logging.info("Couldn't open modes_sqrt_amu.dat, exiting...")
            sys.exit(1)
        #
        eigvecs, norms = parse_modesdat(modes_fh, nat)
        modes_fh.close()
    #
    elif os.path.isfile('vasprun.xml'):
        try:
            vasprun_fh = 'vasprun.xml'
        except IOError:
            logging.info("Couldn't open vasprun.xml, exiting...")
            sys.exit(1)
        #
        eigvals, eigvecs, norms = get_modes_from_vasprun(vasprun_fh, nat)
    #
    else:
        logging.error("Neither vasprun.xml nor freq.dat/modes_sqrt_amu.dat were found, nothing to do, exiting...")
        sys.exit(1)
    
    with open('vasp_raman.dat', 'w') as output_fh:
        output_fh.write(f"# {'mode':<4} {'freq(cm-1)':<12} {'alpha':<10} {'beta2':<10} {'activity':<10}\n")
        for i in range(first-1, last):
            eigval = eigvals[i]
            eigvec = eigvecs[i]
            norm = norms[i]
            
            logging.info(f"Mode #{i+1}: frequency {eigval:.7f} cm-1; norm: {norm:.7f}")
            #
            ra = [[0.0 for x in range(3)] for y in range(3)]
            for j in range(len(disps)):
                disp_filename = f'vasprun.{i+1:04d}.{disps[j]:+d}.xml'
                #
                try:
                    with open(disp_filename, 'r') as vasprun_fh:
                        logging.info(f"File {disp_filename} exists, parsing...")
                        eps = get_epsilon_from_vasprun(vasprun_fh)
                except IOError:
                    if args['use_poscar']:
                        logging.info(f"File {disp_filename} not found, preparing displaced POSCAR")
                        poscar = Poscar.from_file('POSCAR')
                        structure = poscar.structure
                        #
                        for k in range(nat):
                            displacement = eigvec[k] * step_size * disps[j] / norm
                            structure.translate_sites(k, displacement, frac_coords=False)
                        #
                        poscar.comment = f"{disp_filename} {step_size}"
                        poscar.structure = structure
                        poscar.write_file('POSCAR')
                    else:
                        logging.info("Using provided POSCAR")
                    #
                    if args['gen']: # only generate POSCARs
                        poscar_fn = f'POSCAR.{disps[j]:+d}.out'
                        move('POSCAR', poscar_fn)
                        logging.info(f"'-gen' mode -> {poscar_fn} with displaced atoms have been generated")
                        #
                        if j+1 == len(disps): # last iteration for the current displacements list
                            logging.info("'-gen' mode -> POSCAR files with displaced atoms have been generated, exiting now")
                            sys.exit(0)
                    else: # run VASP here
                        print("Running VASP")
                        os.system(VASP_RAMAN_RUN)
                        try:
                            move('vasprun.xml', disp_filename)
                        except IOError:
                            logging.error("ERROR Couldn't find vasprun.xml file, exiting...")
                            sys.exit(1)
                        
                        with open(disp_filename, 'r') as vasprun_fh:
                            eps = get_epsilon_from_vasprun(vasprun_fh)
                except Exception as err:
                    logging.error(f"{err}")
                    logging.error(f"Moving {disp_filename} back to 'vasprun.xml' and exiting...")
                    move(disp_filename, 'vasprun.xml')
                    sys.exit(1)
                #
                for m in range(3):
                    for n in range(3):
                        ra[m][n]   += eps[m][n] * coeffs[j]/step_size * norm * vol/(4.0*pi)
                #units: A^2/amu^1/2 =         dimless   * 1/A         * 1/amu^1/2  * A^3
            #
            alpha = (ra[0][0] + ra[1][1] + ra[2][2])/3.0
            beta2 = ( (ra[0][0] - ra[1][1])**2 + (ra[0][0] - ra[2][2])**2 + (ra[1][1] - ra[2][2])**2 + 6.0 * (ra[0][1]**2 + ra[0][2]**2 + ra[1][2]**2) )/2.0
            logging.info(f"! {i+1:4d}  freq: {eigval:10.5f}  alpha: {alpha:10.7f}  beta2: {beta2:10.7f}  activity: {45.0*alpha**2 + 7.0*beta2:10.7f}")
            output_fh.write(f"{i+1:03d}  {eigval:10.5f}  {alpha:10.7f}  {beta2:10.7f}  {45.0*alpha**2 + 7.0*beta2:10.7f}\n")
            output_fh.flush()