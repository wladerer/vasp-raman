#!/usr/bin/env python3

import logging
import argparse
import os
import re
import sys
from math import pi
from shutil import move

from pymatgen.io.vasp import Poscar, Vasprun

logging.basicConfig(filename='raman.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def MAT_m_VEC(m, v):
    p = [ 0.0 for i in range(len(v)) ]
    for i in range(len(m)):
        assert len(v) == len(m[i]), 'Length of the matrix row is not equal to the length of the vector'
        p[i] = sum( [ m[i][j]*v[j] for j in range(len(v)) ] )
    return p


def T(m):
    p = [[ m[i][j] for i in range(len( m[j] )) ] for j in range(len( m )) ]
    return p


def parse_poscar(filename: str):
    poscar = Poscar.from_file(filename)
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
        log.error("there should be exactly four parameters")
        sys.exit(1)
    [first, last, nderiv, step_size] = [int(tmp[0]), int(tmp[1]), int(tmp[2]), float(tmp[3])]

    return first, last, nderiv, step_size


def get_modes_from_OUTCAR(filename, nat):
    from math import sqrt
    eigvals = [0.0 for i in range(nat * 3)]
    eigvecs = [0.0 for i in range(nat * 3)]
    norms = [0.0 for i in range(nat * 3)]
    #
    with open(filename, 'r') as outcar_fh:
        outcar_fh.seek(0)  # just in case
        while True:
            line = outcar_fh.readline()
            if not line:
                break
            
            if "Eigenvectors after division by SQRT(mass)" in line:
                outcar_fh.readline()  # empty line
                outcar_fh.readline()  # Eigenvectors and eigenvalues of the dynamical matrix
                outcar_fh.readline()  # ----------------------------------------------------
                outcar_fh.readline()  # empty line
                
                for i in range(nat * 3):  # all frequencies should be supplied, regardless of those requested to calculate
                    outcar_fh.readline()  # empty line
                    p = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar_fh.readline())
                    eigvals[i] = float(p.group(2))
                    
                    outcar_fh.readline()  # X         Y         Z           dx          dy          dz
                    eigvec = []
                    
                    for j in range(nat):
                        tmp = outcar_fh.readline().split()
                        eigvec.append([float(tmp[x]) for x in range(3, 6)])
                
                    eigvecs[i] = eigvec
                    norms[i] = sqrt(sum([abs(x) ** 2 for sublist in eigvec for x in sublist]))
                
                return eigvals, eigvecs, norms
            
    log.error("Couldn't find 'Eigenvectors after division by SQRT(mass)' in OUTCAR. Use 'NWRITE=3' in INCAR. Exiting...")
    sys.exit(1)

def get_epsilon_from_OUTCAR(filename):
    epsilon = []
    try:
        with open(filename, 'r') as outcar_fh:
            while True:
                line = outcar_fh.readline()
                if not line:
                    break
                if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
                    outcar_fh.readline()
                    epsilon.append([float(x) for x in outcar_fh.readline().split()])
                    epsilon.append([float(x) for x in outcar_fh.readline().split()])
                    epsilon.append([float(x) for x in outcar_fh.readline().split()])
                    return epsilon
    except Exception as e:
        log.error(f"Error reading dielectric tensor from OUTCAR: {e}")
    
    try:
        vasprun = Vasprun("vasprun.xml")
        epsilon = vasprun.epsilon_static.tolist()
        return epsilon
    except Exception as e:
        log.error(f"Error reading dielectric tensor from vasprun.xml: {e}")
        raise RuntimeError("Couldn't find dielectric tensor in OUTCAR or vasprun.xml")
#
if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="Calculate Raman intensities using VASP")
    parser.add_argument('-g', '--gen', help='Generate POSCAR only', action='store_true')
    parser.add_argument('-u', '--use_poscar', help='Use provided POSCAR in the folder, USE WITH CAUTION!!', action='store_true')
    args = vars(parser.parse_args())
    
    VASP_RAMAN_RUN = os.environ.get('VASP_RAMAN_RUN')
    if VASP_RAMAN_RUN is None:
        log.error("Set environment variable 'VASP_RAMAN_RUN'")
        sys.exit(1)

    log.info(f"User Setting: VASP_RAMAN_RUN='{VASP_RAMAN_RUN}'")
    VASP_RAMAN_PARAMS = os.environ.get('VASP_RAMAN_PARAMS')
    if VASP_RAMAN_PARAMS is None:
        log.error("Environment variable 'VASP_RAMAN_PARAMS' is not set, exiting...")
        sys.exit(1)

    log.info(f"User Setting: VASP_RAMAN_PARAMS= {VASP_RAMAN_PARAMS}")
    first, last, nderiv, step_size = parse_env_params(VASP_RAMAN_PARAMS)
    assert first >= 1,    '[__main__]: First mode should be equal or larger than 1'
    assert last >= first, '[__main__]: Last mode should be equal or larger than first mode'
    if args['gen']:
        assert last == first, "[__main__]: '-gen' mode -> only generation for the one mode makes sense"
    assert nderiv == 2,   '[__main__]: At this time, nderiv = 2 is the only supported'
    disps = [-1, 1]      # hardcoded for
    coeffs = [-0.5, 0.5] # three point stencil (nderiv=2)

    try:
        nat, vol, b, pos, poscar_header = parse_poscar('POSCAR.phon')
    except Exception as e:
        log.error(f"Couldn't open or parse input file POSCAR.phon, exiting... Error: {e}")
        sys.exit(1)
    
    log.info(f"{Poscar.from_file('POSCAR.phon')}")

    if os.path.isfile('OUTCAR.phon'):
        eigvals, eigvecs, norms = get_modes_from_OUTCAR('OUTCAR.phon', nat)
    #
    else:
        log.error("Couldn't find 'OUTCAR.phon', exiting...")
        sys.exit(1)
    
    with open('vasp_raman.dat', 'w') as outfile:
        outfile.write("# mode    freq(cm-1)    alpha    beta2    activity\n")
        for i in range(first-1, last):
            eigval = eigvals[i]
            eigvec = eigvecs[i]
            norm = norms[i]
            
            log.info("Mode #%i: frequency %10.7f cm-1; norm: %10.7f" % ( i+1, eigval, norm ))
            
            ra = [[0.0 for x in range(3)] for y in range(3)]
            for j in range(len(disps)):
                disp_filename = f'OUTCAR.{i+1:04d}.{disps[j]:+d}.out'
                
                try:
                    with open(disp_filename, 'r') as outcar_fh:
                        log.info(f"File {disp_filename} exists, parsing...")
                except IOError:
                    if args['use_poscar'] is not True:
                        print(f"File {disp_filename} not found, preparing displaced POSCAR")
                        with open('POSCAR', 'w') as poscar_fh:
                            poscar_fh.write("%s %4.1e \n" % (disp_filename, step_size))
                            poscar_fh.write(poscar_header)
                            poscar_fh.write("Cartesian\n")
                            #
                            for k in range(nat):
                                pos_disp = [ pos[k][l] + eigvec[k][l]*step_size*disps[j]/norm for l in range(3)]
                                poscar_fh.write( '%15.10f %15.10f %15.10f\n' % (pos_disp[0], pos_disp[1], pos_disp[2]) )
                    
                    else:
                        log.info("Using provided POSCAR")
                    
                    if args['gen']: # only generate POSCARs
                        poscar_fn = f'POSCAR.{disps[j]:+d}.out'
                        move('POSCAR', poscar_fn)
                        log.info(f"'-gen' mode -> {poscar_fn} with displaced atoms have been generated")
                        #
                        if j+1 == len(disps): # last iteration for the current displacements list
                            log.info("'-gen' mode -> POSCAR files with displaced atoms have been generated, exiting now")
                            sys.exit(0)
                    else: 
                        log.info("Running VASP...")
                        os.system(VASP_RAMAN_RUN)
                        try:
                            move('OUTCAR', disp_filename)
                        except IOError:
                            log.error("Couldn't find OUTCAR file, exiting...")
                            sys.exit(1)
                        #
                        try:
                            eps = get_epsilon_from_OUTCAR(disp_filename)
                        except Exception as err:
                            log.error(f"Error while getting epsilon from {disp_filename}: {err}")
                            log.info(f"Moving {disp_filename} back to 'OUTCAR' and exiting...")
                            move(disp_filename, 'OUTCAR')
                            sys.exit(1)
                
                for m in range(3):
                    for n in range(3):
                        ra[m][n]   += eps[m][n] * coeffs[j]/step_size * norm * vol/(4.0*pi)
            
            alpha = (ra[0][0] + ra[1][1] + ra[2][2])/3.0
            beta2 = ( (ra[0][0] - ra[1][1])**2 + (ra[0][0] - ra[2][2])**2 + (ra[1][1] - ra[2][2])**2 + 6.0 * (ra[0][1]**2 + ra[0][2]**2 + ra[1][2]**2) )/2.0
            log.info("! %4i  freq: %10.5f  alpha: %10.7f  beta2: %10.7f  activity: %10.7f " % (i+1, eigval, alpha, beta2, 45.0*alpha**2 + 7.0*beta2))
            outfile.write("%03i  %10.5f  %10.7f  %10.7f  %10.7f\n" % (i+1, eigval, alpha, beta2, 45.0*alpha**2 + 7.0*beta2))
