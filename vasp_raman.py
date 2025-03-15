#!/usr/bin/env python3

import os
import re
import sys
import logging
from math import sqrt, pi
from shutil import move
import argparse

from pymatgen.io.vasp import Vasprun
import numpy as np


def format_array(arr):
    """Formats a NumPy array or a nested list for better logging."""
    if isinstance(arr, np.ndarray):
        return np.array2string(arr, precision=4, separator=", ", suppress_small=True)
    elif isinstance(arr, list):  # Convert lists to NumPy for consistent formatting
        return np.array2string(np.array(arr), precision=4, separator=", ", suppress_small=True)
    return str(arr)

def funclog(func):
    def wrapper(*args, **kwargs):
        logging.info(f"Running function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename="raman.log",  # Redirect logs to a file
    filemode="w",  # Overwrite log file each run (use "a" for append mode)
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
def parse_poscar(poscar_fh):
    poscar_fh.seek(0) # just in case
    lines = poscar_fh.readlines()
    
    scale = float(lines[1])
    if scale < 0.0:
        logging.error("Negative scale not implemented.")
        sys.exit(1)
    
    b = []
    for i in range(2, 5):
        b.append([float(x)*scale for x in lines[i].split()[:3]])
    
    vol = b[0][0]*b[1][1]*b[2][2] + b[1][0]*b[2][1]*b[0][2] + b[2][0]*b[0][1]*b[1][2] - \
          b[0][2]*b[1][1]*b[2][0] - b[2][1]*b[1][2]*b[0][0] - b[2][2]*b[0][1]*b[1][0]
    
    try:
        num_atoms = [int(x) for x in lines[5].split()]
        line_at = 6
    except ValueError:
        #symbols = [x for x in lines[5].split()] # not used but keep just in case
        num_atoms = [int(x) for x in lines[6].split()]
        line_at = 7
    nat = sum(num_atoms)
    
    if lines[line_at][0].lower() == 's':
        line_at += 1
    
    if (lines[line_at][0].lower() == 'c' or lines[line_at][0].lower() == 'k'):
        is_scaled = False
    else:
        is_scaled = True
    
    line_at += 1
    
    positions = []
    for i in range(line_at, line_at + nat):
        pos = [float(x) for x in lines[i].split()[:3]]
        
        if is_scaled:
            pos = MAT_m_VEC(T(b), pos)
        
        positions.append(pos)

    logging.debug(f"Volume: {vol:.7f}")
    logging.debug(f"Lattice vectors: {format_array(b)}")
    logging.debug(f"Number of atoms: {nat}")
    logging.debug(f"Positions: {format_array(positions)}")

    
    poscar_header = ''.join(lines[1:line_at-1]) # will add title and 'Cartesian' later
    return nat, vol, b, positions, poscar_header

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
def parse_freqdat(freqdat_fh, nat):
    freqdat_fh.seek(0) 
    eigvals = [ 0.0 for i in range(nat*3) ]
    for i in range(nat*3): # all frequencies should be supplied, regardless of requested to calculate
        tmp = freqdat_fh.readline().split()
        eigvals[i] = float(tmp[0])
    
    logging.debug(f"Eigenvalues: {format_array(eigvals)}")
    return eigvals

@funclog
def parse_modesdat(modesdat_fh, nat):
    modesdat_fh.seek(0) 

    eigvecs = [ 0.0 for i in range(nat*3) ]
    norms =   [ 0.0 for i in range(nat*3) ]

    for i in range(nat*3): # all frequencies should be supplied, regardless of requested to calculate
        eigvec = []
        for j in range(nat):
            tmp = modesdat_fh.readline().split()
            eigvec.append([ float(tmp[x]) for x in range(3) ])

        modesdat_fh.readline().split() 
        eigvecs[i] = eigvec
        norms[i] = sqrt( sum( [abs(x)**2 for sublist in eigvec for x in sublist] ) )

    return eigvecs, norms

@funclog
def get_modes_from_OUTCAR(outcar_fh, nat):
    eigvals = [ 0.0 for i in range(nat*3) ]
    eigvecs = [ 0.0 for i in range(nat*3) ]
    norms   = [ 0.0 for i in range(nat*3) ]
    
    outcar_fh.seek(0) 
    while True:
        line = outcar_fh.readline()
        if not line:
            break
        
        if "Eigenvectors after division by SQRT(mass)" in line:
            outcar_fh.readline() # empty line
            outcar_fh.readline() # Eigenvectors and eigenvalues of the dynamical matrix
            outcar_fh.readline() # ----------------------------------------------------
            outcar_fh.readline() # empty line
            
            logging.debug("Parsing eigenvectors and eigenvalues from OUTCAR")
            for i in range(nat*3): # all frequencies should be supplied, regardless of those requested to calculate
                outcar_fh.readline() # empty line
                p = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar_fh.readline())
                eigvals[i] = float(p.group(2))
                #
                outcar_fh.readline() # X         Y         Z           dx          dy          dz
                eigvec = []
                
                for j in range(nat):
                    tmp = outcar_fh.readline().split()
                    eigvec.append([ float(tmp[x]) for x in range(3,6) ])
                    
                eigvecs[i] = eigvec
                norms[i] = sqrt( sum( [abs(x)**2 for sublist in eigvec for x in sublist] ) )
            
            return eigvals, eigvecs, norms
        
    logging.error("Couldn't find 'Eigenvectors after division by SQRT(mass)' in OUTCAR. Use 'NWRITE=3' in INCAR. Exiting...")
    sys.exit(1)

@funclog
def get_epsilon_from_OUTCAR(outcar_fh):
    try:
        vasprun = Vasprun('vasprun.xml', parse_dos=False, parse_projected_eigen=False) 
        epsilon = vasprun.epsilon_static
        logging.debug(f"Dielectric tensor: {format_array(epsilon)}")
        return epsilon
    except Exception as e:
        logging.error(f"Error parsing dielectric tensor from vasprun.xml: {e}")
        raise RuntimeError("Couldn't find dielectric tensor in vasprun.xml")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Raman off-resonant activity calculator using VASP as a back-end.")
    parser.add_argument('-g', '--gen', help='Generate POSCAR only', action='store_true')
    parser.add_argument('-u', '--use_poscar', help='Use provided POSCAR in the folder, USE WITH CAUTION!!', action='store_true')
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
    
    try:
        poscar_fh = open('POSCAR.phon', 'r')
    except IOError:
        logging.error("Couldn't open input file POSCAR.phon, exiting...")
        sys.exit(1)
    
    nat, vol, b, pos, poscar_header = parse_poscar(poscar_fh)
    logging.info(f"POSCAR info: {format_array(pos)}")
    
    # either use modes from vtst tools or VASP
    if os.path.isfile('freq.dat') and os.path.isfile('modes_sqrt_amu.dat'):
        try:
            freqdat_fh = open('freq.dat', 'r')
        except IOError:
            logging.error("Couldn't open freq.dat, exiting...")
            sys.exit(1)
        
        eigvals = parse_freqdat(freqdat_fh, nat)
        freqdat_fh.close()
        
        try: 
            modes_fh = open('modes_sqrt_amu.dat' , 'r')
        except IOError:
            logging.error("Couldn't open modes_sqrt_amu.dat, exiting...")
            sys.exit(1)
       
        eigvecs, norms = parse_modesdat(modes_fh, nat)
        modes_fh.close()
    
    elif os.path.isfile('OUTCAR.phon'):
        try:
            outcar_fh = open('OUTCAR.phon', 'r')
        except IOError:
            logging.error("Couldn't open OUTCAR.phon, exiting...")
            sys.exit(1)
   
        eigvals, eigvecs, norms = get_modes_from_OUTCAR(outcar_fh, nat)
        logging.debug(f"Eigenvalues: {eigvals}")
        logging.debug(f"Eigenvectors: {eigvecs}")
        logging.debug(f"Norms: {norms}")

        outcar_fh.close()
   
    else:
        logging.error("Neither OUTCAR.phon nor freq.dat/modes_sqrt_amu.dat were found, nothing to do, exiting...")
        sys.exit(1)
    
    output_fh = open('vasp_raman.dat', 'w')
    output_fh.write("# mode    freq(cm-1)    alpha    beta2    activity\n")
    for i in range(first-1, last):
        eigval = eigvals[i]
        eigvec = eigvecs[i]
        norm = norms[i]
        
        logging.info(f"Mode #{i+1}: frequency {eigval:.7f} cm-1; norm: {norm:.7f}")
       
        ra = [[0.0 for x in range(3)] for y in range(3)]
        for j in range(len(disps)):
            disp_filename = 'OUTCAR.%04d.%+d.out' % (i+1, disps[j])
            
            try:
                outcar_fh = open(disp_filename, 'r')
                logging.info(f"File {disp_filename} exists, parsing...")
            except IOError:
                if args['use_poscar']:
                    logging.info(f"File {disp_filename} not found, preparing displaced POSCAR")
                    poscar_fh = open('POSCAR', 'w')
                    poscar_fh.write("%s %4.1e \n" % (disp_filename, step_size))
                    poscar_fh.write(poscar_header)
                    poscar_fh.write("Cartesian\n")
                    
                    for k in range(nat):
                        pos_disp = [ pos[k][l] + eigvec[k][l]*step_size*disps[j]/norm for l in range(3)]
                        poscar_fh.write( '%15.10f %15.10f %15.10f\n' % (pos_disp[0], pos_disp[1], pos_disp[2]) )
                        
                    poscar_fh.close()
                else:
                    logging.info("Using provided POSCAR")
                
                if args['gen']: # only generate POSCARs
                    poscar_fn = 'POSCAR.%+d.out' % disps[j]
                    move('POSCAR', poscar_fn)
                    logging.info("'-gen' mode -> {poscar_fn} with displaced atoms have been generated")
                    
                    if j+1 == len(disps): # last iteration for the current displacements list
                        logging.info("'-gen' mode -> POSCAR files with displaced atoms have been generated, exiting now")
                        sys.exit(0)
                else: # run VASP here
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
                        move('OUTCAR', disp_filename)
                    except IOError:
                        logging.error("ERROR Couldn't find OUTCAR file, exiting...")
                        sys.exit(1)
                    
                    outcar_fh = open(disp_filename, 'r')
            
            try:
                eps = get_epsilon_from_OUTCAR(outcar_fh)
                outcar_fh.close()
            except Exception as err:
                logging.error(f"{err}")
                logging.error(f"Moving {disp_filename} back to 'OUTCAR' and exiting...")
                move(disp_filename, 'OUTCAR')
                sys.exit(1)
            
            for m in range(3):
                for n in range(3):
                    ra[m][n]   += eps[m][n] * coeffs[j]/step_size * norm * vol/(4.0*pi)
            #units: A^2/amu^1/2 =         dimless   * 1/A         * 1/amu^1/2  * A^3
        
        alpha = (ra[0][0] + ra[1][1] + ra[2][2])/3.0
        beta2 = ( (ra[0][0] - ra[1][1])**2 + (ra[0][0] - ra[2][2])**2 + (ra[1][1] - ra[2][2])**2 + 6.0 * (ra[0][1]**2 + ra[0][2]**2 + ra[1][2]**2) )/2.0
        logging.info(f"! {i+1:4d}  freq: {eigval:10.5f}  alpha: {alpha:10.7f}  beta2: {beta2:10.7f}  activity: {45.0*alpha**2 + 7.0*beta2:10.7f}")
        output_fh.write(f"{i+1:03d}  {eigval:10.5f}  {alpha:10.7f}  {beta2:10.7f}  {45.0*alpha**2 + 7.0*beta2:10.7f}\n")
        output_fh.flush()
    
    output_fh.close()