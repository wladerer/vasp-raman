#!/usr/bin/env python3

import logging
import argparse
import os
import re
import sys
from math import pi
from shutil import move

from pymatgen.io.vasp import Poscar, Vasprun
import numpy as np

logging.basicConfig(filename='raman.log', filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def compute_polarizability(ra):
    """
    Compute isotropic polarizability (alpha) and anisotropy (beta²)
    from a 3x3 polarizability tensor using NumPy.
    """
    ra = np.array(ra)  # Ensure input is a NumPy array

    # Compute Alpha (Isotropic Polarizability)
    alpha = np.trace(ra) / 3.0

    # Compute Beta² (Anisotropy)
    diag_elements = np.diag(ra)  # Extract diagonal elements: [A_xx, A_yy, A_zz]
    off_diag_elements = ra[np.triu_indices(3, k=1)]  # Extract upper triangle off-diagonal elements: [A_xy, A_xz, A_yz]

    # Compute diagonal differences squared
    diag_diffs = np.array([
        diag_elements[0] - diag_elements[1], 
        diag_elements[0] - diag_elements[2], 
        diag_elements[1] - diag_elements[2]
    ])

    beta2 = (np.linalg.norm(diag_diffs) ** 2 + 6.0 * np.linalg.norm(off_diag_elements) ** 2) / 2.0

    return alpha, beta2


def write_displaced_poscar(structure, eigvec, step_size, disp, norm, filename) -> Poscar:
    """
    Writes a displaced POSCAR file using pymatgen.
    
    Parameters:
    - structure (Structure): Pymatgen structure object.
    - eigvec (list): Eigenvector displacements for atoms.
    - step_size (float): Displacement step size.
    - disp (int): Displacement direction (-1 or 1).
    - norm (float): Normalization factor for displacement.
    - filename (str): Output POSCAR filename.
    """
    # Displace atoms
    displaced_structure = structure.copy()
    for i, site in enumerate(displaced_structure.sites):
        displacement = np.array(eigvec[i]) * step_size * disp / norm
        displaced_structure.translate_sites(i, displacement, frac_coords=False)

    # Write new POSCAR
    poscar = Poscar(displaced_structure)
    poscar.write_file(filename)
    logging.info(f"Wrote displaced POSCAR to {filename}\n{poscar}\n")

    return poscar

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
                    norms[i] = np.sqrt(sum([abs(x) ** 2 for sublist in eigvec for x in sublist]))
                
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
        structure = Poscar.from_file('POSCAR.phon').structure
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
    
    with open('raman.dat', 'w') as outfile:
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
                        eps = get_epsilon_from_OUTCAR(disp_filename)
                        log.info(f"Found existing epsilon: {eps}")
                except IOError:
                    if args['use_poscar'] is False:
                        print(f"File {disp_filename} not found, preparing displaced POSCAR")
                        displaced_poscar = write_displaced_poscar(structure, eigvec, step_size, disps[j], norm, "POSCAR")

                        if not os.path.exists('modes'):
                            os.makedirs('structures')
                        mode_dir = f'modes/mode_{i+1:04d}'
                        if not os.path.exists(mode_dir):
                            os.makedirs(mode_dir)
                        displaced_poscar.write_file(f'{mode_dir}/POSCAR.{disps[j]:+d}.out')
                        log.info(f"Displaced POSCAR has been archived as '{mode_dir}/POSCAR.{disps[j]:+d}.out'")

                    else: 
                        log.info("Running VASP...")
                        os.system(VASP_RAMAN_RUN)
                        try:
                            move('OUTCAR', disp_filename)
                        except IOError:
                            log.error("Couldn't find OUTCAR file, exiting...")
                            sys.exit(1)
                        
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
            
            alpha, beta2 = compute_polarizability(ra)

            log.info(f"Mode {i+1:4d}  freq: {eigval:10.5f}  alpha: {alpha:10.7f}  beta2: {beta2:10.7f}  activity: {45.0*alpha**2 + 7.0*beta2:10.7f}")
            outfile.write(f"{i+1:03d}  {eigval:10.5f}  {alpha:10.7f}  {beta2:10.7f}  {45.0*alpha**2 + 7.0*beta2:10.7f}\n")
