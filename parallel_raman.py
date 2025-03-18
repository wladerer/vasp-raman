#!/usr/bin/env python3

import os
import re
import sys
import logging
import numpy as np
from shutil import rmtree
from pymatgen.io.vasp import Poscar, Potcar, Kpoints, Incar, Vasprun
from pymatgen.core import Structure
from argparse import ArgumentParser


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def read_modes_from_outcar(filename, nat):
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

def setup_mode_directory(mode_idx):
    """Creates and prepares a directory for a given mode."""
    mode_dir = f"mode_{mode_idx+1:03d}"
    if os.path.exists(mode_dir):
        rmtree(mode_dir)
    os.makedirs(mode_dir)
    return mode_dir

def write_vasp_files(mode_dir, potcar, incar, kpoints):
    """Writes VASP input files to the specified directory."""
    potcar.write_file(f"{mode_dir}/POTCAR")
    incar.write_file(f"{mode_dir}/INCAR")
    kpoints.write_file(f"{mode_dir}/KPOINTS")

def generate_displaced_structures(structure, eigvec, norm, step_size, displacements):
    """Generates displaced structures based on eigenvectors."""
    displaced_structures = {}
    for disp in displacements:
        displaced_structure = structure.copy()
        for i, site in enumerate(displaced_structure.sites):
            displacement = np.array(eigvec[i]) * step_size * disp / norm
            displaced_structure.translate_sites(i, displacement, frac_coords=False)
        displaced_structures[disp] = displaced_structure
    return displaced_structures

def generate_displaced_poscars(poscar_file, outcar_file, step_size, displacements, incar_file, potcar_file, kpoints_file):
    """Generates displaced POSCAR files in separate directories."""
    structure = Poscar.from_file(poscar_file).structure
    nat = len(structure)
    eigvals, eigvecs, norms = read_modes_from_outcar(outcar_file, nat)
    
    potcar = Potcar.from_file(potcar_file)
    incar = Incar.from_file(incar_file)
    kpoints = Kpoints.from_file(kpoints_file)
    
    for mode_idx, (eigval, eigvec, norm) in enumerate(zip(eigvals, eigvecs, norms)):
        mode_dir = setup_mode_directory(mode_idx)
        
        displaced_structures = generate_displaced_structures(structure, eigvec, norm, step_size, displacements)
        for disp, displaced_structure in displaced_structures.items():
            direction = "forward" if disp > 0 else "backward"
            disp_dir = os.path.join(mode_dir, direction)
            if not os.path.exists(disp_dir):
                os.makedirs(disp_dir, exist_ok=True)
            filename = f"{disp_dir}/POSCAR"
            Poscar(displaced_structure).write_file(filename)
            log.info(f"Generated {filename}")
            
            # Write VASP input files to the forward and backward directories
            write_vasp_files(disp_dir, potcar, incar, kpoints)

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
        epsilon = vasprun.epsilon_static
        return epsilon
    except Exception as e:
        log.error(f"Error reading dielectric tensor from vasprun.xml: {e}")
        raise RuntimeError("Couldn't find dielectric tensor in OUTCAR or vasprun.xml")


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

def collect_results(directory: str, output: str = "raman.dat"):
    """Collects results from the generated directories."""
    nat = Structure.from_file("POSCAR.phon").num_sites
    eigvals, eigvecs, norms = read_modes_from_outcar("OUTCAR.phon", nat)
    disps = [-1, 1]  # Assuming displacements are -1 and 1 as default

    results = {}
    with open('raman.dat', 'w') as outfile:
        outfile.write("# mode    freq(cm-1)    alpha    beta2    activity\n")
        for mode_idx, (eigval, eigvec, norm) in enumerate(zip(eigvals, eigvecs, norms)):
            log.info("Mode #%i: frequency %10.7f cm-1; norm: %10.7f" % (mode_idx + 1, eigval, norm))
            ra = np.zeros((3, 3))
            for disp in disps:
                disp_dir = os.path.join(directory, f"mode_{mode_idx+1:03d}", "forward" if disp > 0 else "backward")
                disp_filename = os.path.join(disp_dir, "vasprun.xml")
                try:
                    vasprun = Vasprun(disp_filename)
                    eps = vasprun.epsilon_static
                    log.info(f"Found existing epsilon in {disp_filename}: {eps}")
                except Exception as e:
                    log.error(f"Error reading vasprun.xml from {disp_dir}: {e}")
                    continue
                
                coeff = disp / norm
                ra += np.array(eps) * coeff / 0.01  # Assuming step_size is 0.01

            alpha, beta2 = compute_polarizability(ra)
            activity = 45.0 * alpha**2 + 7.0 * beta2
            log.info(f"Mode {mode_idx+1:4d}  freq: {eigval:10.5f}  alpha: {alpha:10.7f}  beta2: {beta2:10.7f}  activity: {activity:10.7f}")
            outfile.write(f"{mode_idx+1:03d}  {eigval:10.5f}  {alpha:10.7f}  {beta2:10.7f}  {activity:10.7f}\n")
            results[mode_idx + 1] = {"freq": eigval, "alpha": alpha, "beta2": beta2, "activity": activity}
    return results


def main():
    parser = ArgumentParser(description="Generate displaced POSCAR files from OUTCAR.phon.")
    file_group = parser.add_argument_group('files', 'Input files')
    file_group.add_argument("-p", "--poscar", default="POSCAR.phon", help="Path to POSCAR.phon file")
    file_group.add_argument("-O", "--outcar", default="OUTCAR.phon", help="Path to OUTCAR.phon file")
    file_group.add_argument("-I", "--incar", default="INCAR", help="Path to INCAR file")
    file_group.add_argument("-P", "--potcar", default="POTCAR", help="Path to POTCAR file")
    file_group.add_argument("-K", "--kpoints", default="KPOINTS", help="Path to KPOINTS file")

    disp_group = parser.add_argument_group('displacements', 'Displacement parameters')
    disp_group.add_argument("-s", "--step", type=float, default=0.01, help="Displacement step size")
    disp_group.add_argument("-d", "--disps", type=int, nargs="+", default=[-1, 1], help="Displacements to apply")

    collect_group = parser.add_argument_group('collect', 'Collect results')
    collect_group.add_argument("-c", "--collect", action="store_true", help="Collect results from generated directories")
    collect_group.add_argument("-o", "--output", default="raman.dat", help="Output file for results")
    collect_group.add_argument("--directory", default=".", help="Directory to search")

    args = parser.parse_args()
    if args.collect:
        collect_results(args.directory, args.output)
    else:
        generate_displaced_poscars(args.poscar, args.outcar, args.step, args.disps, args.incar, args.potcar, args.kpoints)


if __name__ == "__main__":
    main()
