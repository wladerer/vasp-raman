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

#!/usr/bin/env python3

import os
import sys
import logging
import numpy as np
from shutil import rmtree
from pymatgen.io.vasp import Poscar
from argparse import ArgumentParser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

def read_modes_from_outcar(outcar_file, nat):
    """
    Extracts eigenvalues and eigenvectors from an OUTCAR.phon file.

    Parameters:
    - outcar_file (str): Path to the OUTCAR.phon file.
    - nat (int): Number of atoms in the structure.

    Returns:
    - tuple: (eigvals, eigvecs, norms) where each is a list.
    """
    eigvals, eigvecs, norms = [], [], []
    try:
        with open(outcar_file, "r") as f:
            lines = f.readlines()
        
        found = False
        for i, line in enumerate(lines):
            if "Eigenvectors after division by SQRT(mass)" in line:
                found = True
                break

        if not found:
            log.error("Could not find eigenvectors in OUTCAR. Ensure 'NWRITE=3' is set in INCAR.")
            sys.exit(1)

        index = i + 4  # Skip headers
        for mode in range(nat * 3):
            index += 2  # Skip empty line + mode header
            freq_line = lines[index].strip()
            parts = freq_line.split()
            eigvals.append(float(parts[-2]))  # Extract frequency in cm^-1
            index += 2  # Skip column headers

            eigvec = []
            for _ in range(nat):
                values = list(map(float, lines[index].split()[3:6]))  # Read x, y, z displacements
                eigvec.append(values)
                index += 1
            
            eigvecs.append(eigvec)
            norms.append(np.linalg.norm(eigvec))

        return eigvals, eigvecs, norms
    except Exception as e:
        log.error(f"Error reading OUTCAR.phon: {e}")
        sys.exit(1)

def generate_displaced_poscars(poscar_file, outcar_file, step_size, displacements):
    """
    Generates displaced POSCAR files in separate directories.

    Parameters:
    - poscar_file (str): Path to the original POSCAR.phon.
    - outcar_file (str): Path to the OUTCAR.phon.
    - step_size (float): Step size for displacement.
    - displacements (list): List of displacement values (e.g., [-1, 1]).
    """
    # Read structure from POSCAR
    structure = Poscar.from_file(poscar_file).structure
    nat = len(structure)

    # Read modes from OUTCAR
    eigvals, eigvecs, norms = read_modes_from_outcar(outcar_file, nat)

    for mode_idx, (eigval, eigvec, norm) in enumerate(zip(eigvals, eigvecs, norms)):
        mode_dir = f"mode_{mode_idx+1:03d}"
        
        # Remove existing directory and recreate
        if os.path.exists(mode_dir):
            rmtree(mode_dir)
        os.makedirs(mode_dir)

        for disp in displacements:
            displaced_structure = structure.copy()
            for i, site in enumerate(displaced_structure.sites):
                displacement = np.array(eigvec[i]) * step_size * disp / norm
                displaced_structure.translate_sites(i, displacement, frac_coords=False)

            # Write new POSCAR
            filename = f"{mode_dir}/POSCAR.{disp:+d}"
            Poscar(displaced_structure).write_file(filename)
            log.info(f"Wrote {filename}")

def main():
    parser = ArgumentParser(description="Generate displaced POSCAR files from OUTCAR.phon.")
    parser.add_argument("-p", "--poscar", default="POSCAR.phon", help="Path to POSCAR.phon file")
    parser.add_argument("-o", "--outcar", default="OUTCAR.phon", help="Path to OUTCAR.phon file")
    parser.add_argument("-I", "--incar", default="INCAR", help="Path to INCAR file to be used in VASP calculations")
    parser.add_argument("-p", "--potcar", default="POTCAR", help="Path to POTCAR file to be used in VASP calculations")
    parser.add_argument("-k", "--kpoints", default="KPOINTS", help="Path to KPOINTS file to be used in VASP calculations")
    parser.add_argument("-s", "--step", type=float, default=0.01, help="Displacement step size")
    parser.add_argument("-d", "--disps", type=int, nargs="+", default=[-1, 1], help="Displacements to apply")
    args = parser.parse_args()

    generate_displaced_poscars(args.poscar, args.outcar, args.step, args.disps)

if __name__ == "__main__":
    main()


