import re
import sys
from math import sqrt, pi
from shutil import move
import os
import datetime

def MAT_m_VEC(m, v):
    p = [0.0 for _ in range(len(v))]
    for i, row in enumerate(m):
        assert len(v) == len(row), 'Length of the matrix row is not equal to the length of the vector'
        p[i] = sum(row[j] * v[j] for j in range(len(v)))
    return p

def T(m):
    return [[m[i][j] for i in range(len(m[j]))] for j in range(len(m))]

def parse_poscar(poscar_fh):
    poscar_fh.seek(0)
    lines = poscar_fh.readlines()
    
    scale = float(lines[1])
    if scale < 0.0:
        print("[parse_poscar]: ERROR negative scale not implemented.")
        sys.exit(1)
    
    b = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    
    vol = (b[0][0] * b[1][1] * b[2][2] + b[1][0] * b[2][1] * b[0][2] + b[2][0] * b[0][1] * b[1][2] -
           b[0][2] * b[1][1] * b[2][0] - b[2][1] * b[1][2] * b[0][0] - b[2][2] * b[0][1] * b[1][0])
    
    try:
        num_atoms = list(map(int, lines[5].split()))
        line_at = 6
    except ValueError:
        symbols = lines[5].split()
        num_atoms = list(map(int, lines[6].split()))
        line_at = 7
    
    nat = sum(num_atoms)
    if lines[line_at][0].lower() == 's':
        line_at += 1
    
    is_scaled = lines[line_at][0].lower() not in ['c', 'k']
    line_at += 1
    
    positions = []
    for i in range(line_at, line_at + nat):
        pos = list(map(float, lines[i].split()[:3]))
        if is_scaled:
            pos = MAT_m_VEC(T(b), pos)
        positions.append(pos)
    
    poscar_header = ''.join(lines[1:line_at-1])
    return nat, vol, b, positions, poscar_header

def parse_env_params(params):
    tmp = params.strip().split('_')
    if len(tmp) != 4:
        print("[parse_env_params]: ERROR there should be exactly four parameters")
        sys.exit(1)
    return int(tmp[0]), int(tmp[1]), int(tmp[2]), float(tmp[3])

def parse_freqdat(freqdat_fh, nat):
    freqdat_fh.seek(0)
    return [float(freqdat_fh.readline().split()[0]) for _ in range(nat * 3)]

def parse_modesdat(modesdat_fh, nat):
    modesdat_fh.seek(0)
    eigvecs, norms = [], []
    for _ in range(nat * 3):
        eigvec = [list(map(float, modesdat_fh.readline().split()[:3])) for _ in range(nat)]
        modesdat_fh.readline()
        eigvecs.append(eigvec)
        norms.append(sqrt(sum(abs(x) ** 2 for sublist in eigvec for x in sublist)))
    return eigvecs, norms

def get_modes_from_OUTCAR(outcar_fh, nat):
    eigvals, eigvecs, norms = [0.0] * (nat * 3), [0.0] * (nat * 3), [0.0] * (nat * 3)
    outcar_fh.seek(0)
    
    for line in outcar_fh:
        if "Eigenvectors after division by SQRT(mass)" in line:
            [outcar_fh.readline() for _ in range(4)]
            for i in range(nat * 3):
                outcar_fh.readline()
                p = re.search(r'^\s*(\d+).+?([\.\d]+) cm-1', outcar_fh.readline())
                eigvals[i] = float(p.group(2))
                outcar_fh.readline()
                eigvecs[i] = [list(map(float, outcar_fh.readline().split()[3:6])) for _ in range(nat)]
                norms[i] = sqrt(sum(abs(x) ** 2 for sublist in eigvecs[i] for x in sublist))
            return eigvals, eigvecs, norms
    
    print("[get_modes_from_OUTCAR]: ERROR Couldn't find 'Eigenvectors after division by SQRT(mass)' in OUTCAR. Use 'NWRITE=3' in INCAR. Exiting...")
    sys.exit(1)

def get_epsilon_from_OUTCAR(outcar_fh):
    outcar_fh.seek(0)
    for line in outcar_fh:
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR" in line:
            outcar_fh.readline()
            return [list(map(float, outcar_fh.readline().split())) for _ in range(3)]
    raise RuntimeError("[get_epsilon_from_OUTCAR]: ERROR Couldn't find dielectric tensor in OUTCAR")

if __name__ == '__main__':
    print(f"    Started at: {datetime.datetime.now():%Y-%m-%d %H:%M}\n")
    
    VASP_RAMAN_RUN = os.environ.get('VASP_RAMAN_RUN')
    if VASP_RAMAN_RUN is None:
        print("[__main__]: ERROR Set environment variable 'VASP_RAMAN_RUN'\n")
        sys.exit(1)
    print(f"[__main__]: VASP_RAMAN_RUN='{VASP_RAMAN_RUN}'")
    
    VASP_RAMAN_PARAMS = os.environ.get('VASP_RAMAN_PARAMS')
    if VASP_RAMAN_PARAMS is None:
        print("[__main__]: ERROR Set environment variable 'VASP_RAMAN_PARAMS'\n")
        sys.exit(1)
    print(f"[__main__]: VASP_RAMAN_PARAMS='{VASP_RAMAN_PARAMS}'")
    
    first, last, nderiv, step_size = parse_env_params(VASP_RAMAN_PARAMS)
    assert first >= 1, '[__main__]: First mode should be equal or larger than 1'
    assert last >= first, '[__main__]: Last mode should be equal or larger than first mode'
    assert nderiv == 2, '[__main__]: At this time, nderiv = 2 is the only supported'

    print("[__main__]: Script execution completed.")
