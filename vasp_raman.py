import re
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def MAT_m_VEC(m, v):
    p = [0.0 for _ in range(len(v))]
    for i in range(len(m)):
        assert len(v) == len(m[i]), f'Length of matrix row {i} does not match vector length'
        p[i] = sum(m[i][j] * v[j] for j in range(len(v)))
    return p

def T(m):
    return [[m[i][j] for i in range(len(m[j]))] for j in range(len(m))]

def parse_poscar(poscar_fh):
    poscar_fh.seek(0)
    lines = poscar_fh.readlines()
    scale = float(lines[1])
    
    if scale < 0.0:
        log.error("Negative scale not implemented.")
        sys.exit(1)
    
    b = [[float(x) * scale for x in lines[i].split()[:3]] for i in range(2, 5)]
    
    vol = (
        b[0][0] * b[1][1] * b[2][2] + b[1][0] * b[2][1] * b[0][2] + b[2][0] * b[0][1] * b[1][2]
        - b[0][2] * b[1][1] * b[2][0] - b[2][1] * b[1][2] * b[0][0] - b[2][2] * b[0][1] * b[1][0]
    )
    
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
    is_scaled = lines[line_at][0].lower() not in ('c', 'k')
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
        log.error("There should be exactly four parameters")
        sys.exit(1)
    return int(tmp[0]), int(tmp[1]), int(tmp[2]), float(tmp[3])

if __name__ == '__main__':
    import os
    from datetime import datetime
    from shutil import move
    import optparse
    
    log.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    parser = optparse.OptionParser()
    parser.add_option('-g', '--gen', help='Generate POSCAR only', action='store_true')
    parser.add_option('-u', '--use_poscar', help='Use provided POSCAR in the folder, USE WITH CAUTION!!', action='store_true')
    (options, args) = parser.parse_args()
    
    VASP_RAMAN_RUN = os.environ.get('VASP_RAMAN_RUN')
    if VASP_RAMAN_RUN is None:
        log.error("Set environment variable 'VASP_RAMAN_RUN'")
        parser.print_help()
        sys.exit(1)
    log.info(f"VASP_RAMAN_RUN='{VASP_RAMAN_RUN}'")
    
    VASP_RAMAN_PARAMS = os.environ.get('VASP_RAMAN_PARAMS')
    if VASP_RAMAN_PARAMS is None:
        log.error("Set environment variable 'VASP_RAMAN_PARAMS'")
        parser.print_help()
        sys.exit(1)
    log.info(f"VASP_RAMAN_PARAMS='{VASP_RAMAN_PARAMS}'")
    
    first, last, nderiv, step_size = parse_env_params(VASP_RAMAN_PARAMS)
    assert first >= 1, "First mode should be equal or larger than 1"
    assert last >= first, "Last mode should be equal or larger than first mode"
    if options.gen:
        assert last == first, "'-gen' mode -> only generation for one mode makes sense"
    assert nderiv == 2, "At this time, nderiv = 2 is the only supported"
    
    try:
        with open('POSCAR.phon', 'r') as poscar_fh:
            nat, vol, b, pos, poscar_header = parse_poscar(poscar_fh)
    except IOError:
        log.error("Couldn't open input file POSCAR.phon, exiting...")
        sys.exit(1)
    
    log.debug(f"Number of atoms: {nat}, Volume: {vol}")
    
    output_fh = open('vasp_raman.dat', 'w')
    output_fh.write("# mode    freq(cm-1)    alpha    beta2    activity\n")
    
    for i in range(first - 1, last):
        eigval, eigvec, norm = 0.0, [], 0.0  # Placeholder for calculations
        log.info(f"Mode #{i+1}: frequency {eigval:.7f} cm-1; norm: {norm:.7f}")
        
        alpha, beta2, activity = 0.0, 0.0, 0.0  # Placeholder for calculations
        log.info(f"! {i+1:4d}  freq: {eigval:10.5f}  alpha: {alpha:10.7f}  beta2: {beta2:10.7f}  activity: {activity:10.7f}")
        output_fh.write(f"{i+1:03d}  {eigval:10.5f}  {alpha:10.7f}  {beta2:10.7f}  {activity:10.7f}\n")
        output_fh.flush()
    
    output_fh.close()
    log.info("Calculation complete.")
