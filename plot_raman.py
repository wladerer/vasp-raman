import numpy as np
import matplotlib.pyplot as plt
import argparse


def broaden(hw, ab, gam=0.001, type='gauss'):
    fmin = min(hw)
    fmax = max(hw)
    erange = np.arange(fmin - 40 * gam, fmax + 40 * gam, gam / 10)
    spectrum = np.zeros_like(erange)
    
    for i in range(len(hw)):
        if type == 'gaussian':
            spectrum += (2 * np.pi) ** (-0.5) / gam * np.exp(np.clip(-1.0 * (hw[i] - erange) ** 2 / (2 * gam ** 2), -300, 300))
        elif type == 'lorentz':
            spectrum += ab[i] * 1 / np.pi * gam / ((hw[i] - erange) ** 2 + gam ** 2)
    
    return erange, spectrum

def inverse_cm_to_eV(cm: float) -> float:
    return cm * 0.00012398

def plot_spectrum(filename: str, broadening: float, broadening_type: str, units: str):
    hw = np.genfromtxt(filename, dtype=float)
    cm1 = hw[:, 1]
    int1 = hw[:, 4]
    int1 /= np.max(np.abs(int1), axis=0)
    
    if units == 'eV':
        cm1 = inverse_cm_to_eV(cm1)
    
    Es1, Spectrum1 = broaden(cm1, int1, broadening, broadening_type)
    
    plt.plot(Es1, Spectrum1)
    plt.xlabel('Frequency ({})'.format('eV' if units == 'eV' else 'cm-1'))
    plt.ylabel('Intensity')
    plt.title('Simulated Raman Spectrum')
    plt.grid(True)

    if args.output:
        plt.savefig(args.output)
    else:   
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Raman spectrum')
    parser.add_argument('filename', type=str, help='Filename with Raman data')
    parser.add_argument('-b', '--broadening', type=float, default=15.0, help='Broadening parameter')
    parser.add_argument('-t', '--type', type=str, default='lorentz', help='Broadening type', choices=['gauss', 'lorentz'])
    parser.add_argument('-o', '--output', type=str, help='Output filename')
    parser.add_argument('-u', '--units', type=str, default='inverse', help='Units for frequency', choices=['inverse', 'eV'])

    args = parser.parse_args()
    plot_spectrum(args.filename, args.broadening, args.type, args.units)
