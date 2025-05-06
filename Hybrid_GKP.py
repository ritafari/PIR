import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, fftfreq
from matplotlib.animation import FuncAnimation
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
import warnings
from scipy.integrate import quad, IntegrationWarning



# Step 1 - GKP STATE GENERATION (Continuous Variable Core)
def setup_figures():
    plt.close('all')  # Close any existing figures
    fig1 = plt.figure(1, figsize=(8, 6))
    fig2 = plt.figure(2, figsize=(8, 6))
    fig3 = plt.figure(3, figsize=(8, 6))
    fig4 = plt.figure(4, figsize=(8, 6))
    return fig1, fig2, fig3, fig4



# Step 2 - GKP STATE GENERATION (Continuous Variable Core)
def gkp_state(delta, N_cutoff=50):
    """
    Generate finite-energy GKP state.
    delta: squeezing parameter (smaller = better protection)
    N_cutoff: Fock state truncation
    """
    x = np.linspace(-10, 10, 2048)
    psi = np.zeros_like(x, dtype=np.complex128)
    
    # Generate peaks with proper normalization
    for n in range(-N_cutoff, N_cutoff + 1):
        peak_center = 2 * n * np.sqrt(np.pi)
        psi += np.exp(-(x - peak_center)**2/(2*delta**2))
    
    # Normalization with improved numerical integration
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=IntegrationWarning)
        norm = np.sqrt(quad(lambda x_val: np.interp(x_val, x, np.abs(psi)**2), 
                      -10, 10, 
                      limit=200)[0])  # Increased limit from 50 to 200
    
    psi /= norm
    return x, psi

def plot_initial_position(psi, x, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    plt.plot(x, np.abs(psi)**2)
    plt.title('Initial GKP State (Position)')
    plt.xlabel('q')
    plt.ylabel('|ψ(q)|²')
    plt.grid(True)

def plot_initial_momentum(psi, x, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    N = len(x)
    dx = x[1] - x[0]
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2*np.pi)
    plt.plot(p, np.abs(psi_p)**2)
    plt.title('Initial GKP State (Momentum)')
    plt.xlabel('p')
    plt.ylabel('|ψ(p)|²')
    plt.grid(True)



# Step 3 - ERROR APPLICATION (Hybrid Classical-Quantum)
# Common Errors for GKP codes are small shifts in position or momentum
def apply_shift_error(psi, shift_q, shift_p, x):
    """
    Applies position and momentum shifts to the GKP state.
    Implements the Weyl displacement operator exp(i(p̂Δq - q̂Δp))
    """
    N = len(x)
    dx = x[1] - x[0]
    p = fftshift(fftfreq(N, d=dx)) * 2 * np.pi  # Momentum basis

    # Momentum shift in position space
    psi_shifted = psi * np.exp(-1j * x * shift_p)  # Note: Negative sign convention

    # Position shift via Fourier transform
    psi_p = fftshift(fft(fftshift(psi_shifted)))
    psi_p *= np.exp(1j * p * shift_q)
    psi_shifted = fftshift(ifft(fftshift(psi_p)))

    return psi_shifted

def plot_error_position(psi_err, x, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    plt.plot(x, np.abs(psi_err)**2)
    plt.title('After Shift Error (Position)')
    plt.xlabel('q')
    plt.ylabel('|ψ(q)|²')
    plt.grid(True)

def plot_error_momentum(psi_err, x, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    N = len(x)
    dx = x[1] - x[0]
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi_err))) * dx / np.sqrt(2*np.pi)
    plt.plot(p, np.abs(psi_p)**2)
    plt.title('After Shift Error (Momentum)')
    plt.xlabel('p')
    plt.ylabel('|ψ(p)|²')
    plt.grid(True)

def test_shift_measurement(delta, test_shifts=[0.1, 0.3, 0.5]):
    """
    Test function to verify shift measurement accuracy
    """
    x = np.linspace(-10, 10, 2048)
    _, psi = gkp_state(delta)
    
    print("Shift Measurement Verification:")
    print("-----------------------------")
    print("Applied Shift | Measured q_syndrome | Measured p_syndrome")
    
    for shift in test_shifts:
        # Apply same shift to both quadratures
        psi_shifted = apply_shift_error(psi, shift, shift, x)
        
        # Measure syndromes
        q_syn, p_syn = gkp_syndrome_measurement(psi_shifted, x)
        
        print(f"{shift:12.3f} | {q_syn:19.3f} | {p_syn:18.3f}")



# Step 4 - SYNDROME MEASUREMENT (Quantum-Classical Interface)
def gkp_syndrome_measurement(psi, x):
    """
    Measure the syndrome (shift from lattice)
    Returns (q_syndrome, p_syndrome) in range [-√π/2, √π/2]
    """
    dx = x[1] - x[0]
    N = len(x)

    # POSITION (q) measurement
    # Calculate the fractional part of position relative to √π lattice
    q_shifts = (x + np.sqrt(np.pi)/2) % np.sqrt(np.pi) - np.sqrt(np.pi)/2
    q_syndrome = -np.sum(q_shifts * np.abs(psi)**2) * dx  # Note negative sign (otherwise it would be a shift in the opposite direction)
    
    # Momentum measurement via phase estimation
    psi_p = fftshift(fft(fftshift(psi))) * dx/np.sqrt(2*np.pi)
    p = fftshift(fftfreq(N, d=dx)) * 2 * np.pi
    theta_p = np.angle(np.sum(psi_p * np.exp(-1j*p*np.sqrt(np.pi)/2)) * (p[1]-p[0]))
    p_syndrome = ((theta_p + np.pi) % (2*np.pi) - np.pi) * (np.sqrt(np.pi)/np.pi)
    
    return q_syndrome, p_syndrome

# ERROR CORRECTION (Hybrid Implementation)
def gkp_correct(psi, x, q_syndrome, p_syndrome, delta):
    """
    Apply correction based on syndrome
    Combines continuous-variable correction with optional qubit-level logic.
    """
    # Continuous-variable correction
    psi_corr = apply_shift_error(psi, -q_syndrome, -p_syndrome, x)
    
    # Optional: Qubit-level stabilization (simulated)
    if delta < 0.3:  # Only for well-squeezed states
        qc = QuantumCircuit(1)
        if abs(q_syndrome) > np.sqrt(np.pi)/4:  # Large position error
            qc.x(0)  # Bit-flip correction
        if abs(p_syndrome) > np.sqrt(np.pi)/4:  # Large momentum error
            qc.z(0)  # Phase-flip correction
        # Note: This is a simplified qubit approximation
        
    # Normalization
    dx = x[1] - x[0]
    psi_corr /= np.sqrt(np.sum(np.abs(psi_corr)**2) * dx)
    return psi_corr

def plot_corrected_position(psi_corr, x, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    plt.plot(x, np.abs(psi_corr)**2)
    plt.title('After Correction (Position)')
    plt.xlabel('q')
    plt.ylabel('|ψ(q)|²')
    plt.grid(True)

def plot_corrected_momentum(psi_corr, x, fig=None):
    if fig is None:
        fig = plt.figure()
    plt.figure(fig.number)
    N = len(x)
    dx = x[1] - x[0]
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi_corr))) * dx / np.sqrt(2*np.pi)
    plt.plot(p, np.abs(psi_p)**2)
    plt.title('After Correction (Momentum)')
    plt.xlabel('p')
    plt.ylabel('|ψ(p)|²')
    plt.grid(True)



# Step 5 - VISUALIZATION & QISKIT INTEGRATION
def plot_hybrid_results(psi, x, psi_err, psi_corr, fig=None):
    """Visualize both continuous and discrete aspects"""
    fig = plt.figure(figsize=(12, 6))
    
    # Continuous space plots
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(x, np.abs(psi)**2, label='Original')
    ax1.plot(x, np.abs(psi_err)**2, 'r', label='With Errors')
    ax1.plot(x, np.abs(psi_corr)**2, 'g--', label='Corrected')
    ax1.set_title('Position Space')
    ax1.legend()
    
    # Create separate figure for Bloch sphere
    bloch_fig = plt.figure(figsize=(6, 6))
    qc = QuantumCircuit(1)
    state = Statevector.from_instruction(qc)
    plot_bloch_multivector(state)  
    
    return fig, bloch_fig



# Step 7 - Threshold Implementation
def compute_fidelity(psi1, psi2, dx):
    """Compute fidelity between two states"""
    overlap = np.abs(np.sum(np.conj(psi1) * psi2) * dx)
    return np.abs(overlap)**2

def find_hybrid_threshold(psi, x, delta, fidelity_threshold=0.99, max_shift=0.5, steps=100):
    """
    Find maximum correctable shift for hybrid GKP-qubit system.
    Returns: (q_threshold, p_threshold, q_qubit_rate, p_qubit_rate)
    """
    dx = x[1] - x[0]
    
    # Initialize results
    q_shifts = np.linspace(0, max_shift, steps)
    q_fidelities = np.zeros_like(q_shifts)
    q_qubit_used = np.zeros_like(q_shifts, dtype=bool)
    
    p_shifts = np.linspace(0, max_shift, steps)
    p_fidelities = np.zeros_like(p_shifts)
    p_qubit_used = np.zeros_like(p_shifts, dtype=bool)

    # Test position shifts
    for i, shift in enumerate(q_shifts):
        psi_err = apply_shift_error(psi, shift, 0, x)
        q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
        
        # Track if qubit correction was triggered
        qubit_correction = (abs(q_syn) > np.sqrt(np.pi)/4)
        q_qubit_used[i] = qubit_correction
        
        psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)
        q_fidelities[i] = compute_fidelity(psi, psi_corr, dx)

    # Test momentum shifts
    for i, shift in enumerate(p_shifts):
        psi_err = apply_shift_error(psi, 0, shift, x)
        q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
        
        qubit_correction = (abs(p_syn) > np.sqrt(np.pi)/4)
        p_qubit_used[i] = qubit_correction
        
        psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)
        p_fidelities[i] = compute_fidelity(psi, psi_corr, dx)

    # Calculate thresholds
    q_threshold_idx = np.argmax(q_fidelities < fidelity_threshold)
    if q_threshold_idx == 0 and q_fidelities[0] >= fidelity_threshold:
        q_threshold = max_shift
    else:
        q_threshold = q_shifts[q_threshold_idx - 1] if q_threshold_idx > 0 else 0
    
    p_threshold_idx = np.argmax(p_fidelities < fidelity_threshold)
    if p_threshold_idx == 0 and p_fidelities[0] >= fidelity_threshold:
        p_threshold = max_shift
    else:
        p_threshold = p_shifts[p_threshold_idx - 1] if p_threshold_idx > 0 else 0
    
    # Calculate qubit intervention rates within correctable range
    q_qubit_rate = np.mean(q_qubit_used[:q_threshold_idx]) if q_threshold_idx > 0 else 0
    p_qubit_rate = np.mean(p_qubit_used[:p_threshold_idx]) if p_threshold_idx > 0 else 0
    
    return q_threshold, p_threshold, q_qubit_rate, p_qubit_rate

def plot_hybrid_threshold_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=20):
    """
    Plot threshold behavior and qubit correction rates vs delta
    """
    deltas = np.linspace(*delta_range, steps)
    q_thresholds = np.zeros_like(deltas)
    p_thresholds = np.zeros_like(deltas)
    q_qubit_rates = np.zeros_like(deltas)
    p_qubit_rates = np.zeros_like(deltas)

    for i, delta in enumerate(deltas):
        q_thresh, p_thresh, q_rate, p_rate = find_hybrid_threshold(psi, x, delta)
        q_thresholds[i] = q_thresh
        p_thresholds[i] = p_thresh
        q_qubit_rates[i] = q_rate
        p_qubit_rates[i] = p_rate
    
    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot thresholds on primary axis
    ax1.plot(deltas, q_thresholds, 'b-', label='Position shift threshold')
    ax1.plot(deltas, p_thresholds, 'r-', label='Momentum shift threshold')
    ax1.set_xlabel('Delta (squeezing parameter)')
    ax1.set_ylabel('Maximum correctable shift')
    ax1.grid(True)
    
    # Plot qubit rates on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(deltas, q_qubit_rates, 'b--', label='Qubit X correction rate')
    ax2.plot(deltas, p_qubit_rates, 'r--', label='Qubit Z correction rate')
    ax2.set_ylabel('Qubit correction rate', rotation=270, labelpad=15)
    ax2.set_ylim(0, 1)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Hybrid GKP-Qubit Correction Thresholds')
    plt.tight_layout()
    return fig



# Step 8 - Perrformance Analysis
def performance_analysis(psi, x, delta, num_trials=1000):
    """
    Analyze average correction performance with random shifts within threshold limits.
    Returns: (avg_fidelity, avg_qubit_usage)
    """
    # First determine the maximum correctable shifts
    q_thresh, p_thresh, _, _ = find_hybrid_threshold(psi, x, delta)
    
    # Initialize statistics
    fidelities = []
    qubit_usages = []
    dx = x[1] - x[0]
    
    for _ in range(num_trials):
        # Generate random shifts within threshold bounds
        shift_q = np.random.uniform(-q_thresh, q_thresh)
        shift_p = np.random.uniform(-p_thresh, p_thresh)
        
        # Apply error and correct
        psi_err = apply_shift_error(psi, shift_q, shift_p, x)
        q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
        psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)
        
        # Track fidelity
        fidelity = compute_fidelity(psi, psi_corr, dx)
        fidelities.append(fidelity)
        
        # Track if qubit correction was used
        qubit_used = (abs(q_syn) > np.sqrt(np.pi)/4) or (abs(p_syn) > np.sqrt(np.pi)/4)
        qubit_usages.append(qubit_used)
    
    # Calculate statistics
    avg_fidelity = np.mean(fidelities)
    avg_qubit_usage = np.mean(qubit_usages)
    
    return avg_fidelity, avg_qubit_usage, q_thresh, p_thresh

def plot_performance_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=10, num_trials=500):
    """
    Plot average performance metrics vs delta parameter
    """
    deltas = np.linspace(*delta_range, steps)
    avg_fidelities = []
    qubit_rates = []
    thresholds_q = []
    thresholds_p = []
    
    for delta in deltas:
        print(f"Analyzing delta={delta:.2f}...")
        fid, qubit_rate, q_thresh, p_thresh = performance_analysis(psi, x, delta, num_trials)
        avg_fidelities.append(fid)
        qubit_rates.append(qubit_rate)
        thresholds_q.append(q_thresh)
        thresholds_p.append(p_thresh)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot fidelity and thresholds
    ax1.plot(deltas, avg_fidelities, 'b-o', label='Average Fidelity')
    ax1.set_xlabel('Delta (squeezing parameter)')
    ax1.set_ylabel('Fidelity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    ax1b = ax1.twinx()
    ax1b.plot(deltas, thresholds_q, 'r--', label='Position Threshold')
    ax1b.plot(deltas, thresholds_p, 'g--', label='Momentum Threshold')
    ax1b.set_ylabel('Threshold Shift', color='r')
    ax1b.tick_params(axis='y', labelcolor='r')
    
    # Plot qubit usage rates
    ax2.plot(deltas, qubit_rates, 'm-s')
    ax2.set_xlabel('Delta (squeezing parameter)')
    ax2.set_ylabel('Qubit Correction Rate', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax2.grid(True)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines1b, labels1b = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines1b, labels1 + labels1b, loc='upper right')
    
    plt.suptitle('Hybrid GKP Performance Analysis')
    plt.tight_layout()
    return fig



# Step 9 - Main function to run the simulation
def main():
    # Parameters
    delta = 0.2 # Squeezing parameter
    shift_q, shift_p = 0.1, 0.2  # Random errors to apply

    # Setup all figures first
    fig1, fig2, fig3, fig4 = setup_figures()
    fig5, fig6, fig7 = plt.figure(5), plt.figure(6), plt.figure(7)

    # Generate GKP state
    x, psi = gkp_state(delta)

    # Plot initial states
    plot_initial_position(psi, x, fig1)
    plot_initial_momentum(psi, x, fig2)

    # Apply errors
    psi_err = apply_shift_error(psi, shift_q, shift_p, x)
    plot_error_position(psi_err, x, fig3)
    plot_error_momentum(psi_err, x, fig4)
    test_shift_measurement(delta, test_shifts=[0.1, 0.3, 0.5])

    # Measure syndrome
    q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
    print(f"Measured syndromes - q: {q_syn:.3f}, p: {p_syn:.3f}")

    # Correct errors
    psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)

    # Plot corrected states
    plot_corrected_position(psi_corr, x, fig5)
    plot_corrected_momentum(psi_corr, x, fig6)

    # Generate and display animation
    fig, bloch_fig = plot_hybrid_results(psi, x, psi_err, psi_corr, fig7)
    
    # Calculate and display fidelity
    dx = x[1] - x[0]
    fidelity = compute_fidelity(psi, psi_corr, dx)
    print(f"Fidelity after correction: {fidelity:.4f}")

    # Print the original and shifted states
    print(f"Applied shift_q: {shift_q}, Measured q_syndrome: {q_syn}")
    print(f"Applied shift_p: {shift_p}, Measured p_syndrome: {p_syn}")

    # Run threshold analysis
    print("\nRunning threshold analysis...")
    q_thresh, p_thresh, q_rate, p_rate = find_hybrid_threshold(psi, x, delta)
    print(f"\nFor delta = {delta}:")
    print(f"Max correctable position shift: {q_thresh:.4f} (qubit X rate: {q_rate:.1%})")
    print(f"Max correctable momentum shift: {p_thresh:.4f} (qubit Z rate: {p_rate:.1%})")
    
    # Plot threshold vs delta
    fig8 = plot_hybrid_threshold_vs_delta(psi, x)

    # Run performance analysis
    print("\nRunning performance analysis...")
    avg_fidelity, qubit_rate, q_thresh, p_thresh = performance_analysis(psi, x, delta)
    print(f"\nPerformance for delta={delta}:")
    print(f"Average fidelity: {avg_fidelity:.4f}")
    print(f"Qubit correction rate: {qubit_rate:.1%}")
    print(f"Max correctable shifts: q={q_thresh:.3f}, p={p_thresh:.3f}")
    
    # Plot performance vs delta
    fig9 = plot_performance_vs_delta(psi, x)
    
    # Show all plots at once
    plt.show()

if __name__ == "__main__":
    main()