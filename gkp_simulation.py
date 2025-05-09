import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import hermite
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation
import warnings
from scipy.integrate import IntegrationWarning



# Step 1 - GKP STATE GENERATION (Continuous Variable Core)
def setup_figures():
    plt.close('all')  # Close any existing figures
    fig1 = plt.figure(1, figsize=(8, 6))
    fig2 = plt.figure(2, figsize=(8, 6))
    fig3 = plt.figure(3, figsize=(8, 6))
    fig4 = plt.figure(4, figsize=(8, 6))
    return fig1, fig2, fig3, fig4



# Step 2 - Generating the GKP state
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



# Step 3 - Simulating the GKP errors
# Common Errors for GKP codes are small shifts in position or momentum
def apply_shift_error(psi, shift_q, shift_p, x):
    N = len(x)
    dx = x[1] - x[0]
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi

    # Apply momentum shift (in position space) with proper sign
    psi_shifted = psi * np.exp(-1j * x * shift_p)   # Negative sign for momentum

    # Apply position shift (in momentum space)
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi_shifted)))
    psi_p *= np.exp(1j * p * shift_q)
    psi_shifted = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(psi_p)))

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



# Step 4 - Simulating the GKP error correction
def gkp_syndrome_measurement(psi, x):
    """
    Measure the syndrome (shift from lattice)
    """
    dx = x[1] - x[0]
    N = len(x)

    # POSITION (q) measurement
    # Calculate the fractional part of position relative to √π lattice
    q_shifts = (x + np.sqrt(np.pi)/2) % np.sqrt(np.pi) - np.sqrt(np.pi)/2
    q_syndrome = -np.sum(q_shifts * np.abs(psi)**2) * dx  # Note negative sign (otherwise it would be a shift in the opposite direction)
    
    # MOMENTUM (p) measurement - improved precision
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2*np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    dp = p[1] - p[0]
    
    # Use complex phase for more accurate momentum measurement
    theta_p = np.angle(np.sum(psi_p * np.exp(-1j*p*np.sqrt(np.pi)/2)) * dp)
    p_syndrome = (theta_p + np.pi) % (2*np.pi) - np.pi  # Wrapped to [-π, π]
    p_syndrome *= np.sqrt(np.pi)/np.pi  # Scale to GKP lattice
    
    return q_syndrome, p_syndrome

def gkp_correct(psi, x, q_syndrome, p_syndrome, delta):
    """
    Apply correction based on syndrome
    """
    # Apply correction with proper boundary conditions
    psi_corr = apply_shift_error(psi, -q_syndrome, -p_syndrome, x)
    
    # Additional stabilization step - project to nearest logical state
    dx = x[1] - x[0]
    q_shifts = np.mod(x + np.sqrt(np.pi)/2, np.sqrt(np.pi)) - np.sqrt(np.pi)/2
    psi_corr *= np.exp(-q_shifts**2 / (2*delta**2))  # Soft projection
    
    # Normalize
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



# Step 5 - Animating the error correction process
def gkp_correction_animation(psi, x, q_syndrome, p_syndrome, shift_q, shift_p, delta, save_path="gkp_correction.gif"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    dx = x[1] - x[0]
    
    # Step 1: Apply known error
    psi_err = apply_shift_error(psi, shift_q, shift_p, x)

    # Step 2: Measure syndrome (redundant since we pass them, but keeps logic)
    q_syndrome, p_syndrome = gkp_syndrome_measurement(psi_err, x)

    # Step 3: Compute correction WITH delta parameter
    psi_corr_target = gkp_correct(psi_err, x, q_syndrome, p_syndrome, delta)

    # Fidelity at the end
    # Removed unused variable final_fidelity

    # Step 4: Set up plots
    ax1.plot(x, np.abs(psi)**2, 'b-', label='Original')
    ax1.plot(x, np.abs(psi_err)**2, 'r--', label='Errored')
    line_corr, = ax1.plot(x, np.abs(psi_err)**2, 'g:', label='Correcting...', linewidth=2)

    for n in range(-4, 5):
        ax1.axvline(n*np.sqrt(np.pi), color='gray', linestyle=':', alpha=0.3)  # Correct usage of axvline and linestyle

    ax1.set_xlim(x[0], x[-1])  # Correct usage of set_xlim
    ax1.set_ylim(0, 1.1*np.max(np.abs(psi)**2))  # Correct usage of set_ylim
    ax1.set_xlabel('Position (q)')  # Correct usage of set_xlabel
    ax1.set_ylabel('Probability Density')  # Correct usage of set_ylabel
    ax1.legend()

    # Text box
    text_info = ax2.text(0.5, 0.6, "", fontsize=12, ha='center', transform=ax2.transAxes)  # Correct usage of fontsize
    text_fidelity = ax2.text(0.5, 0.4, "", fontsize=12, ha='center', transform=ax2.transAxes)  # Correct usage of fontsize
    ax2.axis('off')

    # Correction values
    q_corr = q_syndrome
    p_corr = p_syndrome

    def update(frame):
        alpha = min(1.0, frame / 30)
        
        # Apply correction in both quadratures simultaneously
        psi_step = apply_shift_error(psi_err, -alpha * q_syndrome, -alpha * p_syndrome, x)
        
        # Calculate intermediate fidelity
        current_fidelity = compute_fidelity(psi, psi_step, dx)
        
        # Update plots
        line_corr.set_ydata(np.abs(psi_step)**2)
        text_info.set_text(
            f"Applied Shifts: Δq={shift_q:.3f}, Δp={shift_p:.3f}\n"
            f"Measured Syndromes: Δq={q_syndrome:.3f}, Δp={p_syndrome:.3f}\n"
            f"Correction Progress: {alpha*100:.1f}%"
        )
        text_fidelity.set_text(f"Current Fidelity: {current_fidelity:.4f}")
        
        return line_corr, text_info, text_fidelity

    ani = FuncAnimation(fig, update, frames=60, interval=50, blit=True)  # Correct usage of blit
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f"Animation saved to {save_path}")



# Step 6 - Calculate fidelity
def compute_fidelity(psi1, psi2, dx):
    overlap = np.abs(np.sum(np.conj(psi1) * psi2) * dx)**2
    # Also compute logical fidelity by projecting to nearest peak
    return overlap

def logical_fidelity(psi, x, delta):
    dx = x[1] - x[0]
    # Project to nearest logical state
    logical_psi = np.zeros_like(psi)
    for n in range(-5, 6):
        peak = np.exp(-(x - 2*n*np.sqrt(np.pi))**2/(2*delta**2))
        peak /= np.sqrt(np.sum(np.abs(peak)**2) * dx)
        overlap = np.abs(np.sum(np.conj(peak) * psi) * dx)**2
        logical_psi += overlap * peak
    return np.abs(np.sum(np.conj(psi) * logical_psi) * dx)**2

def fidelity_vs_shift_plot(psi, x, delta, shift_range=(-0.6, 0.6), steps=30):
    """
    Compute and plot fidelity vs. various shift errors.
    Now properly includes delta parameter.
    """
    dx = x[1] - x[0]
    shift_vals = np.linspace(*shift_range, steps)
    fidelity_map = np.zeros((steps, steps))

    for i, dq in enumerate(shift_vals):
        for j, dp in enumerate(shift_vals):
            psi_err = apply_shift_error(psi, dq, dp, x)
            q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
            psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)  # Now passing delta
            fidelity_map[i, j] = compute_fidelity(psi, psi_corr, dx)

    # Plotting code remains the same...
    plt.figure(figsize=(8, 6))
    plt.imshow(fidelity_map, extent=(shift_range[0], shift_range[1], shift_range[0], shift_range[1]),
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Position shift Δq')
    plt.ylabel('Momentum shift Δp')
    plt.title('Fidelity vs Shift Errors')
    plt.grid(False)
    plt.show()



# Step 6.5 - Plotting the threshold for maximum correctable shift
# Threshold for max correctable shift
def find_threshold(psi, x, delta, fidelity_threshold=0.99, max_shift=0.5, steps=100):
    """
    Find the maximum correctable shift for a given GKP state.
    
    Parameters:
    - psi: The GKP state wavefunction (from gkp_state())
    - x: The position space array
    - delta: squeezing parameter (used for correction)
    - fidelity_threshold: fidelity value below which we consider correction unsuccessful
    - max_shift: maximum shift to test (will search from 0 to this value)
    - steps: number of steps in the search
    
    Returns:
    - q_threshold: maximum correctable position shift
    - p_threshold: maximum correctable momentum shift
    """
    dx = x[1] - x[0]
    
    # Test position shifts
    q_shifts = np.linspace(0, max_shift, steps)
    q_fidelities = np.zeros_like(q_shifts)
    
    for i, shift in enumerate(q_shifts):
        psi_err = apply_shift_error(psi, shift, 0, x)  # Only position shift
        q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
        psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)
        q_fidelities[i] = compute_fidelity(psi, psi_corr, dx)
    
    # Find where fidelity drops below threshold
    q_threshold_idx = np.argmax(q_fidelities < fidelity_threshold)   # finds the first index where fidelity drops below our threshold( 0 if 1st elem < threshold; len(q_fidelities) if all fidelities > threshold)
    
    # Special Case: ALL tested shifts (up to max_shift) have fidelity ≥ threshold
    if q_threshold_idx == 0 and q_fidelities[0] >= fidelity_threshold:
        q_threshold = max_shift  # All tested shifts are below threshold
    
    # Normal Case
    else:
        q_threshold = q_shifts[q_threshold_idx - 1] if q_threshold_idx > 0 else 0
        # If we found a drop point (q_threshold_idx > 0), we take the shift just before the dro
        # otherwise means even the smallest shift failed, so threshold is 0
    
    # Test momentum shifts
    p_shifts = np.linspace(0, max_shift, steps)
    p_fidelities = np.zeros_like(p_shifts)
    
    for i, shift in enumerate(p_shifts):
        psi_err = apply_shift_error(psi, 0, shift, x)  # Only momentum shift
        q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
        psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)
        p_fidelities[i] = compute_fidelity(psi, psi_corr, dx)
    
    # Find where fidelity drops below threshold
    p_threshold_idx = np.argmax(p_fidelities < fidelity_threshold)
    if p_threshold_idx == 0 and p_fidelities[0] >= fidelity_threshold:
        p_threshold = max_shift  # All tested shifts are below threshold
    else:
        p_threshold = p_shifts[p_threshold_idx - 1] if p_threshold_idx > 0 else 0
    
    return q_threshold, p_threshold

def plot_threshold_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=20):
    """
    Plot the threshold shift vs delta parameter using the same GKP state.
    """
    deltas = np.linspace(*delta_range, steps)
    q_thresholds = np.zeros_like(deltas)
    p_thresholds = np.zeros_like(deltas)
    
    for i, delta in enumerate(deltas):
        q_thresh, p_thresh = find_threshold(psi, x, delta)
        q_thresholds[i] = q_thresh
        p_thresholds[i] = p_thresh
    
    plt.figure(figsize=(10, 6))
    plt.plot(deltas, q_thresholds, 'b-', label='Position shift threshold')  # How much shift in position can be corrected.
    plt.plot(deltas, p_thresholds, 'r-', label='Momentum shift threshold')  # How much shift in momentum can be corrected.
    plt.xlabel('Delta (squeezing parameter)')   # X-axis: Delta (squeezing parameter, typically ranging from ~0.1 to 0.5)
    plt.ylabel('Maximum correctable shift')     # Y-axis: Maximum correctable shift (threshold)
    plt.title('Error correction threshold vs. delta')
    plt.legend()
    plt.grid(True)
    plt.show()



# Step 7 - Performance Analysis
def performance_analysis(psi, x, delta, num_trials=1000):
    """
    Analyze average correction performance with random shifts within threshold limits.
    Returns: (avg_fidelity, q_threshold, p_threshold)
    """
    # First determine the maximum correctable shifts
    q_thresh, p_thresh = find_threshold(psi, x, delta)
    
    # Initialize statistics
    fidelities = []
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
    
    # Calculate average fidelity
    avg_fidelity = np.mean(fidelities)
    
    return avg_fidelity, q_thresh, p_thresh

def plot_performance_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=10, num_trials=500):
    """
    Plot average performance metrics vs delta parameter
    """
    deltas = np.linspace(*delta_range, steps)
    avg_fidelities = []
    thresholds_q = []
    thresholds_p = []
    
    for delta in deltas:
        print(f"Analyzing delta={delta:.2f}...")
        fid, q_thresh, p_thresh = performance_analysis(psi, x, delta, num_trials)
        avg_fidelities.append(fid)
        thresholds_q.append(q_thresh)
        thresholds_p.append(p_thresh)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot fidelity
    ax1.plot(deltas, avg_fidelities, 'b-o')
    ax1.set_xlabel('Delta (squeezing parameter)')
    ax1.set_ylabel('Average Fidelity', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.set_title('Average Fidelity vs Delta')
    
    # Plot thresholds
    ax2.plot(deltas, thresholds_q, 'r-', label='Position Threshold')
    ax2.plot(deltas, thresholds_p, 'g-', label='Momentum Threshold')
    ax2.set_xlabel('Delta (squeezing parameter)')
    ax2.set_ylabel('Threshold Shift')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Correctable Shift Thresholds vs Delta')
    
    plt.tight_layout()
    return fig



# Step 8 - Main function to run the simulation
def main():
    # Parameters
    delta = 0.2 # Squeezing parameter
    shift_q, shift_p = 0.1, 0.2  # Random errors to apply

    # Setup all figures first
    fig1, fig2, fig3, fig4 = setup_figures()
    fig5, fig6 = plt.figure(5), plt.figure(6)

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

    # Calculate fidelity
    fidelity = compute_fidelity(psi, psi_corr, x[1]-x[0])
    print(f"Fidelity after correction: {fidelity:.4f}")

    # Generate and display animation
    gkp_correction_animation(psi, x, q_syn, p_syn, shift_q, shift_p, delta)
    fidelity_vs_shift_plot(psi, x, delta)

    # Print the original and shifted states: check if they match to know effectiveness of the code
    print(f"Applied shift_q: {shift_q}, Measured q_syndrome: {q_syn}")
    print(f"Applied shift_p: {shift_p}, Measured p_syndrome: {p_syn}")

    # Find and print threshold for current delta
    q_thresh, p_thresh = find_threshold(psi, x, delta)
    print(f"\nFor delta = {delta}:")
    print(f"Maximum correctable position shift: {q_thresh:.4f}")
    print(f"Maximum correctable momentum shift: {p_thresh:.4f}")
    
    # Plot threshold vs delta using the same psi
    plot_threshold_vs_delta(psi, x)

    # Run performance analysis
    print("\nRunning performance analysis...")
    avg_fidelity, q_thresh, p_thresh = performance_analysis(psi, x, delta)
    print(f"\nPerformance for delta={delta}:")
    print(f"Average fidelity: {avg_fidelity:.4f}")
    print(f"Max correctable shifts: q={q_thresh:.3f}, p={p_thresh:.3f}")
    
    # Plot performance vs delta
    fig7 = plot_performance_vs_delta(psi, x)

    # Show all plots at once
    plt.show()

if __name__ == "__main__":
    main()
