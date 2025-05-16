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
    dx = x[1] - x[0]
    N = len(x)
    
    # Position measurement (working correctly)
    q_shifts = (x + np.sqrt(np.pi)/2) % np.sqrt(np.pi) - np.sqrt(np.pi)/2
    q_syndrome = -np.sum(q_shifts * np.abs(psi)**2) * dx
    
    # FIXED Momentum measurement
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2*np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    
    # Proper phase measurement at p=√π
    phase = np.angle(np.sum(psi_p * np.exp(-1j*p*np.sqrt(np.pi))))
    p_syndrome = phase / np.sqrt(np.pi)  # Convert to displacement units
    
    # Ensure syndrome is within [-√π/2, √π/2]
    p_syndrome = (p_syndrome + np.sqrt(np.pi)/2) % np.sqrt(np.pi) - np.sqrt(np.pi)/2
    
    return q_syndrome, p_syndrome

def validate_syndrome(q_syn, p_syn, psi, x, delta):
    """Advanced version that considers actual correctable range"""
    q_thresh, p_thresh = find_threshold(psi, x, delta)
    return abs(q_syn) < q_thresh and abs(p_syn) < p_thresh

def gkp_correct(psi, x, q_syn, p_syn, delta):
    """More robust correction"""
    # Apply shift correction
    dx = x[1] - x[0]
    psi_corr = apply_shift_error(psi, -q_syn, -p_syn, x)
    
    # Soft projection with adaptive strength
    q_shifts = (x + np.sqrt(np.pi)/2) % np.sqrt(np.pi) - np.sqrt(np.pi)/2
    projection_strength = min(1.0, 0.5/delta)  # Adaptive based on delta
    psi_corr *= np.exp(-q_shifts**2 / (2 * (delta*projection_strength)**2))
    
    # Normalize carefully
    norm = np.sqrt(np.sum(np.abs(psi_corr)**2 * dx))
    return psi_corr / norm if norm > 1e-10 else psi

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
    """Guaranteed to return values between 0 and 1"""
    overlap = np.abs(np.sum(np.conj(psi1) * psi2) * dx)
    return overlap**2  # Squaring ensures proper normalization

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

def fidelity_vs_shift_plot(psi, x, delta, shift_range=(-0.5, 0.5), steps=30):
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
    plt.imshow(fidelity_map, vmin=0.0, vmax=1.0,  cmap='viridis')
    plt.colorbar(label='Fidelity (0-1)')
    plt.xlabel('Position shift Δq')
    plt.ylabel('Momentum shift Δp')
    plt.title('Fidelity vs Shift Errors')
    plt.grid(False)
    plt.show()



# Step 6.5 - Plotting the threshold for maximum correctable shift
# Threshold for max correctable shift
def find_threshold(psi, x, delta, fidelity_threshold=0.9, max_shift=0.5, steps=50):
    """More reliable threshold detection"""
    dx = x[1] - x[0]
    
    def test_shifts(shift_type):
        shifts = np.linspace(0, max_shift, steps)
        last_good = 0.0
        
        for shift in shifts:
            # Apply shift
            if shift_type == 'q':
                psi_err = apply_shift_error(psi, shift, 0, x)
            else:
                psi_err = apply_shift_error(psi, 0, shift, x)
            
            # Measure and correct
            q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
            psi_corr = gkp_correct(psi_err, x, q_syn, p_syn, delta)
            
            # Compute fidelity
            fid = compute_fidelity(psi, psi_corr, dx)
            
            if fid >= fidelity_threshold:
                last_good = shift
            else:
                break
                
        return last_good
    
    q_thresh = test_shifts('q')
    p_thresh = test_shifts('p')
    return q_thresh, p_thresh

def plot_threshold_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=20, max_shift=0.5):
    """
    Plot the threshold shift vs delta parameter using the same GKP state.
    
    Parameters:
    - psi: Initial GKP state wavefunction
    - x: Position space array
    - delta_range: Range of delta values to test (min, max)
    - steps: Number of delta values to test
    - max_shift: Maximum shift to test for threshold finding
    """
    deltas = np.linspace(*delta_range, steps)
    q_thresholds = np.zeros_like(deltas)
    p_thresholds = np.zeros_like(deltas)
    
    print("Calculating thresholds...")
    print(f"{'Delta':<8} {'Q Threshold':<12} {'P Threshold':<12}")
    print("-"*35)
    
    for i, delta in enumerate(deltas):
        # Generate fresh GKP state for each delta to ensure consistency
        _, psi_current = gkp_state(delta)
        
        q_thresh, p_thresh = find_threshold(psi_current, x, delta, max_shift=max_shift)
        q_thresholds[i] = q_thresh
        p_thresholds[i] = p_thresh
        
        print(f"{delta:.4f}   {q_thresh:.6f}    {p_thresh:.6f}")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Theoretical maximum (0.5√π ≈ 0.886)
    theoretical_max = 0.5 * np.sqrt(np.pi)
    plt.axhline(y=theoretical_max, color='gray', linestyle='--', 
                label='Theoretical max (0.5√π)')
    
    # Actual thresholds
    plt.plot(deltas, q_thresholds, 'b-o', label='Position shift threshold', markersize=5)
    plt.plot(deltas, p_thresholds, 'r-s', label='Momentum shift threshold', markersize=5)
    
    plt.xlabel('Delta (squeezing parameter)')
    plt.ylabel('Maximum correctable shift')
    plt.title('Error Correction Threshold vs. Delta\n(Testing shifts up to {:.2f})'.format(max_shift))
    
    # Adjust y-axis to show interesting range
    y_max = min(1.1 * theoretical_max, 1.1 * max(max(q_thresholds), max(p_thresholds)))
    plt.ylim(0, y_max)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text annotation if thresholds are zero
    if np.all(q_thresholds == 0) or np.all(p_thresholds == 0):
        plt.text(0.5, 0.5, 'WARNING: All thresholds zero\nCheck syndrome measurement!',
                 ha='center', va='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='red', alpha=0.2))
    
    plt.tight_layout()
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
    delta = 0.2  # Squeezing parameter
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

    # Correct errors with ADAPTIVE DELTA and VALIDATION
    print("\n--- Correction Process ---")

    # Measure syndrome
    q_syn1, p_syn1 = gkp_syndrome_measurement(psi_err, x)
    print(f"Measured syndromes - q: {q_syn1:.3f}, p: {p_syn1:.3f}")

    # Correct errors
    # Applyinig the three correctors one after the other to better fidelity
    psi_corr1 = gkp_correct(psi_err, x, q_syn1, p_syn1, delta) #1st correction
    q_syn2, p_syn2 = gkp_syndrome_measurement(psi_corr1, x)
    print(f"Measured syndromes - q: {q_syn2:.3f}, p: {p_syn2:.3f}")
     # 2nd correction (medium) - only if syndromes are valid
    if validate_syndrome(q_syn2, p_syn2, psi_corr1, x, delta):
        psi_corr2 = gkp_correct(psi_corr1, x, q_syn2, p_syn2, delta=0.1)
        q_syn3, p_syn3 = gkp_syndrome_measurement(psi_corr2, x)
        print(f"After 2nd correction: syndromes q={q_syn3:.3f}, p={p_syn3:.3f}")
    else:
        print("Skipping 2nd correction - invalid syndromes")
        psi_corr2 = psi_corr1
        q_syn3, p_syn3 = q_syn2, p_syn2
    
    # 3rd correction (fine) - only if syndromes are valid
    if validate_syndrome(q_syn3, p_syn3, psi_corr2, x, delta):
        psi_corr3 = gkp_correct(psi_corr2, x, q_syn3, p_syn3, delta=0.05)
    else:
        print("Skipping 3rd correction - invalid syndromes")
        psi_corr3 = psi_corr2

    # Plot corrected states
    plot_corrected_position(psi_corr1, x, fig5) # 1st correction
    plot_corrected_momentum(psi_corr1, x, fig6)
    plot_corrected_position(psi_corr2, x, fig5) # 2nd correction
    plot_corrected_momentum(psi_corr2, x, fig6)
    plot_corrected_position(psi_corr3, x, fig5) # 3rd correction
    plot_corrected_momentum(psi_corr3, x, fig6)

    # Calculate fidelity
    fidelity = compute_fidelity(psi, psi_corr1, x[1]-x[0]) # Fidelity after 1st correction
    print(f"Fidelity after correction: {fidelity:.4f}")
    F = compute_fidelity(psi, psi_corr2, x[1]-x[0]) # Fidelity after 2nd correction
    print(f"Fidelity after 2nd correction: {F:.4f}")
    F1 = compute_fidelity(psi, psi_corr3, x[1]-x[0]) # Fidelity after 3rd correction
    print(f"Fidelity after 3rd correction: {F1:.4f}")

    # Generate and display animation
    gkp_correction_animation(psi, x, q_syn1, p_syn1, shift_q, shift_p, delta)
    fidelity_vs_shift_plot(psi, x, delta)

    # Print the original and shifted states: check if they match to know effectiveness of the code
    print(f"Applied shift_q: {shift_q}, Measured q_syndrome: {q_syn1}")
    print(f"Applied shift_p: {shift_p}, Measured p_syndrome: {p_syn1}")

    # Are the Syndromes Being Updated Correctly
    print(f"1st correction syndromes: q={q_syn1:.3f}, p={p_syn1:.3f}")
    print(f"2nd correction syndromes: q={q_syn2:.3f}, p={p_syn2:.3f}") 
    print(f"3rd correction syndromes: q={q_syn3:.3f}, p={p_syn3:.3f}")


    # Find and print threshold for current delta
    q_thresh, p_thresh = find_threshold(psi, x, delta)
    print(f"\nFor delta = {delta}:")
    print(f"Maximum correctable position shift: {q_thresh:.4f}")
    print(f"Maximum correctable momentum shift: {p_thresh:.4f}")
    
    # Plot threshold vs delta using the same psi
    plot_threshold_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=20, max_shift=0.5)

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
