import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.special import hermite
from scipy.integrate import quad
from matplotlib.animation import FuncAnimation


# Step 1 - Generating the GKP state
def gkp_state(delta, N_cutoff=50):
    """
    Generate finite-energy GKP state.
    delta: squeezing parameter (smaller = better protection)
    N_cutoff: Fock state truncation
    """
    # Position basis
    x = np.linspace(-10, 10, 1000)
    
    # Ideal GKP state is a sum of delta functions
    # Finite-energy version uses Gaussian peaks
    psi = np.zeros_like(x, dtype=np.complex128)
    for n in range(-10, 11):
        psi += np.exp(-(x - 2*n*np.sqrt(np.pi))**2/(2*delta**2))
    
    # Normalize: The integral of |ψ|² over all space must be 1!!
    norm = np.sqrt(quad(lambda x_val: np.interp(x_val, x, np.abs(psi)**2), -10, 10)[0])
    psi /= norm
    
    return x, psi

def plot_gkp_wavefunction(x, psi):
    plt.plot(x, np.abs(psi)**2) # Shows the probability density
    plt.xlabel('Position (q)')
    plt.ylabel('Probability density')
    plt.title('Finite-energy GKP state')
    plt.show()


# Step 2 - Simulating the GKP errors
# Common Errors for GKP codes are small shifts in position or momentum
def apply_shift_error(psi, shift_q, shift_p, x):
    N = len(x)
    dx = x[1] - x[0]
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi

    # Apply momentum shift (in position space)
    psi_shifted = psi * np.exp(1j * x * shift_p)

    # Apply position shift (in momentum space)
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi_shifted)))
    psi_p *= np.exp(1j * p * shift_q)
    psi_shifted = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(psi_p)))

    return psi_shifted

def plot_shifted_wavefunction(x, psi_shifted):
    plt.plot(x, np.abs(psi_shifted)**2) # Shows the probability density
    plt.xlabel('Position (q)')
    plt.ylabel('Probability density')
    plt.title('Shifted GKP state')
    plt.show()


# Step 3 - Simulating the GKP error correction
def gkp_syndrome_measurement(psi, x):
    """
    Measure the syndrome (shift from lattice)
    """
    dx = x[1] - x[0]
    N = len(x)

    # Position expectation ⟨q⟩
    q_mean = np.sum(x * np.abs(psi)**2) * dx
    q_syndrome = q_mean % np.sqrt(np.pi)

    # Fourier transform to momentum space (properly normalized)
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2 * np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi

    dp = p[1] - p[0]
    p_mean = np.sum(p * np.abs(psi_p)**2) * dp
    p_syndrome = p_mean % np.sqrt(np.pi)

    return q_syndrome, p_syndrome

def plot_syndrome_wavefunction(x, psi, q_syndrome, p_syndrome):
    plt.plot(x, np.abs(psi)**2) # Shows the probability density
    plt.axvline(q_syndrome, color='r', linestyle='--', label='q_syndrome')
    plt.axvline(p_syndrome, color='g', linestyle='--', label='p_syndrome')
    plt.xlabel('Position (q)')
    plt.ylabel('Probability density')
    plt.title('Syndrome Measurement')
    plt.legend()
    plt.show()

def gkp_correct(psi, x, q_syndrome, p_syndrome):
    """
    Apply correction based on syndrome
    """
    # Correct q shift
    if q_syndrome > np.sqrt(np.pi)/2:
        q_corr = q_syndrome - np.sqrt(np.pi)
    else:
        q_corr = q_syndrome
    
    # Correct p shift
    if p_syndrome > np.sqrt(np.pi)/2:
        p_corr = p_syndrome - np.sqrt(np.pi)
    else:
        p_corr = p_syndrome
    
    return apply_shift_error(psi, -q_corr, -p_corr, x)



# Step 4 - Animating the error correction process
def gkp_correction_animation(psi, x, q_syndrome, p_syndrome, save_path="gkp_correction.gif"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Initialize plots
    line_orig, = ax1.plot(x, np.abs(psi)**2, 'b-', label='Original')
    psi_shifted = apply_shift_error(psi, q_syndrome, p_syndrome, x)
    line_shifted, = ax1.plot(x, np.abs(psi_shifted)**2, 'r--', label='Shifted')
    line_corrected, = ax1.plot(x, np.abs(psi)**2, 'g:', label='Corrected', linewidth=2)
    
    # Draw GKP grid
    for n in range(-4, 5):
        ax1.axvline(n*np.sqrt(np.pi), color='gray', linestyle=':', alpha=0.3)
    
    ax1.set_xlim(x[0], x[-1])
    ax1.set_ylim(0, 1.1*np.max(np.abs(psi)**2))
    ax1.set_xlabel('Position (q)')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    
    # Text annotations
    text_syndrome = ax2.text(0.5, 0.7, "", fontsize=12, ha='center', transform=ax2.transAxes)
    text_correction = ax2.text(0.5, 0.5, "", fontsize=12, ha='center', transform=ax2.transAxes)
    ax2.axis('off')
    
    # Calculate correction amounts
    q_corr = q_syndrome - np.sqrt(np.pi) if q_syndrome > np.sqrt(np.pi)/2 else q_syndrome
    p_corr = p_syndrome - np.sqrt(np.pi) if p_syndrome > np.sqrt(np.pi)/2 else p_syndrome
    
    def update(frame):
        alpha = min(1.0, frame/30)  # Normalized progress 0→1
        
        # Apply partial correction
        psi_temp = apply_shift_error(psi_shifted, -alpha*q_corr, -alpha*p_corr, x)
        line_corrected.set_ydata(np.abs(psi_temp)**2)
        
        # Update text
        if frame < 15:
            text_syndrome.set_text(f"Syndrome Measurement:\nΔq = {q_syndrome:.2f}\nΔp = {p_syndrome:.2f}")
            text_correction.set_text("")
        else:
            text_syndrome.set_text(f"Measured Syndromes:\nΔq = {q_syndrome:.2f}\nΔp = {p_syndrome:.2f}")
            text_correction.set_text(f"Applying Correction:\nΔq = {-q_corr:.2f}\nΔp = {-p_corr:.2f}")
        
        return line_corrected, text_syndrome, text_correction
    
    ani = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
    
    # Save as GIF
    ani.save(save_path, writer='pillow', fps=20)
    plt.close()
    print(f"Animation saved to {save_path}")



# Step 5 - Calculate fidelity
def compute_fidelity(psi1, psi2, dx):
    return np.abs(np.sum(np.conj(psi1) * psi2) * dx)**2



# Step 6 - Main function to run the simulation
# Parameters
delta = 0.2  # Squeezing parameter
shift_q, shift_p = 0.3, 0.4  # Random errors to apply

# Generate GKP state
x, psi = gkp_state(delta)
plot_gkp_wavefunction(x, psi)

# Apply errors
psi_err = apply_shift_error(psi, shift_q, shift_p, x)
plot_gkp_wavefunction(x, psi_err)

# Measure syndrome
q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
print(f"Measured syndromes - q: {q_syn:.3f}, p: {p_syn:.3f}")

# Correct errors
psi_corr = gkp_correct(psi_err, x, q_syn, p_syn)
plot_syndrome_wavefunction(x, psi, q_syn, p_syn)

# Calculate fidelity
psi_corr = gkp_correct(psi_err, x, q_syn, p_syn)
fidelity = compute_fidelity(psi, psi_corr, x[1]-x[0])
print(f"Fidelity after correction: {fidelity:.4f}")

# Generate and display animation
gkp_correction_animation(psi, x, q_syn, p_syn)

# Print the original and shifted states: check if they match to know effectiveness of the code
print(f"Applied shift_q: {shift_q}, Measured q_syndrome: {q_syn}")
print(f"Applied shift_p: {shift_p}, Measured p_syndrome: {p_syn}")

