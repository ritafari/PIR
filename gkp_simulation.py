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
    """
    Apply shift errors in position and momentum
    """
    # Position shift is multiplication by exp(i*p̂*shift_q)
    psi_shifted = psi * np.exp(1j * x * shift_p)
    
    # Momentum shift is convolution with exp(i*q̂*shift_p)
    # For discrete x, this becomes a Fourier shift
    psi_shifted = np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.fftshift(psi_shifted)) * np.exp(1j * np.fft.fftfreq(len(x), x[1]-x[0]) * shift_q)))
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
    # Measure q mod sqrt(pi)
    q_values = x[np.abs(psi)**2 > 0.1*np.max(np.abs(psi)**2)]      # Find positions where the wavefunction has significant probability
    q_syndromes = q_values % np.sqrt(np.pi)                        # Compute those positions modulo √π
    
    # Measure p mod sqrt(pi) via Fourier transform
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi)))      # Fourier transform to momentum space
    p = np.fft.fftfreq(len(x), x[1]-x[0]) * 2*np.pi                # Get corresponding momentum values
    p_values = p[np.abs(psi_p)**2 > 0.1*np.max(np.abs(psi_p)**2)]  # Find peaks in momentum space
    p_syndromes = p_values % np.sqrt(np.pi)                        # Compute those positions modulo √π
    
    return np.mean(q_syndromes), np.mean(p_syndromes)

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


# Step 5 - Main function to run the simulation
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
plot_gkp_wavefunction(x, psi_corr)

# Calculate fidelity
fidelity = np.abs(np.sum(psi.conj() * psi_corr))**2
print(f"Recovery fidelity: {fidelity:.4f}")

# Generate and display animation
gkp_correction_animation(psi, x, q_syn, p_syn)
