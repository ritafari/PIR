# 16/04's Work

## Fidelity Troubleshooting
_pb: low fidelity_

**1 -  Incorrect Application of Error Shifts**
I apply a shift using both shift_q and shift_p using apply_shift_error, but then when correcting, you apply a second shift again based on syndrome estimates. This assumes that syndromes are perfectly accurate, which isn’t true especially for finite-energy GKP states (with nonzero delta).
Moreover, there's potential double application of correction (you apply it twice in gkp_correct, check line duplication).
[DONE]

**2 - Imposter Syndrome Extraction**
The way we compute the syndrome is ```q_mean = ⟨q⟩, then q_syndrome = q_mean % √π```
This does not necessarily give you the actual shift. For example, a shift of -0.3 and +√π - 0.3 will give the same modulo result.
➡️ Consider using maximum-likelihood or adaptive rounding to nearest lattice point instead of modulo. 
That modulo-based "rounding" can misidentify shifts if the initial displacement is more than ±√π/2.
Update version of the ```gkp_syndrome_measurement(psi, x)```:
```def gkp_syndrome_measurement_improved(psi, x):
    """
    Improved syndrome estimation using rounding to nearest lattice point.
    """
    dx = x[1] - x[0]
    N = len(x)

    # ⟨q⟩
    q_mean = np.sum(x * np.abs(psi)**2) * dx
    q_nearest = np.round(q_mean / np.sqrt(np.pi)) * np.sqrt(np.pi)
    q_syndrome = q_mean - q_nearest

    # ⟨p⟩
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2 * np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    dp = p[1] - p[0]
    p_mean = np.sum(p * np.abs(psi_p)**2) * dp
    p_nearest = np.round(p_mean / np.sqrt(np.pi)) * np.sqrt(np.pi)
    p_syndrome = p_mean - p_nearest

    return q_syndrome, p_syndrome
```
[DONE]

**3 - Finite Grid Issues**
➡️ Increase the number of points in x (e.g. 2048 instead of 1000) and double-check that all Gaussians in your GKP state are well within range and resolved.
After Checking, if we replace ```for n in range(-10, 11):``` by 
```n_max = int((x[-1] - 3*delta) / (2 * np.sqrt(np.pi)))
for n in range(-n_max, n_max + 1):
```
then Applied shift_q: 0.4 and Measured p_syndrome: 0.4000 instead of 0.3905 !!!🥳
[DONE]

**4 - Missing Normalization Checks After Correction**
After applying the shifts the state can become slightly non-normalized due to numerical noise => Always re-normalize the wavefunction after applying errors or corrections:
```psi /= np.linalg.norm(psi) * np.sqrt(dx)``` as shown in updated version of the ```gkp_correct(psi, x, q_syndrome, p_syndrome)```:
```def gkp_correct_improved(psi, x, q_syndrome, p_syndrome):
    """
    Apply correction using improved syndrome values.
    """
    psi_corr = apply_shift_error(psi, -q_syndrome, -p_syndrome, x)
    dx = x[1] - x[0]
    psi_corr /= np.sqrt(np.sum(np.abs(psi_corr)**2) * dx)  # Normalize
    return psi_corr
```
[DONE]


## Are the Graphs Relevant? 

<u>✅ Position-Space Graph</u>
They let you visually verify wether the GKP peaks are aligned with the lattice grid and wether correction worked

<u>❌ Missing Momentum View</u>
I should plot the momentum space (p), momentum errors are corrected just like position ones, and without seeing the momentum distribution, you're blind to half the errors. 
Add this plot:
```def plot_momentum_wavefunction(psi, x):
    N = len(x)
    dx = x[1] - x[0]
    p = np.fft.fftshift(np.fft.fftfreq(N, d=dx)) * 2 * np.pi
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2*np.pi)
    
    plt.plot(p, np.abs(psi_p)**2)
    plt.xlabel('Momentum (p)')
    plt.ylabel('Probability density')
    plt.title('Momentum-space GKP state')
    plt.show()```
And call it: 
```plot_momentum_wavefunction(psi)
plot_momentum_wavefunction(psi_err)
plot_momentum_wavefunction(psi_corr)
```
[DONE]

<u>❌ Update the Animation Function</u>
```gkp_correction_animation()``` is visually great, but it’s tied to the original (modulo-based) syndrome extraction, hence we have to update it. 
What should be changed:
• Replace ```q_syndrome```and ```p_syndrome```input with the improved calculation internally. 
• Use ```apply_shift_error``` and ```gkp_correct```consistently.
• Show all three: original, errored, and corrected stats dynamically.
• Add fidelity annotation during animation.

[DONE]


<u>✅ Fidelity vs. Applied Shift Plot</u>
A new function I can add to analyze how the correction performs for varying shift errors:
```def fidelity_vs_shift_plot(psi, x, delta, shift_range=(-0.6, 0.6), steps=30):
    """
    Compute and plot fidelity vs. various shift errors.
    """
    dx = x[1] - x[0]
    shift_vals = np.linspace(*shift_range, steps)
    fidelity_map = np.zeros((steps, steps))

    for i, dq in enumerate(shift_vals):
        for j, dp in enumerate(shift_vals):
            psi_err = apply_shift_error(psi, dq, dp, x)
            q_syn, p_syn = gkp_syndrome_measurement_improved(psi_err, x)
            psi_corr = gkp_correct_improved(psi_err, x, q_syn, p_syn)
            fidelity_map[i, j] = compute_fidelity(psi, psi_corr, dx)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(fidelity_map, extent=(shift_range[0], shift_range[1], shift_range[0], shift_range[1]),
               origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='Fidelity')
    plt.xlabel('Position shift Δq')
    plt.ylabel('Momentum shift Δp')
    plt.title('Fidelity vs Shift Errors')
    plt.grid(False)
    plt.show()```
Call it with ```fidelity_vs_shift_plot(psi, x, delta)```
[DONE]


<u>❌ Change the graph names sso that it isn't confusing</u>

1. Display lattice before applying the error shift
        plot_initial_position(psi, x)
        plot_initial_momentum(psi, x)
2. Display lattice after applying the error shift
        plot_error_position(psi_err, x)
        plot_error_momentum(psi_err, x)
3. Display lattice after correcting the error shift
        plot_corrected_position(psi_corr, x)
        plot_corrected_momentum(psi_corr, x)
[DONE]

### Relevancy of each implemented plots:
• Position/Momentum Plots: Show the state's structure and how errors affect it
• Fidelity vs Shift Plot: Demonstrates the correction capability across different error magnitudes
• Animation: Visualizes the correction process dynamically


## Updated Workflow
After implementing all of the above, the simulation should follow this updated structure:
```delta = 0.2  # Squeezing parameter
shift_q, shift_p = 0.3, 0.4  # Random errors to apply

# Generate GKP state
x, psi = gkp_state(delta)
plot_initial_position(psi, x)
plot_initial_momentum(psi, x)

# Apply errors
psi_err = apply_shift_error(psi, shift_q, shift_p, x)
plot_error_position(psi_err, x)
plot_error_momentum(psi_err, x)

# Measure syndrome
q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
print(f"Measured syndromes - q: {q_syn:.3f}, p: {p_syn:.3f}")

# Correct errors
psi_corr = gkp_correct(psi_err, x, q_syn, p_syn)
plot_corrected_position(psi_corr, x)
plot_corrected_momentum(psi_corr, x)

# Calculate fidelity
fidelity = compute_fidelity(psi, psi_corr, x[1]-x[0])
print(f"Fidelity after correction: {fidelity:.4f}")

# Generate and display animation
gkp_correction_animation(psi, x, q_syn, p_syn)
fidelity_vs_shift_plot(psi, x, delta)

# Print the original and shifted states: check if they match to know effectiveness of the code
print(f"Applied shift_q: {shift_q}, Measured q_syndrome: {q_syn}")
print(f"Applied shift_p: {shift_p}, Measured p_syndrome: {p_syn}")
```



## Other Possible Adjustments for Boosting Fidelity

1. Lower delta — 0.2 is okay, but try 0.1 or even 0.05 if you want better protection (more computationally expensive though). 
[DONE: doesn't change shit...and fidelity gets worst!]❌

2. Try smaller error shifts (0.3 and 0.4 are getting close to the correctable threshold of GKP codes).
-> Modified error shifts to 0.1 and 0.2, fidelity went from 0.011 to 0.6!! 🥳
[DONE]

3. Implement rounding to nearest GKP lattice point, not just taking modulo.

4. Use logical GKP states like |0_L⟩ or |+_L⟩ (by adjusting which lattice sites you use) to test logical preservation.



## Remaining issues so far:
1. Fidelity is low - For small shifts (0.1, 0.2) we should be able to expect higher fidelity (>0.9) with proper implementation. 
2. Syndrome measurement mismatch - issue in how position shifts are being measured and applied.
3. Animation function correctness - try to understand it cause i'm pretty sure it's fucked ... 

<u>Dealing with the Syndrome Measurement Mismatch</u>
The momentum measurement works because FFTs handle periodic boundary conditions naturally. Position measurement doesn't account for the GKP lattice periodicity properly. The current implementation issues would be that the direct expectation value calculation doesn't respect the √π-periodic nature of GKP states => The modulo operation needs to be applied differently for position measurements. 

### gkp_syndrome_measurement
**current** ```gkp_syndrome_measurement```code:
``` dx = x[1] - x[0]
    N = len(x)

    # Use modulo sqrt(pi) to find the fractional shift
    # This better captures the periodic nature of GKP states
    q_shifts = np.mod(x + np.sqrt(np.pi)/2, np.sqrt(np.pi)) - np.sqrt(np.pi)/2
    q_syndrome = np.sum(q_shifts * np.abs(psi)**2) * dx

    # Similarly for momentum
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2 * np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * np.pi
    dp = p[1] - p[0]
    p_shifts = np.mod(p + np.sqrt(np.pi)/2, np.sqrt(np.pi)) - np.sqrt(np.pi)/2
    p_syndrome = np.sum(p_shifts * np.abs(psi_p)**2) * dp

    return q_syndrome, p_syndrome
```
**Revised** version
```dx = x[1] - x[0]
    N = len(x)
    
    # POSITION (q) measurement
    # Calculate the fractional part of position relative to √π lattice
    q_shifts = (x + np.sqrt(np.pi)/2) % np.sqrt(np.pi) - np.sqrt(np.pi)/2
    q_syndrome = np.sum(q_shifts * np.abs(psi)**2) * dx
    
    # Similarly for momentum
    psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi))) * dx / np.sqrt(2 * np.pi)
    p = np.fft.fftshift(np.fft.fftfreq(len(x), d=dx)) * 2 * np.pi
    dp = p[1] - p[0]
    p_shifts = np.mod(p + np.sqrt(np.pi)/2, np.sqrt(np.pi)) - np.sqrt(np.pi)/2
    p_syndrome = np.sum(p_shifts * np.abs(psi_p)**2) * dp
    
    return q_syndrome, p_syndrome
```

### Test Shift Function
```def test_shift_measurement(delta=0.2, test_shifts=[0.1, 0.3, 0.5]):
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
```

**Expected outcome**
Shift Measurement Verification:
-----------------------------
Applied Shift | Measured q_syndrome | Measured p_syndrome
       0.100 |               0.100 |              0.100
       0.300 |               0.300 |              0.300
       0.500 |               0.500 |              0.500

**My Outcome**
Shift Measurement Verification:
-----------------------------
Applied Shift | Measured q_syndrome | Measured p_syndrome
       0.100 |              -0.100 |              0.084
       0.300 |              -0.300 |              0.274
       0.500 |              -0.495 |              0.425

=> Fix the sign q in ```gkp_syndrome_measurement```
instead of ```q_syndrome = np.sum(q_shifts * np.abs(psi)**2) * dx```
we do ```q_syndrome = -np.sum(q_shifts * np.abs(psi)**2) * dx  # Note negative sign```

=> Fix the sign in ```apply_shift_error```
instead of ```psi_shifted = psi * np.exp(1j * x * shift_p)```
we do ```psi_shifted = psi * np.exp(-1j * x * shift_p)  # Negative sign for momentum```




Eclaircire 
Processeurs analogiques 

