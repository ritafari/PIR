# PIR
Quantic Error Correcting Code (QECCs)

[blablabla]



***GKP***
_Step 1: Setting Up Your Environment_
After discussing with our teacher during the 14/04/2025 session, we were advised to use the qiskit library in order to simplify the simulation of our QECCs.
Hence, first, make sure you have the necessary packages installed:
```pip install qiskit qiskit-aer numpy matplotli```
I preferred not to use qiskit in mine because i had grepping errors in my folder... If you want to avoid using Qiskit altogether, we can simulate GKP states and their error correction using just NumPy and SciPy. Install requirements:
```pip install numpy scipy matplotlib```



_Step 2: Simulating GKP States in Qiskit_
While qiskit does not have built-in GKP states, we can approximate them by superposing Gaussian peaks:
```def gkp_state(d, delta, logical_zero)```
Here d is the number of peaks to include in the superposition, delta is the width of each Gaussian peak and logical_zero, if True, returns   |0> state, else |1> state.
    > Remember that for the |0> state we have a comb of squeezed states centered at integer multiple sqrt(n.pi). For |1> state we get the same comb bu shifted by sqrt(n.pi)/2

In this step, the variable "psi" represents the quantum wavefunction of the GKP state in the position basis.
1 - **Mathematically**:
For a quantum harmonic oscillator, ```psi(x)``` is the wavefunction such that ```|psi(x)|²``` gives the probability density of finding the particle at position```x```. For GKP states specifically, it's a series of Gaussian peaks spaced at regular intervals (√π apart for the |0⟩ state).
2 - **In the code**:
psi is a NumPy array containing complex numbers ```(dtype=np.complex128)```, each element corresponds to the value of the wavefunction at a particular position ```x[i]```.

Why do we normalize it?         ```np.sum(np.abs(psi)**2) * (x[1]-x[0]) ≈ 1```
As we already know, a qubit can exist in a complex superposition of the two states (|0⟩ and |1⟩). Dirac notation is used to represent the two basis states, denoted as |0⟩and |1⟩. The arbitrary state of an individual qubit , |ψ⟩, can be expressed as,
*|ψ⟩= α|0⟩+ β|1⟩*
where |0⟩ and |1⟩ are the two orthonormal basis states of the qubit and α and β are complex numbers. We also have the condition,
*α^2 + β^2 = 1*
which corresponds to the probability of observing a 0 or 1 when measuring that qubit. α^2 is the probabilty of observing a 0 and β^2 is the probability of observing a 1.

Whats is the physical meaning? 
For the ideal GKP |0⟩ state, ψ(x) would be an infinite sum of delta functions at positions 2n√π. In our simulation, we use Gaussian approximations (finite-energy GKP states): ```psi += np.exp(-(x - 2*n*np.sqrt(np.pi))**2/(2*delta**2))```



_Step 3: Simulating Errors_
*Position Shift*
We shift the position by multiplying by ```exp(i*p̂*shift_q)```, in this code it is implemented by ```x * shift_p```.
In the position basis:
    Position operator: q̂ψ(x) = x·ψ(x)
    Momentum operator: p̂ψ(x) = -i·dψ(x)/dx (in natural units where ħ=1)
This is because for the GKP states, small position errors (shift_q) appear as phases in momentum space. The error correction will then measure these phase shifts modulo √π. The correction will then apply the inverse shift. 

*Momentum Shift*
In the GKP simulation code, applying a momentum shift is more involved than a position shift because momentum space is the Fourier dual of position space and a momentum shift corresponds to a position-space translation in the wavefunction's phase
code implementation:
```
# Momentum shift is convolution with exp(i*q̂*shift_p)
psi_shifted = np.fft.fftshift(np.fft.ifft(np.fft.fft(np.fft.fftshift(psi_shifted)) * np.exp(1j * np.fft.fftfreq(len(x), x[1]-x[0]) * shift_q)))
```
Indeed, A momentum shift in position space is equivalent to multiplying by a phase factor in momentum space:
ψ(p)→ψ(p)⋅e^(−iq⋅shift_p)
But since we are working in discrete space (x), we use the Discrete Fourier Transform (DFT) via ```np.fft.fft```

**Applying the Momentum Shift in Fourier Space**
• ```np.fft.fftfreq(len(x), x[1]-x[0])``` computes the momentum grid (since p=ℏk, and k is the Fourier frequency).
• Multiplying by ```np.exp(1j * p_grid * shift_p)``` applies the phase shift corresponding to a momentum displacement.
• After modifying the wavefunction in momentum space, we transform back to position space using ```np.fft.iff```.
• ```np.fft.fftshift```ensures the zero-frequency component is centered (important for visualization and correct phase shifts).

An alternative Interpretation would be the Displacement Operator D(α), that shifts phase space. For a pure momentum shift ```α=i⋅shift_p```. This generates a phase-space translation, which in position space is implemented via Fourier transforms. 



_Step 3 - Simulating the GKP error correction_
**1 - Measuring the position (q) Modulo √π**
``` 
q_values = x[np.abs(psi)**2 > 0.1*np.max(np.abs(psi)**2)]  # Find positions where the wavefunction has significant probability
q_syndromes = q_values % np.sqrt(np.pi)                     # Compute those positions modulo √π 
```
A perfect GKP state has peaks at integer multiples of √π in position space (q = n√π), if an error shifts that state (e.g. by ∂q), the peaks move to ```q = n√π + ∂q```. Hence, to detect an error, we measure how far the peaks are from the nearest GKP lattice point (this is ∂q mod √π)

<u>e.g.</u>
Suppose the peak is at position ```q = 3√π +0.4``` then the ```q.mod√π = 0.4```=> the syndrome tells us the shift is +0.4

**2 - Measuring the momentum (p) Modulo √π**
```
psi_p = np.fft.fftshift(np.fft.fft(np.fft.fftshift(psi)))  # Fourier transform to momentum space
p = np.fft.fftfreq(len(x), x[1]-x[0]) * 2*np.pi            # Get corresponding momentum values
p_values = p[np.abs(psi_p)**2 > 0.1*np.max(np.abs(psi_p)**2)]  # Find peaks in momentum space
p_syndromes = p_values % np.sqrt(np.pi)                     # Compute momenta modulo √π
```
In momentum space, a perfect GKP state also has peaks at ```p=n√π```, a momentum shift (e.g. ∂p) oves the peaks to ```p = n√π + ∂p```. The syndrome ```p mod √π``` measures how far teh peaks are shifted from the ideal lattice. The measuring of the momentum differs a bit from the measuring of the position because we have to apply the Fourier Transform beforehand. 
The position (q) and momentum (p) are Fourier duals in quantum mechanics, which means, a **shift position** (q → q + ∂q) **introduces a phase in momentum space** (ψ(p)→ψ(p)⋅e^(−ip⋅∂q)) and a **shift in momentum** (p → p + ∂p) iss directly measurable in the Fourier-transformed wavefunction.  

<u>Summup & Visualization</u>

**Position Space (q):**
Ideal GKP peaks:    |    |    |    |    |    (spaced by √π)
Shifted peaks:      |  • |  • |  • |  • |    (• = peaks shifted by δq)
                   ↑
               δq mod √π is the offset from the dashed line.

**Momentum Space (p):**
The Fourier transform of the GKP state looks like another grid of peaks (but in momentum).
A shift in q → phase oscillations in p-space.
A shift in p → direct displacement of momentum peaks.




_Coming up_
1. Performance Analysis: Run many trials with random shifts to calculate average fidelity
2. Noise Models: Implement more realistic noise models (Gaussian shifts, photon loss)
3. Concatenation: Study GKP codes concatenated with other QEC codes
4. Thresholds: Determine the maximum correctable shift for given delta

First idea of code
```
def performance_analysis(num_trials=1000, max_shift=0.5):
    fidelities = []
    for _ in range(num_trials):
        # Random shift errors
        shift_q, shift_p = np.random.uniform(-max_shift, max_shift, 2)
        
        # Generate state and apply errors
        x, psi = gkp_state(delta)
        psi_err = apply_shift_error(psi, shift_q, shift_p, x)
        
        # Correct
        q_syn, p_syn = gkp_syndrome_measurement(psi_err, x)
        psi_corr = gkp_correct(psi_err, x, q_syn, p_syn)
        
        # Calculate fidelity
        fid = np.abs(np.sum(psi.conj() * psi_corr))**2
        fidelities.append(fid)
    
    print(f"Average fidelity: {np.mean(fidelities):.4f} ± {np.std(fidelities):.4f}")
    plt.hist(fidelities, bins=20)
    plt.xlabel('Fidelity')
    plt.ylabel('Count')
    plt.show()
```