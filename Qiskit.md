_pb:_

```
wouldn't it be get a better result using qiskit (like with qiskit.circuit, qiskit.transpiler, qiskit.primitives, or runtime)? Should I implement gates for a better simulation of GKP ECC? and if yes is qiskit a better solution than my actual code?
```

After checking the qiskit website and checking with DeepSeek, qiskit is designed for discrete qubits (not oscillators). GKP states are Bosonic codes (infinite-dimensional Hilbert spaces), which qiskit doesn't natively handle. To force GKP into qubits, I'd need to:
• Truncate the oscillator Hilbert space (losing GKP's infinite dimensional advantages) => I still approximate the qubits in the program that doesn't use qiskit (I use finite energy in ```def gkp_state()```), hence approximating a qubit with qiskit could still be better ...
• Manually implement modular quadrature measurements 


**Comparing my custom code & qiskit**
According to DeepSeek and Replit: my current custom code is better for continous variable creation (native support), GKP Shifts (exact simulation), Modular Measurement (direct). Whereas qiskit is better than my custom code for Hardware compatibility (my custom code doesn't support it...) and gate-level simulation (built in with qiskit). 
Than two possible methods:

1. Stick to my custom code, if I only need to focus on GKP's continous variable properties. => maybe should add gates though ...
2. Hybrid version (custom code + qiskit), if I want to interface with qubit-based error correction and I want to run on IBMQ hardware. 




## Hybrid GKP code version
The only changes done where concerning ```apply_shift_error()```,```gkp_correct()``` where hybrid method was implemented and the animation function was replaced by ```plot_hybrid_results()```

This last function features a new graph called "Bloch Sphere". Every point on the sphere's surface corresponds to a possible pure state of a qubit.
• North Pole = |0> state
• South Pole = |1> state
• Equator = Superposition states (e.g. |+> = (|0>+|1>)/2)
• Other Points = Mixed States with phase information
=> The Bloch sphere shows the logical qubit state inferred from the corrected GKP state.For example If the correction shifted peaks back to q=n*sqrt(π), the Bloch vector would point toward ∣0L⟩ or ∣1L⟩.

If we get a **perfect correction** than the bloch vector points exactly to ∣0L⟩ or ∣1L⟩. For **partial correction** vector points somewhere between the poles. For **phase errors**, the vector moves along the equator. 

--> Why is it helpful with GKP? GKP codes encode a logical qubit in an oscillator. The Bloch sphere helps bridge continuous-variable physics with discrete qubit operations.





## Something that still isn't clear for me ...
### Pb: qiskit does not natively support continuous-variable (CV) quantum systems, it is made for qubit based (DV) models 
### qst: Why isn't GKP qubit based if it also deals with qubits?
As a reminder, GKP is a way to encode a qubit (2D Hilbert space) into a continuous-variable (CV) system, like a harmonic oscillator. Hence, the logical state is a qubit BUT the physical implementation ins't!!! - it's a CV Bosonic mode. <=> GKP represents a qubit, but uses a continuous degree of freedom (position/momentum, aka quadratures of light). 
