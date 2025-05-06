# Thresholds

**Thresholds:** Determine the maximum correctable shift for given delta.
The smaller the delta, the better the result normally...

### Pseudocode to implement

1. create a function that tests various shifts and determines the point where the fidelity drops below a certain threshold (e.g., 0.50).
2. test both position and momentum shifts independently to find the maximum correctable shift for each.
=> The function will return the threshold values for position and momentum shifts.

### Code implementation Idea for gkp_simulation.py

**function**

def find_threshold(psi, x, delta, fidelity_threshold=0.50, max_shift=0.5, steps=100):
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
    q_threshold_idx = np.argmax(q_fidelities < fidelity_threshold)  # finds the first index where fidelity drops below our threshold( 0 if 1st elem < threshold; len(q_fidelities) if all fidelities > threshold)

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

**in main**

def main():
    # Parameters
    delta = 0.2 # Squeezing parameter
    shift_q, shift_p = 0.1, 0.2  # Random errors to apply

    # Setup all figures first
    fig1, fig2, fig3, fig4 = setup_figures()
    fig5, fig6 = plt.figure(5), plt.figure(6)

    # Generate GKP state (ONCE)
    x, psi = gkp_state(delta)

    # ... (rest of your existing main function code)

    # Find and print threshold for current delta
    q_thresh, p_thresh = find_threshold(psi, x, delta)
    print(f"\nFor delta = {delta}:")
    print(f"Maximum correctable position shift: {q_thresh:.4f}")
    print(f"Maximum correctable momentum shift: {p_thresh:.4f}")
    
    # Plot threshold vs delta using the same psi
    plot_threshold_vs_delta(psi, x)


#### What we should observe

**Case 1: All shifts good**
``` 
Shifts: [0.1, 0.2, 0.3, 0.4, 0.5]
Fidelities: [0.995, 0.993, 0.991, 0.990, 0.989]
```
argmax(< threshold) returns 4 (index where fidelity first drops)
We take shift[3] = 0.4 as threshold

**Case 2: All shifts above threshold**
```
Shifts: [0.1, 0.2, 0.3, 0.4, 0.5]
Fidelities: [0.995, 0.993, 0.992, 0.991, 0.990]
```
argmax(< threshold) returns 5 (past array end)
First condition triggers (idx=0 check is just for safety)
We return max_shift = 0.5 (might need to test larger shifts)

**Case 3: No shifts good**
```
Shifts: [0.1, 0.2, 0.3, 0.4, 0.5]
Fidelities: [0.985, 0.980, 0.975, 0.970, 0.965]
```
argmax(< threshold) returns 0
First condition fails (fidelity[0] < threshold)
Else clause returns 0

#### Explanation 
As delta decreases (more squeezing) <=> Thresholds increase (the code can correct larger shifts) since sharper peaks make it easier to detect and correct small shifts.
Conversely, delta increases <=>  Threshold decreases since wider peaks means harder to detect and correct small shifts...
--> Steeper curves mean squeezing (delta) has a stronger impact on correctability.
--> Flat regions suggest diminishing returns (e.g., squeezing beyond delta = 0.1 may not improve thresholds significantly).

If we observe an asymmetry between q and p curves, might be due to: Discretization effects in the lattice OR Numerical approximations in the Fourier transforms (accroding to deepseek)

⚠️ **CRITICAL DELTA VALUE**: If thresholds plummet for delta > 0.3, you’d aim for delta ≤ 0.3 in hardware implementations.


### Code implementation Idea for Hybrid_GKP.py
For the hybrid GKP-qubit implementation, we need to modify the threshold analysis to account for both the continuous-variable correction and the discrete qubit-level corrections. Here's how the threshold functions would change

**function**

def find_hybrid_threshold(psi, x, delta, fidelity_threshold=0.50, max_shift=0.5, steps=100):
    """
    Find maximum correctable shift for hybrid GKP-qubit system.
    Returns: (q_threshold, p_threshold, qubit_success_rate)
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
    q_threshold = q_shifts[np.argmax(q_fidelities < fidelity_threshold) - 1]
    p_threshold = p_shifts[np.argmax(p_fidelities < fidelity_threshold) - 1]
    
    # Calculate qubit intervention rates
    q_qubit_rate = np.mean(q_qubit_used[:np.argmax(q_fidelities < fidelity_threshold)])
    p_qubit_rate = np.mean(p_qubit_used[:np.argmax(p_fidelities < fidelity_threshold)])
    
    return q_threshold, p_threshold, q_qubit_rate, p_qubit_rate

def plot_hybrid_threshold_vs_delta(psi, x, delta_range=(0.1, 0.5), steps=20):
    """
    Enhanced plot showing both CV and qubit correction effects
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
    
    # Plot thresholds
    ax1.plot(deltas, q_thresholds, 'b-', label='Position shift threshold')
    ax1.plot(deltas, p_thresholds, 'r-', label='Momentum shift threshold')
    ax1.set_xlabel('Delta (squeezing parameter)')
    ax1.set_ylabel('Maximum correctable shift')
    ax1.grid(True)
    
    # Create second axis for qubit rates
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
    plt.show()

**main**

def main():
    # ... (previous setup code)
    
    # Hybrid threshold analysis
    q_thresh, p_thresh, q_rate, p_rate = find_hybrid_threshold(psi, x, delta)
    print(f"\nHybrid thresholds for delta={delta}:")
    print(f"Position: {q_thresh:.3f} (qubit X rate: {q_rate:.1%})")
    print(f"Momentum: {p_thresh:.3f} (qubit Z rate: {p_rate:.1%})")
    
    # Plot hybrid thresholds
    plot_hybrid_threshold_vs_delta(psi, x)
    
    # ... (rest of main function)


"This analysis reveals how the hybrid system degrades differently from pure CV GKP codes - the qubit backstop prevents complete failure at large delta but comes with increased quantum overhead. The plot helps find the "sweet spot" delta value where qubit use remains manageable while maintaining good error correction." [chatGPT hihi]









# Performance Analysis

**Performance Analysis:** Run many trials with random shifts to calculate average fidelity 

### Code implementation Idea for gkp_simulation.py - using max_shift determined by thresholds function!!

```
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
```

main function
```
def main():
    # [Previous setup code remains the same...]
    
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
```

• **Threshold-Based Testing:**
Uses the find_threshold function to determine maximum testable shifts
Tests random shifts within these bounds for realistic performance metrics
• **Comprehensive Analysis:**
Calculates average fidelity across many trials
Tracks both position and momentum thresholds
Provides visual comparison of fidelity vs thresholds
• **Visualization:**
Two-panel plot shows:
Average fidelity vs delta (top)
Threshold shifts vs delta (bottom)
Clear visualization of trade-offs
• **Practical Insights:**
Shows real-world performance expectations
Helps identify optimal delta values
Quantifies the relationship between squeezing and correctability

For each delta value, the average fidelity when correcting random errors
The maximum correctable shift sizes
How these metrics vary with the squeezing parameter delta


### Code implementation Idea for gkp_simulation.py - using max_shift determined by thresholds function!!

```
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
```
And the main: 
```
def main():
    # [Previous setup code remains the same...]
    
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
```

• **Automated Threshold-Based Testing:**
Uses the threshold function to determine maximum testable shifts
Tests random shifts within these bounds for realistic performance metrics
• **Comprehensive Statistics:**
Calculates average fidelity across many trials
Tracks how often qubit corrections are needed
Reports both position and momentum thresholds
• **Visual Analysis:**
Two-panel plot shows fidelity/thresholds and qubit usage rates
Clear visualization of trade-offs between squeezing and performance
• **Practical Insights:**
Shows real-world performance expectations
Helps identify optimal delta values
Quantifies qubit resource requirements

For each delta value, the average fidelity when correcting random errors
How often the qubit-level corrections are triggered
The relationship between squeezing (delta) and correctable shift size


**Main Difference With the hybrid version**
No qubit correction tracking (pure CV implementation)
Simpler fidelity calculation
Single performance metric (no qubit usage rates)
Cleaner visualization focused on CV performance

