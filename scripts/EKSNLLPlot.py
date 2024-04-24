import subprocess
import matplotlib.pyplot as plt

'''
Generates a plot of EKS Negative Log Likelihood results at different Smoothing Parameters
'''

# Smooth params to try:
smooth_params = [0.01, 0.1, 1, 5, 10, 100, 1000]  # parameters to be tested

# Collect output nll lists as a list of lists
nll_values_list = []

print('Starting runs')
for param in smooth_params:
    # Run existing Python script with different parameters
    result = subprocess.run([
        'python', 'scripts/multicam_example.py',
        '--csv-dir', './data/mirror-mouse',
        '--bodypart-list', 'paw1LH', 'paw2LF', 'paw3RF', 'paw4RH',
        '--camera-names', 'top', 'bot', '--s', str(param)],
        capture_output=True,
        text=True
    )
    print(f'Run successful at smooth_param {param}')

    # Extract nll_values from result
    output_lines = result.stdout.strip().split('\n')
    nll_values = []
    for line in output_lines:
        if line.startswith('NLL is'):
            # Split the line to extract the NLL value
            nll = float(line.split("is")[1].split()[0])
            nll_values.append(nll)

    # Store nll_values in the list
    nll_values_list.append(nll_values)

# Plot results for each list of nll_values
for i, nll_values in enumerate(nll_values_list):
    # Create x-axis values evenly spaced
    x_values = [i] * len(nll_values)
    plt.plot(x_values, nll_values, marker='o', label=f'Smoothing Param: {smooth_params[i]}')

plt.xlabel('Smoothing Parameter')
plt.ylabel('NLL')
plt.xticks(range(len(smooth_params)), smooth_params)  # Set x-axis ticks to smooth_params values
plt.title('mirror-mouse multi-cam EKS NLL vs Smoothing Parameter')
plt.grid(True)

# Save plot as PDF
plt.savefig('nll_vs_smoothing_param.pdf')
print('PDF MADE')
plt.show()
