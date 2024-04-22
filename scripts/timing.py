import subprocess
import time

# Time script for timing runs of EKS.
start_time = time.time()
time_version_1 = subprocess.check_output([
    'python', 'scripts/multicam_example.py',
    '--csv-dir', './data/mirror-mouse',
    '--bodypart-list', 'paw1LH', 'paw2LF', 'paw3RF', 'paw4RH',
    '--camera-names', 'top', 'bot'],
    text=True
)
end_time = time.time()

# Calculate the execution time
execution_time = end_time - start_time

# Print the results
print("Execution time:", execution_time)
