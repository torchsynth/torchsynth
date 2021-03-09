"""
Default parameters
"""
DEFAULT_SAMPLE_RATE = 44100

# Number of samples for fixed length synthesis.
# 4 seconds by default
DEFAULT_BUFFER_SIZE = 4 * DEFAULT_SAMPLE_RATE

# Small value to avoid log errors.
EPSILON = 1e-6

# Equal power coefficient. 1/sqrt(2)
EQ_POW = 0.70710678118
