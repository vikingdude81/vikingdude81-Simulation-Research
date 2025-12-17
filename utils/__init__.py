"""
Utils module for crypto ML trading system.
Helper functions for multiscale processing and spike encoding.
"""

from .multiscale_utils import (
    aggregate_timeframes,
    create_missing_mask,
    interpolate_missing_data,
    align_timeframes
)

from .snn_utils import (
    encode_to_spikes,
    decode_from_spikes,
    calculate_spike_rate,
    poisson_spike_generator
)

__all__ = [
    'aggregate_timeframes',
    'create_missing_mask',
    'interpolate_missing_data',
    'align_timeframes',
    'encode_to_spikes',
    'decode_from_spikes',
    'calculate_spike_rate',
    'poisson_spike_generator'
]
