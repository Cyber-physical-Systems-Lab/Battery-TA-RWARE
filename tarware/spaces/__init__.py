"""
Observation-space registry for Tarware environments.

The package exports the global and partial multi-agent observation builders and
maps the Gymnasium observation_type strings used during environment
registration to their concrete classes.
"""

from .MultiAgentGlobalObservationSpace import MultiAgentGlobalObservationSpace
from .MultiAgentPartialObservationSpace import \
    MultiAgentPartialObservationSpace

observation_map = {
    'partial': MultiAgentPartialObservationSpace,
    'global': MultiAgentGlobalObservationSpace
}
