"""GPU/simulator side: adapters, builders, rollouts, sanity gate.

adapters and builders, record rollouts, and the SR sanity gate.

Import discipline: `vln_backends.config` is side-effect free;
`vln_backends.bootstrap` chdirs into the DUET source tree and imports
MatterSim + DUET (order-sensitive), so import it only from GPU entry points.
"""
