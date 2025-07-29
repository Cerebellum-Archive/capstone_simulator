"""
Test suite for the Blue Water Macro Capstone Simulator.

This package contains comprehensive unit tests for all core functionality,
ensuring reliability and correctness of quantitative trading simulations.

Test Structure:
- test_utils_simulate.py: Core utilities and transformers
- test_single_target_simulator.py: Single-target prediction tests
- test_multi_target_simulator.py: Multi-target portfolio tests
- test_integration.py: End-to-end integration tests
- conftest.py: Shared fixtures and test configuration

Educational Note:
    Unit testing is crucial in quantitative finance because:
    1. Models handle real money - bugs can be extremely costly
    2. Regulatory compliance requires thorough validation
    3. Complex mathematical operations need verification
    4. Data pipeline integrity must be maintained
    5. Performance metrics must be accurate

Run tests with: pytest tests/
"""

__version__ = "0.1.0"
__author__ = "Conrad Gann (Blue Water Macro Corp.)"