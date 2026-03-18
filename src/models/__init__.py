"""
DPC-GNN-Acoustic Models Package.

Provides differentiable ultrasound simulation using GNN-based wave propagation.

Main Components:
    - WaveEquationMP: Message passing layer for wave equation spatial discretization
    - AcousticGNN: Full CT-to-US translation model
    - AcousticPropertyMapper: CT HU to acoustic property conversion

Example:
    >>> from src.models import AcousticGNN, WaveEquationMP
    >>> model = AcousticGNN(hidden_dim=64, n_mp_layers=10)
    >>> outputs = model(hu, edge_index, edge_attr, transducer_mask)
"""

from .wave_equation_mp import (
    WaveEquationMP,
    TimeStepping,
    WavePropagationLayer,
    build_acoustic_graph,
    check_cfl_condition,
)

from .acoustic_gnn import (
    AcousticGNN,
    USImageRenderer,
    create_acoustic_gnn,
)

# Import from physics module
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'physics'))
    from acoustic_properties import (
        AcousticPropertyMapper,
        BatchAcousticMapper,
        TISSUE_PROPERTIES,
        db_to_neper,
        neper_to_db,
        get_tissue_name,
        create_acoustic_field,
    )
    
    __all__ = [
        # Wave equation components
        'WaveEquationMP',
        'TimeStepping', 
        'WavePropagationLayer',
        
        # Main model
        'AcousticGNN',
        'USImageRenderer',
        'create_acoustic_gnn',
        
        # Physics
        'AcousticPropertyMapper',
        'BatchAcousticMapper',
        'TISSUE_PROPERTIES',
        'db_to_neper',
        'neper_to_db',
        'get_tissue_name',
        'create_acoustic_field',
        
        # Utilities
        'build_acoustic_graph',
        'check_cfl_condition',
    ]
    
except ImportError:
    # Physics module not available
    __all__ = [
        'WaveEquationMP',
        'TimeStepping',
        'WavePropagationLayer',
        'AcousticGNN',
        'USImageRenderer',
        'create_acoustic_gnn',
        'build_acoustic_graph',
        'check_cfl_condition',
    ]
