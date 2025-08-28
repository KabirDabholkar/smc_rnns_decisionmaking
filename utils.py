import h5py

def load_binned_trials_from_h5(h5_path):
    """
    Load binned trial data from an HDF5 file.
    
    Parameters:
    -----------
    h5_path : str
        Path to the .h5 file
        
    Returns:
    --------
    tuple : (all_matrices, binned_spikes, metadata)
    """
    
    with h5py.File(h5_path, 'r') as f:
        # Load main 3D array
        all_matrices = f['all_matrices'][:]
        
        # Load individual trial matrices
        binned_spikes = {}
        trial_group = f['trial_matrices']
        for trial_name in trial_group.keys():
            trial_idx = int(trial_name.split('_')[1])
            binned_spikes[trial_idx] = trial_group[trial_name][:]
        
        # Load metadata
        metadata = {}
        meta_group = f['metadata']
        
        # Load attributes
        for key, value in meta_group.attrs.items():
            metadata[key] = value
        
        # Load datasets
        for key in meta_group.keys():
            metadata[key] = meta_group[key][:]
    
    return all_matrices, binned_spikes, metadata