from datetime import datetime
t_start=datetime.now()

import In_Situ_Functions as isf
import numpy as np
import matplotlib.pyplot as plt
import sys


path_name = 'E:/IntTagLib1v2_10June2021/cycles/cycles'
save_path = 'E:/IntTagLib1v2_10June2021/Processed'

isf.Manage_Save_Directories(['Reads', 'Nuclei', 'Cells'], save_path)

for n_well in [1]:
    for n_tile in [100]:

        data = isf.Assemble_Data_From_ND2(n_tile, n_well, path_name)

        maxed, peaks = isf.Find_Peaks(data)

        # nucs = fc.Segment_Nuclei(data[0])
        nucs = np.load(nuc_save_path)

        cells = isf.Segment_Cells(data[0,1:], nucs, method='Watershed')
        # cells = np.load(cell_save_path)

        df_reads = isf.Call_Bases(cells, maxed, peaks, THRESHOLD_STD=500)

        reads_save_path = save_path + '/Reads/reads_tile_' + str(n_tile) + '_well_' + str(n_well) + '.csv'
        nuc_save_path = save_path + '/Nuclei/nuclei_tile_' + str(n_tile) + '_well_' + str(n_well) + '.npy'
        cell_save_path = save_path + '/Cells/cells_tile_' + str(n_tile) + '_well_' + str(n_well) + '.npy'

        df_reads.to_csv(reads_save_path, index=False)
        np.save(nuc_save_path, nucs)
        np.save(cell_save_path, cells)

t_end=datetime.now()
print('Time: ',t_end-t_start)

