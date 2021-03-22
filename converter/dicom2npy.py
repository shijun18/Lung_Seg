import os,glob
import numpy as np
import pylidc as pl
from pylidc.utils import consensus
import h5py

def save_as_hdf5(data, save_path, key):
    hdf5_file = h5py.File(save_path, 'a')
    hdf5_file.create_dataset(key, data=data)
    hdf5_file.close()

def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


save_path = '/acsa-med/radiology/Med_Seg/LIDC/new_npy_data'
if not os.path.exists(save_path):
    os.makedirs(save_path)

pid_prefix = 'LIDC-IDRI-'

for i in range(1,1011):
    pid = pid_prefix + str(i).zfill(4)
    
    # get scan
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

    # get image
    try:
        vol = scan.to_volume().astype(np.int16)
    except:
        continue
    print(vol.shape)
    print(np.max(vol),np.min(vol))

    # get nodule annotation
    nods = scan.cluster_annotations()

    label = np.zeros_like(vol,dtype=np.uint8)
    if len(nods) != 0:
        for nod in nods:
            cmask,cbbox,masks = consensus(nod, clevel=0.5)
            if np.sum(cmask) > 114: # filter the nodules with a radius < 3mm 
                label[cbbox] = cmask.astype(np.float32)
        if np.sum(label) != 0:
            vol = np.transpose(vol,(2,0,1))
            label = np.transpose(label,(2,0,1))
            save_as_hdf5(vol,os.path.join(save_path,pid+'.hdf5'),'image')
            save_as_hdf5(label,os.path.join(save_path,pid+'.hdf5'),'label')
            assert list(np.unique(label)) == [0,1]
            print('%s done !'%pid)