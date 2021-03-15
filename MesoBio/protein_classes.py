import numpy as np
import MesoBio.PDB as PDB


def prepare_lhcii():
    lhcii = PDB.MMCIFParser().get_structure('1RWT', './rw/1rwt.cif')
    list_chain_keep = ['J', 'D', 'I']
    list_chain_all = [chain.id for chain in lhcii[0].child_list]

    for id in list_chain_all:
        print(id)
        if id not in list_chain_keep:
            lhcii[0].detach_child(id)

    chl_list = []
    for res in lhcii.get_residues():
        if res.get_resname() == 'CLA' or res.get_resname() == 'CHL':
            chl_list.append(res)
            res.prepare_chlorophyll()

    lhcii = lhcii[0]
    lhcii._name = 'LHCII'
    lhcii.list_pigments = chl_list
    lhcii.dict_pigments = {chl._name: chl for chl in chl_list}
    lhcii.list_names = [chl._name for chl in chl_list]
    lhcii.transform(np.eye(3), -lhcii.center_of_mass)
    lhcii.flatten()

    return lhcii



