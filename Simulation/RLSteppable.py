from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import numpy as np
import random
import math

import MuscleRegenSteppables as sim_params
from MuscleRegenSteppables import (
    sscActivationTime, sscDivisionTime, sscDifferentiationTime, myotubeMatureTime,
    sscTGFApoptosisThreshold, sscApoptosisProb, VEGFblockApop, quiescentThreshold,
    lowCollagenCutoff, fuseProb, lmSSC, lmRepel, secretionUptakeValSSC, uptakeRel,
    SSCRecruitmentFreq, macRecruitStop, recruitMultiplier, recruitmentProportionSSC,
    sscActivationThreshold, quiescentProb, sscToInitialFiber,
)

_CYTO_MAX = 500.0   # rough normalisation for cytokine values


class RLSteppable(SteppableBasePy):
    """
    Replaces SSCSteppable with RL-driven decisions.

    Each MCS it:
      1. Reads  pg.input_object  = {"actions": {cell_id: action_int}}
      2. Executes those actions on SSC cells
      3. Writes pg.return_object = {"obs": {cell_id: [22 floats]},
                                    "reward": float, "done": bool, "mcs": int}

    If input_object is None (first step before agent sends anything) every cell
    defaults to action 0 (do nothing).

    Actions
    -------
    0  do nothing
    1  activate
    2  divide
    3  differentiate
    4  fuse to adjacent fiber  (myocyte only)
    5  fuse with adjacent myocyte → myotube  (myocyte only)
    6  apoptose
    """

    def __init__(self, frequency=8):
        SteppableBasePy.__init__(self, frequency)
        self.track_cell_level_scalar_attribute(field_name='cellType',
                                               attribute_name='cellType')
        self._prev_fiber_volume = 0.0
        self._initial_fiber_volume = 1.0
        self._prev_myotube = 0
        self._pg = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        self._pg = CompuCellSetup.persistent_globals

        # Mirror SSCSteppable.start(): initialise pool of quiescent SSCs
        sim_params.num_cells = math.floor(sim_params.numFiberCells / sscToInitialFiber)
        sim_params.newMyotube = 0

        for _ in range(sim_params.num_cells):
            placed = False
            while not placed:
                x = np.random.randint(1, self.dim.x)
                y = np.random.randint(1, self.dim.y)
                if not self.cellField[x, y, 1]:
                    ssc = self.new_cell(self.SSC)
                    ssc.targetVolume = 10
                    ssc.lambdaVolume = 50
                    ssc.dict['activationState'] = 0
                    ssc.dict['time2activate']   = -1
                    ssc.dict['time2divide']     = -1
                    ssc.dict['time2diff']       = -1
                    ssc.dict['numDiv']          = 0
                    ssc.dict['cellType']        = 0
                    ssc.dict['moveCount']       = 0
                    self.cell_field[x, y, 1] = ssc
                    placed = True

    def step(self, mcs):
        pg = self._pg

        # Record starting fiber volume on first call
        if mcs == 0:
            self._initial_fiber_volume = max(
                sum(c.volume for c in self.cell_list_by_type(self.FIBER)), 1.0
            )
            self._prev_fiber_volume = self._initial_fiber_volume

        # Field handles
        HGFf  = self.field.HGF
        MMPf  = self.field.MMP
        TGFf  = self.field.TGF
        VEGFf = self.field.VEGF
        TNFf  = self.field.TNF
        IL10f = self.field.IL10

        HGFsec  = self.get_field_secretor("HGF")
        MMPsec  = self.get_field_secretor("MMP")
        MCPsec  = self.get_field_secretor("MCP")
        VEGFsec = self.get_field_secretor("VEGF")

        necrosisRemain = getattr(sim_params, 'necrosisRemain', 1.0)

        # ----------------------------------------------------------
        # SSC recruitment (unchanged from original SSCSteppable)
        # ----------------------------------------------------------
        if mcs % SSCRecruitmentFreq == 0 and necrosisRemain > macRecruitStop:
            sx = np.random.randint(0, self.dim.x - 1, 500)
            sy = np.random.randint(0, self.dim.y - 1, 500)
            HGFMean = float(np.mean([HGFf[int(x), int(y), 1] for x, y in zip(sx, sy)]))
            MMPMean = float(np.mean([MMPf[int(x), int(y), 1] for x, y in zip(sx, sy)]))
            TGFMean = float(np.mean([TGFf[int(x), int(y), 1] for x, y in zip(sx, sy)]))
            numRec  = int(np.ceil(recruitmentProportionSSC * (HGFMean + MMPMean - TGFMean)))
            if numRec > 0:
                repel = self.field.REPEL
                coord = np.empty((0, 2))
                need  = numRec * recruitMultiplier
                while need > 0:
                    x = np.random.randint(1, self.dim.x)
                    y = np.random.randint(1, self.dim.y)
                    if repel[x, y, 1] < 0.1:
                        coord = np.vstack((coord, [x, y]))
                        need -= 1
                vals = [HGFf[int(x), int(y), 1] for x, y in coord]
                top  = coord[np.argsort(vals)[-numRec:]]
                for x, y in top:
                    xi, yi = int(x), int(y)
                    if not self.cellField[xi, yi, 1]:
                        ssc = self.new_cell(self.SSC)
                        self.cell_field[xi, yi, 1] = ssc
                        ssc.targetVolume = 10
                        ssc.lambdaVolume = 50
                        ssc.dict.update({
                            'activationState': 1, 'time2activate': -1,
                            'time2divide': -1,    'time2diff': -1,
                            'numDiv': 0,          'cellType': 0, 'moveCount': 0,
                        })
                        hv = HGFf[ssc.xCOM, ssc.yCOM, ssc.zCOM]
                        if hv >= secretionUptakeValSSC:
                            HGFf[ssc.xCOM, ssc.yCOM, ssc.zCOM] = hv - secretionUptakeValSSC
                        mv = MMPf[ssc.xCOM, ssc.yCOM, ssc.zCOM]
                        if mv >= secretionUptakeValSSC:
                            MMPf[ssc.xCOM, ssc.yCOM, ssc.zCOM] = mv - secretionUptakeValSSC

        # ----------------------------------------------------------
        # Read RL actions
        # ----------------------------------------------------------
        inp = pg.input_object
        actions: dict = inp.get("actions", {}) if isinstance(inp, dict) else {}

        # ----------------------------------------------------------
        # Build observations and execute actions
        # ----------------------------------------------------------
        obs_dict: dict = {}
        _ssc_count = len(list(self.cell_list_by_type(self.SSC)))

        for cell in self.cell_list_by_type(self.SSC):
            obs_dict[cell.id] = self._build_obs(cell, mcs, HGFf, MMPf, TGFf,
                                                 VEGFf, TNFf, IL10f)
            action = actions.get(cell.id, 0)
            self._execute_action(cell, action, HGFf, VEGFf, TGFf,
                                 MMPsec, MCPsec, VEGFsec)

            # Active cells chemotax and secrete
            if cell.dict['activationState'] == 1:
                if mcs % 2 == 0:
                    VEGFsec.secreteOutsideCellAtBoundary(cell, secretionUptakeValSSC)
                    MCPsec.secreteOutsideCellAtBoundary(cell, secretionUptakeValSSC)
                if cell.dict['time2divide'] < 0 and cell.dict['time2diff'] < 0:
                    cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                    if cd:
                        cd.setLambda(lmSSC)
                        MMPsec.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValSSC, uptakeRel)
                    cdr = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                    if cdr:
                        cdr.setLambda(lmRepel)

        # ----------------------------------------------------------
        # Environmental apoptosis + quiescence (not agent-controlled)
        # ----------------------------------------------------------
        for cell in self.cell_list_by_type(self.SSC):
            tgf  = TGFf[cell.xCOM, cell.yCOM, cell.zCOM]
            vegf = VEGFf[cell.xCOM, cell.yCOM, cell.zCOM]
            hgf  = HGFf[cell.xCOM, cell.yCOM, cell.zCOM]

            if tgf > sscTGFApoptosisThreshold and np.random.uniform() < sscApoptosisProb:
                mac_nb = any(nb and nb.type == self.MACROPHAGE
                             for nb, _ in self.get_cell_neighbor_data_list(cell))
                if vegf < VEGFblockApop and not mac_nb:
                    cell.targetVolume = 0
                    cell.lambdaVolume = 1_000_000
                    MCPsec.secreteOutsideCellAtBoundary(cell, 0)
                    VEGFsec.secreteOutsideCellAtBoundary(cell, 0)
                    TGFf[cell.xCOM, cell.yCOM, cell.zCOM] = tgf - sscTGFApoptosisThreshold
                elif vegf >= VEGFblockApop:
                    VEGFf[cell.xCOM, cell.yCOM, cell.zCOM] = vegf - VEGFblockApop
            elif hgf < quiescentThreshold:
                cell.dict['activationState'] = 0
                VEGFsec.secreteOutsideCellAtBoundary(cell, 0)
                MCPsec.secreteOutsideCellAtBoundary(cell, 0)

            # Activation check for quiescent cells
            if cell.dict['activationState'] == 0:
                if hgf >= sscActivationThreshold:
                    cell.dict['time2activate'] = sscActivationTime
                    HGFf[cell.xCOM, cell.yCOM, cell.zCOM] = hgf - sscActivationThreshold
                elif (np.random.randint(int(quiescentProb)) == 0
                      and _ssc_count > sim_params.num_cells):
                    cell.targetVolume = 0
                    cell.lambdaVolume = 1_000_000

            # Decrement lag counters
            cell.dict['time2divide']   -= 1
            cell.dict['time2diff']     -= 1
            cell.dict['time2activate'] -= 1

        # ----------------------------------------------------------
        # Reward
        # ----------------------------------------------------------
        cur_fiber = sum(c.volume for c in self.cell_list_by_type(self.FIBER))
        reward = (cur_fiber - self._prev_fiber_volume) / self._initial_fiber_volume
        reward += 0.1 * (sim_params.newMyotube - self._prev_myotube)
        self._prev_fiber_volume = cur_fiber
        self._prev_myotube = sim_params.newMyotube

        done = mcs + 1 >= self.simulator.getNumSteps()

        # ----------------------------------------------------------
        # Write output for RL agent
        # ----------------------------------------------------------
        pg.return_object = {
            "obs":    {cid: o.tolist() for cid, o in obs_dict.items()},
            "reward": float(reward),
            "done":   done,
            "mcs":    mcs,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_obs(self, cell, mcs, HGFf, MMPf, TGFf, VEGFf, TNFf, IL10f):
        x, y, z = cell.xCOM, cell.yCOM, cell.zCOM
        counts = {t: 0 for t in (self.FIBER, self.ECM, self.NECROTIC,
                                  self.CAPILLARY, self.SSC, self.MACROPHAGE)}
        for nb, _ in self.get_cell_neighbor_data_list(cell):
            if nb and nb.type in counts:
                counts[nb.type] += 1

        total_steps = self.simulator.getNumSteps()
        return np.array([
            x / self.dim.x,
            y / self.dim.y,
            float(cell.dict['activationState']),
            cell.dict['cellType'] / 2.0,
            max(cell.dict['time2activate'], 0) / max(sscActivationTime, 1),
            max(cell.dict['time2divide'],   0) / max(sscDivisionTime,   1),
            max(cell.dict['time2diff'],     0) / max(sscDifferentiationTime, 1),
            min(cell.dict['numDiv'], 4) / 4.0,
            HGFf[x, y, z]  / _CYTO_MAX,
            MMPf[x, y, z]  / _CYTO_MAX,
            TGFf[x, y, z]  / _CYTO_MAX,
            VEGFf[x, y, z] / _CYTO_MAX,
            TNFf[x, y, z]  / _CYTO_MAX,
            IL10f[x, y, z] / _CYTO_MAX,
            getattr(sim_params, 'necrosisRemain', 1.0),
            counts[self.FIBER]      / 10.0,
            counts[self.ECM]        / 10.0,
            counts[self.NECROTIC]   / 10.0,
            counts[self.CAPILLARY]  / 10.0,
            counts[self.SSC]        / 10.0,
            counts[self.MACROPHAGE] / 10.0,
            mcs / max(total_steps, 1),
        ], dtype=np.float32)

    def _execute_action(self, cell, action, HGFf, VEGFf, TGFf,
                        MMPsec, MCPsec, VEGFsec):
        if action == 0:
            return

        elif action == 1:  # activate
            if cell.dict['activationState'] == 0 and cell.dict['time2activate'] < 0:
                cell.dict['time2activate'] = sscActivationTime

        elif action == 2:  # divide
            if cell.dict['time2divide'] < 0 and cell.dict['activationState'] == 1:
                cell.dict['time2divide'] = sscDivisionTime
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                if cd:
                    cd.setLambda(0)
                MMPsec.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)

        elif action == 3:  # differentiate
            if cell.dict['time2diff'] < 0 and cell.dict['activationState'] == 1:
                cell.dict['time2diff'] = sscDifferentiationTime
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                if cd:
                    cd.setLambda(0)

        elif action == 4:  # fuse to adjacent fiber (myocyte only)
            if cell.dict['cellType'] != 2:
                return
            below = self.cell_field[cell.xCOM, cell.yCOM, 0]
            if (below and below.type == self.ECM and below.dict['repair'] == 0
                    and (below.dict['collagen'] >= lowCollagenCutoff
                         or np.random.uniform() < fuseProb)):
                fiber_nearby = any(
                    nb and nb.type == self.FIBER
                    for nb, _ in self.get_cell_neighbor_data_list(below)
                )
                if fiber_nearby:
                    below.type = self.FIBER
                    self.remove_all_cell_fpp_links(below)
                    below.dict['repair'] = 1
                    cell.targetVolume = 0
                    idx = sim_params.lowColIdx
                    bb = (below.xCOM, below.yCOM, below.xCOM, below.yCOM)
                    if idx.count(bb) > 0:
                        idx.delete(below.id, bb)

        elif action == 5:  # fuse with adjacent myocyte → myotube
            if cell.dict['cellType'] != 2:
                return
            below = self.cell_field[cell.xCOM, cell.yCOM, 0]
            if (below and below.type == self.ECM and below.dict['repair'] == 0
                    and (below.dict['collagen'] >= lowCollagenCutoff
                         or np.random.uniform() < fuseProb)):
                partners = [
                    nb for nb, _ in self.get_cell_neighbor_data_list(cell)
                    if nb and nb.type == self.SSC and nb.dict['cellType'] == 2
                ]
                if partners:
                    partner = random.choice(partners)
                    below.type = self.FIBER
                    below.dict['time2mature'] = (myotubeMatureTime
                                                 + random.randint(0, myotubeMatureTime))
                    self.remove_all_cell_fpp_links(below)
                    below.dict['repair'] = 1
                    sim_params.newMyotube += 1
                    cell.targetVolume    = 0
                    partner.targetVolume = 0
                    idx = sim_params.lowColIdx
                    bb  = (below.xCOM, below.yCOM, below.xCOM, below.yCOM)
                    if idx.count(bb) > 0:
                        idx.delete(below.id, bb)

        elif action == 6:  # apoptose
            cell.targetVolume = 0
            cell.lambdaVolume = 1_000_000
            MCPsec.secreteOutsideCellAtBoundary(cell, 0)
            VEGFsec.secreteOutsideCellAtBoundary(cell, 0)
