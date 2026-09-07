"""
RLSteppable — external-policy replacement for SSCSteppable.

Registered *instead of* SSCSteppable in MuscleRegenRL.py. Every MCS it:

  1. reads   pg.input_object  = {"actions": {cell_id: action_int}}
  2. builds a 22-float observation per live SSC (from the field / cell state
     BEFORE any action is applied this step)
  3. executes the actions, reproducing the field-uptake / stop-moving
     side-effects of the original hand-coded decisions
  4. runs the parts of SSC behaviour that are NOT agent-controlled
     (recruitment, environmental apoptosis, quiescence, activation-by-HGF)
  5. writes  pg.return_object = {"raw": {...}, "obs": {...}, "done": bool, "mcs": int}

The scalar reward is deliberately NOT computed here — RLSteppable only exports
raw quantities and rl_env.reward_fn() turns them into a number, so the reward can
be changed without touching CC3D or restarting the container.

Actions
-------
0  do nothing / migrate (chemotax along MMP; myoblasts also TNF)
1  activate                 (quiescent -> start activation lag)
2  divide                   (active SSC/myoblast on a fiber edge -> division lag)
3  differentiate            (active SSC/myoblast on a fiber edge -> differentiation lag)
4  fuse to adjacent fiber   (myocyte only)
5  fuse with adjacent myocyte -> myotube  (myocyte only)
6  apoptose

--------------------------------------------------------------------------------
FIDELITY NOTE — read before training against "recover the hand-coded policy"
--------------------------------------------------------------------------------
The original SSCSteppable.step (MuscleRegenSteppables.py ~596-791) gates each
decision on local cytokine thresholds AND applies binding side-effects. This
steppable lets the *agent* choose the decision but still reproduces:
  * divide  -> subtract SSCdivisionThreshold/3 from TNF, VEGF, TGF at the cell
              (orig lines 638-649) and stop migrating this step (orig 690-698)
  * diff    -> subtract SSCdiffThreshold/4 from IL10, HGF, TGF, TNF
              (orig lines 656-673) and stop migrating this step
  * activate-> subtract sscActivationThreshold from HGF (orig 780)
  * fuse    -> remove the ECM-below from lowColIdx, set repair, targetVolume 0
Still simplified vs. the original (decide with the team if it matters):
  * the original also required `fiberPresent` (a mature-fiber neighbour of the
    cell below) for divide/diff; here the agent may divide/diff anywhere. The
    obs exposes the fiber-neighbour count so a learned policy can gate itself.
  * per-numDiv division probability (sscDivisionChanceSubseq) is not applied.
  * `moveCount` is not tracked.
"""

from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import numpy as np
import random
import math

import MuscleRegenSteppables as sim_params
from MuscleRegenSteppables import (
    sscActivationTime, sscDivisionTime, sscDifferentiationTime, myotubeMatureTime,
    sscTGFApoptosisThreshold, sscApoptosisProb, VEGFblockApop, quiescentThreshold,
    lowCollagenCutoff, highCollagenCutoff, fuseProb, lmSSC, lmRepel,
    secretionUptakeValSSC, uptakeRel, SSCRecruitmentFreq, macRecruitStop,
    recruitMultiplier, recruitmentProportionSSC, sscActivationThreshold,
    quiescentProb, sscToInitialFiber, SSCdivisionThreshold, SSCdiffThreshold,
)

_CYTO_MAX = 500.0          # rough cytokine normalisation for the observation
_SSC_BLOWUP = 12000        # match FileSteppable's runaway-population safety stop

NOOP, ACTIVATE, DIVIDE, DIFFERENTIATE, FUSE_FIBER, FUSE_MYOTUBE, APOPTOSE = range(7)


class RLSteppable(SteppableBasePy):

    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self.track_cell_level_scalar_attribute(field_name='cellType',
                                               attribute_name='cellType')
        self._initial_fiber_volume = 1.0

    # ------------------------------------------------------------------ start
    def start(self):
        # Mirror SSCSteppable.start(): seed the quiescent SSC pool.
        sim_params.num_cells = math.floor(sim_params.numFiberCells / sscToInitialFiber)
        sim_params.newMyotube = 0
        if sim_params.num_cells <= 0:
            raise RuntimeError(
                f"RLSteppable: SSC seeding would place 0 cells "
                f"(numFiberCells={sim_params.numFiberCells}, sscToInitialFiber={sscToInitialFiber})")

        for _ in range(sim_params.num_cells):
            placed = False
            while not placed:
                x = np.random.randint(1, self.dim.x)
                y = np.random.randint(1, self.dim.y)
                if not self.cell_field[x, y, 1]:
                    ssc = self.new_cell(self.SSC)
                    ssc.targetVolume = 10
                    ssc.lambdaVolume = 50
                    ssc.dict['activationState'] = 0
                    ssc.dict['time2activate'] = -1
                    ssc.dict['time2divide'] = -1
                    ssc.dict['time2diff'] = -1
                    ssc.dict['numDiv'] = 0
                    ssc.dict['cellType'] = 0
                    ssc.dict['moveCount'] = 0
                    self.cell_field[x, y, 1] = ssc
                    placed = True

    # ------------------------------------------------------------------- step
    def step(self, mcs):
        pg = CompuCellSetup.persistent_globals

        HGFsec  = self.get_field_secretor("HGF")
        MMPsec  = self.get_field_secretor("MMP")
        MCPsec  = self.get_field_secretor("MCP")
        VEGFsec = self.get_field_secretor("VEGF")
        TNFsec  = self.get_field_secretor("TNF")

        HGFf  = self.field.HGF
        MMPf  = self.field.MMP
        VEGFf = self.field.VEGF
        TNFf  = self.field.TNF
        TGFf  = self.field.TGF
        IL10f = self.field.IL10

        necrosisRemain = getattr(sim_params, 'necrosisRemain', 1.0)

        if mcs == 0:
            self._initial_fiber_volume = max(
                sum(c.volume for c in self.cell_list_by_type(self.FIBER)), 1.0)

        # -------------------------------------------------- SSC recruitment
        # Verbatim from SSCSteppable.step (environment behaviour, not an action).
        if mcs % SSCRecruitmentFreq == 0 and necrosisRemain > macRecruitStop:
            # Field means by a 500-point random sample (was a full-lattice loop).
            _sx = np.random.randint(0, self.dim.x - 1, 500)
            _sy = np.random.randint(0, self.dim.y - 1, 500)
            HGFMean = float(np.mean([HGFf[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))
            MMPMean = float(np.mean([MMPf[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))
            TGFMean = float(np.mean([TGFf[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))

            numRecruited = int(np.ceil(recruitmentProportionSSC * (HGFMean + MMPMean - TGFMean)))
            if numRecruited > 0:
                fieldRepel = self.field.REPEL
                coord = np.empty([0, 2])
                need = numRecruited * recruitMultiplier
                while need > 0:
                    cx = np.random.randint(1, self.dim.x, need)
                    cy = np.random.randint(1, self.dim.y, need)
                    ok = [(int(x), int(y)) for x, y in zip(cx, cy)
                          if fieldRepel[int(x), int(y), 1] < 0.1]
                    if ok:
                        coord = np.vstack((coord, np.array(ok)))
                        need -= len(ok)
                vals = [HGFf[int(x), int(y), 1] for x, y in coord]
                for x, y in coord[np.argsort(vals)[-numRecruited:]]:
                    xi, yi = int(x), int(y)
                    if not self.cell_field[xi, yi, 1]:
                        ssc = self.new_cell(self.SSC)
                        self.cell_field[xi, yi, 1] = ssc
                        ssc.targetVolume = 10
                        ssc.lambdaVolume = 50
                        ssc.dict.update({'activationState': 1, 'time2activate': -1,
                                         'time2divide': -1, 'time2diff': -1,
                                         'numDiv': 0, 'cellType': 0, 'moveCount': 0})
                        hv = HGFf[ssc.xCOM, ssc.yCOM, ssc.zCOM]
                        if hv >= secretionUptakeValSSC:
                            HGFf[ssc.xCOM, ssc.yCOM, ssc.zCOM] = hv - secretionUptakeValSSC
                        mv = MMPf[ssc.xCOM, ssc.yCOM, ssc.zCOM]
                        if mv >= secretionUptakeValSSC:
                            MMPf[ssc.xCOM, ssc.yCOM, ssc.zCOM] = mv - secretionUptakeValSSC

        # -------------------------------------------------- observations first
        ssc_cells = list(self.cell_list_by_type(self.SSC))
        obs = {c.id: self._build_obs(c, mcs, HGFf, MMPf, TGFf, VEGFf, TNFf, IL10f,
                                     necrosisRemain)
               for c in ssc_cells}

        inp = pg.input_object
        _raw_actions = inp.get("actions", {}) if isinstance(inp, dict) else {}
        # keys may arrive as str if simservice JSON-serialises the input
        actions = {int(k): int(v) for k, v in _raw_actions.items()}

        # -------------------------------------------------- apply actions
        for cell in ssc_cells:
            if cell.volume == 0:
                continue
            action = int(actions.get(cell.id, NOOP))
            stop_moving = self._apply_action(cell, action, mcs, HGFf, MMPf, TGFf,
                                             VEGFf, TNFf, IL10f,
                                             HGFsec, MMPsec, MCPsec, VEGFsec, TNFsec)

            # activated-SSC secretion (orig 600-602)
            if cell.dict['activationState'] == 1 and mcs % 2 == 0:
                VEGFsec.secreteOutsideCellAtBoundary(cell, secretionUptakeValSSC)
                MCPsec.secreteOutsideCellAtBoundary(cell, secretionUptakeValSSC)

            # migration (orig 700-716) unless a divide/diff stopped the cell
            if not stop_moving and cell.dict['time2divide'] < 0 and cell.dict['time2diff'] < 0:
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                if cd:
                    cd.setLambda(lmSSC)
                    MMPsec.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValSSC, uptakeRel)
                cdr = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                if cdr:
                    cdr.setLambda(lmRepel)
                if cell.dict['cellType'] == 1:
                    cdt = self.chemotaxisPlugin.getChemotaxisData(cell, "TNF")
                    if cdt:
                        cdt.setLambda(lmSSC)
                        TNFsec.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValSSC, uptakeRel)

        # -------------------------------------------------- environmental apop / quiescence / activation
        # Verbatim intent from orig 748-791 (never agent-controlled).
        for cell in ssc_cells:
            if cell.volume == 0:
                continue
            tgf = TGFf[cell.xCOM, cell.yCOM, cell.zCOM]
            hgf = HGFf[cell.xCOM, cell.yCOM, cell.zCOM]

            if tgf > sscTGFApoptosisThreshold and np.random.uniform() < sscApoptosisProb:
                vegf = VEGFf[cell.xCOM, cell.yCOM, cell.zCOM]
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

            if cell.dict['activationState'] == 0:
                hgf = HGFf[cell.xCOM, cell.yCOM, cell.zCOM]
                if hgf >= sscActivationThreshold:
                    cell.dict['time2activate'] = sscActivationTime
                    HGFf[cell.xCOM, cell.yCOM, cell.zCOM] = hgf - sscActivationThreshold
                elif (np.random.randint(int(quiescentProb)) == 0
                      and len(ssc_cells) > sim_params.num_cells):
                    cell.targetVolume = 0
                    cell.lambdaVolume = 1_000_000

            cell.dict['time2divide'] -= 1
            cell.dict['time2diff'] -= 1
            cell.dict['time2activate'] -= 1

        # -------------------------------------------------- export raw signals + obs
        raw = self._raw_signals(mcs, HGFf, MMPf, TGFf, VEGFf, TNFf, IL10f, necrosisRemain)
        done = (mcs + 1 >= self.simulator.getNumSteps()) or (raw["n_ssc"] > _SSC_BLOWUP)

        pg.return_object = {
            "obs":  {cid: o.tolist() for cid, o in obs.items()},
            "raw":  raw,
            "done": bool(done),
            "mcs":  int(mcs),
        }

    def _raw_signals(self, mcs, HGFf, MMPf, TGFf, VEGFf, TNFf, IL10f, necrosisRemain):
        """Whole-tissue quantities for rl_env.reward_fn(). All plain Python scalars."""
        fibers = list(self.cell_list_by_type(self.FIBER))
        sscs   = list(self.cell_list_by_type(self.SSC))
        ecm    = list(self.cell_list_by_type(self.ECM))
        macs   = list(self.cell_list_by_type(self.MACROPHAGE))

        n_active = sum(1 for c in sscs if c.dict['activationState'] == 1)
        n_myoblast = sum(1 for c in sscs if c.dict['cellType'] == 1)
        n_myocyte  = sum(1 for c in sscs if c.dict['cellType'] == 2)
        collagen = [c.dict.get('collagen', 1.0) for c in ecm]

        # mean cytokines from a 200-point random sample (cheap; feeds reward shaping)
        sx = np.random.randint(0, self.dim.x - 1, 200)
        sy = np.random.randint(0, self.dim.y - 1, 200)
        def _mean(f):
            return float(np.mean([f[int(x), int(y), 1] for x, y in zip(sx, sy)]))

        return {
            "mcs":                  int(mcs),
            "fiber_volume":         float(sum(c.volume for c in fibers)),
            "initial_fiber_volume": float(self._initial_fiber_volume),
            "fiber_count":          len(fibers),
            "n_myotube_immature":   sum(1 for c in fibers if c.dict.get('time2mature', -1) > -1),
            "newMyotube":           int(getattr(sim_params, 'newMyotube', 0)),
            "n_ssc":                len(sscs),
            "n_ssc_active":         n_active,
            "n_myoblast":           n_myoblast,
            "n_myocyte":            n_myocyte,
            "n_macrophage":         len(macs),
            "n_neutrophil":         len(list(self.cell_list_by_type(self.NEUTROPHIL))),
            "n_fibroblast":         len(list(self.cell_list_by_type(self.FIBROBLAST))),
            "n_necrotic":           len(list(self.cell_list_by_type(self.NECROTIC))),
            "necrosisRemain":       float(necrosisRemain),
            "collagen_mean":        float(np.mean(collagen)) if collagen else 0.0,
            "collagen_fibrotic_frac": (sum(1 for v in collagen if v > 10) / len(collagen)
                                       if collagen else 0.0),
            "cyto_mean": {"HGF": _mean(HGFf), "MMP": _mean(MMPf), "TGF": _mean(TGFf),
                          "VEGF": _mean(VEGFf), "TNF": _mean(TNFf), "IL10": _mean(IL10f)},
        }

    # ------------------------------------------------------------------ helpers
    def _build_obs(self, cell, mcs, HGFf, MMPf, TGFf, VEGFf, TNFf, IL10f, necrosisRemain):
        x, y, z = cell.xCOM, cell.yCOM, cell.zCOM
        counts = {t: 0 for t in (self.FIBER, self.ECM, self.NECROTIC,
                                 self.CAPILLARY, self.SSC, self.MACROPHAGE)}
        for nb, _ in self.get_cell_neighbor_data_list(cell):
            if nb and nb.type in counts:
                counts[nb.type] += 1
        total = max(self.simulator.getNumSteps(), 1)
        return np.array([
            x / self.dim.x,
            y / self.dim.y,
            float(cell.dict['activationState']),
            cell.dict['cellType'] / 2.0,
            max(cell.dict['time2activate'], 0) / max(sscActivationTime, 1),
            max(cell.dict['time2divide'], 0) / max(sscDivisionTime, 1),
            max(cell.dict['time2diff'], 0) / max(sscDifferentiationTime, 1),
            min(cell.dict['numDiv'], 4) / 4.0,
            HGFf[x, y, z]  / _CYTO_MAX,
            MMPf[x, y, z]  / _CYTO_MAX,
            TGFf[x, y, z]  / _CYTO_MAX,
            VEGFf[x, y, z] / _CYTO_MAX,
            TNFf[x, y, z]  / _CYTO_MAX,
            IL10f[x, y, z] / _CYTO_MAX,
            float(necrosisRemain),
            counts[self.FIBER]      / 10.0,
            counts[self.ECM]        / 10.0,
            counts[self.NECROTIC]   / 10.0,
            counts[self.CAPILLARY]  / 10.0,
            counts[self.SSC]        / 10.0,
            counts[self.MACROPHAGE] / 10.0,
            mcs / total,
        ], dtype=np.float32)

    def _apply_action(self, cell, action, mcs, HGFf, MMPf, TGFf, VEGFf, TNFf, IL10f,
                      HGFsec, MMPsec, MCPsec, VEGFsec, TNFsec):
        """Returns stop_moving: True if the cell should not migrate this step."""
        x, y, z = cell.xCOM, cell.yCOM, cell.zCOM

        if action == NOOP:
            return False

        if action == ACTIVATE:
            if cell.dict['activationState'] == 0 and cell.dict['time2activate'] < 0:
                hgf = HGFf[x, y, z]
                cell.dict['time2activate'] = sscActivationTime
                if hgf >= sscActivationThreshold:
                    HGFf[x, y, z] = hgf - sscActivationThreshold      # orig 780
            return False

        if action == DIVIDE:
            if (cell.dict['activationState'] == 1 and cell.dict['time2divide'] < 0
                    and cell.dict['time2diff'] < 0 and cell.dict['cellType'] in (0, 1)):
                step_uptake = SSCdivisionThreshold / 3.0               # orig 638-649
                for f in (TNFf, VEGFf, TGFf):
                    v = f[x, y, z]
                    f[x, y, z] = v - step_uptake if v >= step_uptake else 0.0
                cell.dict['time2divide'] = sscDivisionTime
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                if cd:
                    cd.setLambda(0)
                MMPsec.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)
                return True
            return False

        if action == DIFFERENTIATE:
            can_diff = ((cell.dict['cellType'] == 0 and cell.dict['activationState'] == 1)
                        or cell.dict['cellType'] == 1)
            if can_diff and cell.dict['time2diff'] < 0:
                step_uptake = SSCdiffThreshold / 4.0                   # orig 656-673
                for f in (IL10f, HGFf, TGFf, TNFf):
                    v = f[x, y, z]
                    f[x, y, z] = v - step_uptake if v >= step_uptake else 0.0
                cell.dict['time2diff'] = sscDifferentiationTime
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                if cd:
                    cd.setLambda(0)
                return True
            return False

        if action == FUSE_FIBER:
            if cell.dict['cellType'] != 2:
                return False
            below = self.cell_field[x, y, 0]
            if self._ecm_below_fusable(below):
                below.type = self.FIBER
                self.remove_all_cell_fpp_links(below)
                below.dict['repair'] = 1
                cell.targetVolume = 0
                self._drop_from_lowcol(below)
            return True

        if action == FUSE_MYOTUBE:
            if cell.dict['cellType'] != 2:
                return False
            below = self.cell_field[x, y, 0]
            if self._ecm_below_fusable(below):
                partners = [nb for nb, _ in self.get_cell_neighbor_data_list(cell)
                            if nb and nb.type == self.SSC and nb.dict['cellType'] == 2]
                if partners:
                    partner = random.choice(partners)
                    below.type = self.FIBER
                    below.dict['time2mature'] = myotubeMatureTime + random.randint(0, myotubeMatureTime)
                    self.remove_all_cell_fpp_links(below)
                    below.dict['repair'] = 1
                    sim_params.newMyotube += 1
                    cell.targetVolume = 0
                    partner.targetVolume = 0
                    self._drop_from_lowcol(below)
            return True

        if action == APOPTOSE:
            cell.targetVolume = 0
            cell.lambdaVolume = 1_000_000
            MCPsec.secreteOutsideCellAtBoundary(cell, 0)
            VEGFsec.secreteOutsideCellAtBoundary(cell, 0)
            return True

        return False

    def _ecm_below_fusable(self, below):
        return (below and below.type == self.ECM and below.dict['repair'] == 0
                and ((lowCollagenCutoff <= below.dict['collagen'] <= highCollagenCutoff)
                     or np.random.uniform() < fuseProb))

    def _drop_from_lowcol(self, below):
        idx = sim_params.lowColIdx
        bb = (below.xCOM, below.yCOM, below.xCOM, below.yCOM)
        if idx.count(bb) > 0:
            idx.delete(below.id, bb)

    def finish(self):
        return

    def on_stop(self):
        return
