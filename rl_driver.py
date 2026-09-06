"""
rl_driver.py — in-process, step-at-a-time CC3D driver for the RL loop.

Why not CC3D's simservice? On CC3D 4.9 / 4.10 `service_cc3d(...)` runs the model
in a spawned subprocess whose Python steppables never get wired to the C++ cell
inventory (`self.cell_list` is empty, `getNumCells()==0` even though the core
created every cell). The plain CML path works fine, so this driver runs it — but
pauses `simulation_setup.main_loop` right before its `while` loop and hands the
stepping back to us.

    from rl_driver import CC3DDriver
    d = CC3DDriver("MuscleRegenRLQuarter.cc3d", seed=0)
    d.start()                                  # runs steppable start(), MCS 0
    out0 = d.sim_output()                       # RLSteppable's pg.return_object
    for _ in range(N):
        out = d.step({"actions": {...}})        # advance one MCS, returns pg.return_object
    d.close()

One sim per process (CC3D core + model module globals are singletons).
No multiprocessing -> safe to import and use from a plain script or notebook,
no `if __name__ == "__main__"` guard required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


class CC3DDriver:
    def __init__(self, cc3d_name: str, seed: int | None = None):
        self._cc3d_name = cc3d_name
        self._seed = seed
        self._sim = None
        self._reg = None
        self._pg = None
        self._mcs = -1
        self._started = False

    # ------------------------------------------------------------------
    def start(self):
        """Load the model, run every steppable's start(), execute MCS 0."""
        if self._started:
            raise RuntimeError("CC3DDriver already started; make a new one per episode")

        if self._seed is not None:
            os.environ["MUSCLEREGEN_SEED"] = str(self._seed)

        from rl_stage import sim_paths
        from cc3d import CompuCellSetup
        import cc3d.CompuCellSetup.simulation_setup as ss
        from cc3d.CompuCellSetup.readers import readCC3DFile

        cc3d_file, out_dir = sim_paths(self._cc3d_name)

        captured = {}

        def _paused_main_loop(sim, simthread=None, steppable_registry=None):
            pg = CompuCellSetup.persistent_globals
            reg = pg.steppable_registry
            ss.initialize_cc3d_sim(sim, simthread)      # -> sim.start(): PIF cells
            reg.init(sim)
            ss.init_lattice_snapshot_objects()
            try:
                ss.init_screenshot_manager()
            except Exception:
                pass
            reg.start()                                 # Python steppable start()
            captured["sim"] = pg.simulator
            captured["reg"] = reg

        self._orig_main_loop = ss.main_loop
        ss.main_loop = _paused_main_loop
        try:
            # run_cc3d_project() would call simulator.cleanAfterSimulation() right
            # after exec and wipe every cell, so inline its body without that.
            data = readCC3DFile(fileName=cc3d_file)
            CompuCellSetup.cc3dSimulationDataHandler = data
            sim_dir = os.path.join(os.path.dirname(cc3d_file), "Simulation")
            if sim_dir not in sys.path:
                sys.path.insert(0, sim_dir)
            script = data.cc3dSimulationData.pythonScript
            code = compile(Path(script).read_text(), script, "exec")
            exec(code, {"__name__": "__cc3d_rl_sim__"})
        finally:
            ss.main_loop = self._orig_main_loop

        self._sim = captured["sim"]
        self._reg = captured["reg"]
        self._pg = CompuCellSetup.persistent_globals
        self._mcs = 0
        self._started = True

        # MCS 0: CML main_loop skips the Potts sub-step at mcs 0 unless asked;
        # run the steppables so RLSteppable populates the first observation.
        self._pg.input_object = {"actions": {}}
        self._reg.stepRunBeforeMCSSteppables(0)
        if self._pg.execute_step_at_mcs_0:
            self._sim.step(0)
        self._reg.step(0)
        return self.sim_output()

    # ------------------------------------------------------------------
    def step(self, sim_input: dict | None = None):
        """Advance exactly one MCS. `sim_input` lands on pg.input_object."""
        if not self._started:
            raise RuntimeError("call start() first")
        self._mcs += 1
        self._pg.input_object = sim_input if sim_input is not None else {}
        self._reg.stepRunBeforeMCSSteppables(self._mcs)
        self._sim.step(self._mcs)
        self._reg.step(self._mcs)
        return self.sim_output()

    # ------------------------------------------------------------------
    def sim_output(self):
        ro = self._pg.return_object
        return ro if isinstance(ro, dict) else {}

    @property
    def mcs(self) -> int:
        return self._mcs

    @property
    def num_steps(self) -> int:
        return self._sim.getNumSteps() if self._sim else 0

    # ------------------------------------------------------------------
    def close(self):
        if self._sim is not None:
            try:
                self._reg.finish()
            except Exception:
                pass
            try:
                self._sim.cleanAfterSimulation()
            except Exception:
                pass
            self._sim = None
            self._reg = None
        try:
            from cc3d import CompuCellSetup
            CompuCellSetup.resetGlobals()
        except Exception:
            pass
        self._started = False
