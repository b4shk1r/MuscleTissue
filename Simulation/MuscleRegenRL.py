
# RL variant of MuscleRegen.py — driven externally by rl_env.py via CC3D simservice.
#
# Differences from MuscleRegen.py:
#   * SSCSteppable is replaced by RLSteppable (external policy drives SSC decisions).
#   * PlotSteppable is dropped   — builds GUI windows, unsafe under headless simservice.
#   * FileSteppable is dropped   — rl_env owns logging; also avoids the known bug where
#     parallel runs clobber each other's *.txt in the model source dir.
#   * PassValuesSteppable is dropped — it also writes pg.return_object; only RLSteppable
#     should own that channel.
#   * Every steppable stays at frequency=1 (biologically correct: lag counters are
#     decremented once per step() and are sized in MCS).

from cc3d import CompuCellSetup

from MuscleRegenSteppables import ConstraintInitializerSteppable
CompuCellSetup.register_steppable(steppable=ConstraintInitializerSteppable(frequency=1))

from MuscleRegenSteppables import FiberSteppable
CompuCellSetup.register_steppable(steppable=FiberSteppable(frequency=1))

from MuscleRegenSteppables import NecrosisSteppable
CompuCellSetup.register_steppable(steppable=NecrosisSteppable(frequency=1))

from MuscleRegenSteppables import NeutrophilSteppable
CompuCellSetup.register_steppable(steppable=NeutrophilSteppable(frequency=1))

# RLSteppable must be registered AFTER NeutrophilSteppable (it reads necrosisRemain)
# and AFTER FiberSteppable (it reads numFiberCells in start()).
from RLSteppable import RLSteppable
CompuCellSetup.register_steppable(steppable=RLSteppable(frequency=1))

from MuscleRegenSteppables import MacrophageSteppable
CompuCellSetup.register_steppable(steppable=MacrophageSteppable(frequency=1))

from MuscleRegenSteppables import SSCMitosisSteppable
CompuCellSetup.register_steppable(steppable=SSCMitosisSteppable(frequency=1))

from MuscleRegenSteppables import LymphaticSteppable
CompuCellSetup.register_steppable(steppable=LymphaticSteppable(frequency=1))

from MuscleRegenSteppables import MicrovesselSteppable
CompuCellSetup.register_steppable(steppable=MicrovesselSteppable(frequency=1))

from MuscleRegenSteppables import FibroblastSteppable
CompuCellSetup.register_steppable(steppable=FibroblastSteppable(frequency=1))

from MuscleRegenSteppables import FibroblastMitosisSteppable
CompuCellSetup.register_steppable(steppable=FibroblastMitosisSteppable(frequency=1))

from MuscleRegenSteppables import M2MitosisSteppable
CompuCellSetup.register_steppable(steppable=M2MitosisSteppable(frequency=1))

CompuCellSetup.run()
