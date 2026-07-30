
from cc3d import CompuCellSetup

from MuscleRegenSteppables import PassValuesSteppable
CompuCellSetup.register_steppable(steppable=PassValuesSteppable(frequency=1))

from MuscleRegenSteppables import ConstraintInitializerSteppable
CompuCellSetup.register_steppable(steppable=ConstraintInitializerSteppable(frequency=1))

from MuscleRegenSteppables import FiberSteppable
CompuCellSetup.register_steppable(steppable=FiberSteppable(frequency=1))

from MuscleRegenSteppables import PlotSteppable
CompuCellSetup.register_steppable(steppable=PlotSteppable(frequency=20))

from MuscleRegenSteppables import FileSteppable
CompuCellSetup.register_steppable(steppable=FileSteppable(frequency=4))

from MuscleRegenSteppables import SSCPeakTrackerSteppable
CompuCellSetup.register_steppable(steppable=SSCPeakTrackerSteppable(frequency=1))

from MuscleRegenSteppables import NecrosisSteppable
CompuCellSetup.register_steppable(steppable=NecrosisSteppable(frequency=4))

from MuscleRegenSteppables import NeutrophilSteppable
CompuCellSetup.register_steppable(steppable=NeutrophilSteppable(frequency=8))

# RLSteppable replaces SSCSteppable — RL policy drives SSC decisions
from RLSteppable import RLSteppable
CompuCellSetup.register_steppable(steppable=RLSteppable(frequency=8))

from MuscleRegenSteppables import MacrophageSteppable
CompuCellSetup.register_steppable(steppable=MacrophageSteppable(frequency=4))

from MuscleRegenSteppables import SSCMitosisSteppable
CompuCellSetup.register_steppable(steppable=SSCMitosisSteppable(frequency=4))

from MuscleRegenSteppables import LymphaticSteppable
CompuCellSetup.register_steppable(steppable=LymphaticSteppable(frequency=10))

from MuscleRegenSteppables import MicrovesselSteppable
CompuCellSetup.register_steppable(steppable=MicrovesselSteppable(frequency=4))

from MuscleRegenSteppables import FibroblastSteppable
CompuCellSetup.register_steppable(steppable=FibroblastSteppable(frequency=4))

from MuscleRegenSteppables import FibroblastMitosisSteppable
CompuCellSetup.register_steppable(steppable=FibroblastMitosisSteppable(frequency=4))

from MuscleRegenSteppables import M2MitosisSteppable
CompuCellSetup.register_steppable(steppable=M2MitosisSteppable(frequency=4))

CompuCellSetup.run()
