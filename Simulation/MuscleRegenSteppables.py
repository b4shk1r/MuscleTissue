from cc3d.core.PySteppables import *
from cc3d import CompuCellSetup
import time
from cc3d.CompuCellSetup import persistent_globals as pg
#import pandas as pd
from rtree import index
import numpy as np
import random
# random.seed(12345)
import math
import os

#VARIABLES#

# Simulation Variables
# Dimensions need to be updated for every new histology image
xdim = 321
ydim = 417
timeconverter = 60/15 # 1 mcs = 15 minute, conversion factor from hours to mcs
recruitMultiplier = 2 # Multiplier of number recruited cells to check locations
uptakeRel = 0.85       # Relative uptake of cytokines

# Neutrophil Variables
neutroRecruitmentProportion = 7.8322 # Number of neutrophils generated at every recruitment as proportion of necrotic cells
secretionUptakeValNeu = 37.3001         # cytokine secreted by neutrophil per step during and after "eating necrosis" also max uptake value
phagocytoseLagTime = math.ceil((20/60) * timeconverter) # Amount of time it takes to phagocytose and amount of time neutrophils and macrophages secrete during phagoccytosis
neutroDeathTime = math.ceil(12.5 * timeconverter) # Time after birth for neutrophil to die
lmNeu = 750                        # Neutrophil speed
necrosisThreshold = 0.9202          # Neutrophils are recruited as long as the necrosis amount (relative to initial) is greater than this

# Macrophage Variables 
macToInitialFiber = 0.2  # Resident macrophages starting proportion = 1 RM per 5 fibers 
macroRecruitmentFreq = 1              # Start time of macrophage recruitment in steps
macroRecruitmentProportion = 0.4701  # Recruitment of macrophages is proportional to mean MCP
secretionUptakeValMac = 48.2090                 # cytokine secreted by macrophages per step also max uptake value
TNFconverter = 109.9025                    # Value of TNF to convert monocyte to M1
lmMac = 9.3                          # Macrophage speed
lmMono = 75                          # Monocyte speed
meanM1trans = 24*timeconverter
stdM1trans = 12*timeconverter

macRecruitThres = 1.4289; # Required mean MCP to signal recruitment of macrophage
macRecruitStop = 0.1406; # Required necrosis remaining before stopping of macrophage recruitment

M1HalfLife = 24.6360 #Corresponds with rate of M1 death
M2HalfLife = 58.1243 #Corresponds with rate of M2 death
MacrophageDeathPoissonM1 = np.log(2)/(M1HalfLife*timeconverter)   # Used in poisson distribution to get randomness in death time 
MacrophageDeathPoissonM2 = np.log(2)/(M2HalfLife*timeconverter)   # Used in poisson distribution to get randomness in death time
#M2TransitionProb = 2            # Probablility of random M1 to M2 transition
IL10TransThreshold = 78.5442            # Local IL-10 that can induce M1 to M2 transition 
phagoThresholdforNeuM2 = 2.0706      # Phagocytosis threshold for neutrophil apop or M1 to M2 transition 
M2prolifProb = 0.2             # Chance of M2 proliferation 

# SSC Variables 
sscActivationThreshold = 44.7091     # Level of HGF to activate ssc
sscToInitialFiber = 4           # Starting ratio of ssc = 1 per 4 fiber (50 fibers in cross section)
SSCRecruitmentFreq = 1         # Frequency to recruit SSCs
recruitmentProportionSSC = 0.8213 # Number of ssc recruited every time step is proportional to sum of hgf + mmp - TGF means
sscActivationTime = math.ceil(8 * timeconverter)     # Lag time for activation
sscDivisionTime = math.ceil(10*timeconverter)          # Lag time for division
sscDifferentiationTime = math.ceil(18 * timeconverter)     # Lag time for differentiation to fiber (from Kelley's code)
myotubeMatureTime = math.ceil(24 * timeconverter)     # Lag time for maturation of myotube to fiber following myocyte to myocyte fusion (no myocyte fusion to myotube until matures)
sscDivisionChanceSubseq = [0.9,0.85,0.65,0.2] # chance of division decreases with each cell division 
quiescentProb = 3.3617              # Probability of ssc apoptosis vs remaining quiescent
lmSSC = 11.3                    # SSC speed
secretionUptakeValSSC = 44.4650        # cytokine secreted by SSC per step when activated also max uptake value
sscTGFApoptosisThreshold = 79.1563  # Amount of TGF to induce apop
sscApoptosisProb = 0.7651      # probability of ssc apoptosis with apop signal  
VEGFblockApop = 80.6055             # Amount of VEGF required to block SSC apoptosis
SSCdivisionThreshold = 126.6268       # TNF + VEGF - TGF has to be greater than this for ssc to divide 
SSCdiffThreshold = 57.9870           # IL-10 -HGF -TNF -TGF has to be greater than this for ssc diff
sscDivideProb = 0.0972             # probability of SSC to divide if signal is not present
sscDiffProb = 0.6136                 # probability for SSC to differntiate without signal
fuseProb = 0.6034                  # Probability of SSC fusing even if the collagen amount is still low 
quiescentThreshold = 36.3109        # if HGF is lower than this at that location the ssc will return to quiescence 

# Fiber/Necrosis/ECM Variables 
#necBreakdownLag = 50            # length of time after that neutrophils and macrophages secrete cytokines as damage is removed
lowCollagenCutoff = 0.5         # collagen required to place fiber or new capaillariy 
highCollagenCutoff = 10        # fibrotic collagen amount
necrosisSecretionVal = 21.9514            # amount of hgf and TGF secreted by necrotic cells per step 
VEGFsecretionValFib = 0.0498        # VEGF secreted by fibers per step
lmRepel = -10                  # lambda to keep cells off healthy fibers
#chanceMergeECM = 10             # 10% change that two low collagen ECM elements next to each other merge
global lowColIdx 
lowColIdx = index.Index()       # Index to store low collagen ECM 
NecroticRadius = 10
NecroticFillRatio = .4

# Lymphatic Variables 
lympForce = 100 #  Force pulling ssc, neutrophils, and macrophages into lymph

# Microvessel Variables 
# capRecovTime = 48 * timeconverter       # capillaries sprout and elongate from surviving microvessel fragments 2-3 dpi (Jacobsen et al. 2021)
reperfusionTime = 120 * timeconverter   # amount of time for reestablished perfused networks is 5 dpi (Jacobsen et al. 2021)
VEGF_MMPthreshold = 469.0449                # Level of VEGF to trigger sprouting/elongation from surviving fragments 
capVEGFUptakeMax = 23.8174                  # Max uptake of VEGF by capillaries 
#capVEGFUptakeRel = 0.3                  # Relative uptake of VEGF by capillaries
MMPsecretionValCap = 8.2444                 # MMP secreted as capillary is regenerating/sprouting 
sproutingTime = 72 * timeconverter      # endothelial sprouts appear within 2-3 dpi (Jacobsen et al. 2021)
capPlacementThreshold =  34.6346             # threshold for if a capillariy will be placed/distance required bewteen fiber and cap
MinCapDist = 9.6060                         # minimum distance between capillary and another capillary
NumFibers = 1.3899                           # number of fibers nearby must be higher than this to check for capillary distances
sproutProb = 0.1566                        # probability of sprouting 
reperfusionLag = 0.8181                    # probability of reperfusing 

# Fibroblast variables
fibroblastToInitialFiber = 2 # Ratio of intial fibroblasts to fiber cells = 1 per 2 fibers (50 fiber cells)
lmFibro = 23  # lambda of fibroblast chemotaxis along tgf-b 
secretionUptakeValFibro = 9.0436 # cytokine secretion from fibroblasts per mcs also max uptake value
collagenSecretionValFibro = 0.1388 # collagen secretion from fibroblasts per mcs WHEN ABOVE ECM
fibActivationTime = math.ceil(8 * timeconverter) # same as ssc
fibDivisionTime = sscDivisionTime/10 # Division time is known to be less than ssc (Ichim et al. 2018)
apopTime = 12*timeconverter     # Time after apop singal that fibroblasts die (from Kelley's code)

#fibroRecruitProb = 0.5 # chance that fibroblast is recruited 
distNearSSCdiv = 16.5776 # if fibroblast is within this distance of a dividing ssc, it will divide (with some probability)
#recruitmentProportionFibro = 0.0006 # proportion of fibroblast recruited relatvie to TGF
TGFActivationThreshold = 40.3643 # level of TGF reguired to activate fibroblast
TGFMyoThreshold = 536.7525       # level of sustained TGF required to turn fibroblast into myofibroblast 
reqTimeExposed = 24*timeconverter # time required to be exposed to TGF beofre transitioning to myofibroblast
#fibroblastActivationProb = 0.2 # probablility of fibroblast activation if tgf threshold is met
#TGFProliferationThreshold = 15 # tgf level required for fibroblast proliferation
#fibroblastProliferationProb = 0.5895 # probability of fibroblast proliferation if threshold is met or if SSCs proliferating
fibroDivisionChanceSubseq = [1,0.25,0.06,0.02] # chance of division decreases with each cell division 
fibroblastTNFApoptosisThreshold = 310.5323 # level of TNF to initiate fibroblast apoptosis
TGFBlockApoptosisThreshold = 848.3706 # level of tgfb required to block TNF-initiated apoptosis
fibroblastApoptosisProb = 0.8423 # probability of fibroblast apoptosis if TNF and TGFB conditions are met 
forceFibroToECM = 500          # force pulling fibroblast to ECM 

# Cluster Variables
ifPythonCall = 0
ifDataSave = 1
ifPlot = 1

InitialDamagePercent = .67

# Perturbation variables
neuDeplete = 0
macDeplete = 0 
M2PolDir = 0
defAngio = 0
IL10_KO = 0
TNF_KO = 0
MCP_KO = 0
VEGFhalfDiff = 0
VEGFdoubleDiff = 0
if neuDeplete == 1:
    neutroRecruitmentProportion = 0
if macDeplete == 1:
    macroRecruitmentProportion = 0
if M2PolDir == 1: 
    IL10TransThreshold = IL10TransThreshold/100            # Local IL-10 that can induce M1 to M2 transition 
if defAngio == 1:
    VEGF_MMPthreshold = VEGF_MMPthreshold * 5


class ImportInitializerSteppable(SteppableBasePy): # Only run to segment histology image 
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):
        fileName = 'Insert path to csv'
        histoData = pd.read_csv(fileName, header=None)
        upscale = True
        if not upscale:
            for i in range(len(histoData)):
                x = histoData.iloc[i,0]
                y = histoData.iloc[i,1]
                tissueType = histoData.iloc[i,2]
                
                # if tissueType == 1: 
                    # cell_ecm = self.new_cell(self.WALL)
                    # self.cellField[x.tolist(),y.tolist(),0] = cell_ecm
                    
                # if tissueType == 3: # Fiber
                    # cell_fib = self.new_cell(self.FIBER)
                    # self.cellField[x.tolist(),y.tolist(),0] = cell_fib

                if tissueType == 2: # ECM 
                    cell_ecm = self.new_cell(self.WALL)
                    self.cellField[x.tolist(),y.tolist(),0] = cell_ecm
                
        if upscale:
            for i in range(len(histoData)):
                x = histoData.iloc[i,0].tolist()
                y = histoData.iloc[i,1].tolist()
                tissueType = histoData.iloc[i,2]
                
                xAdjusted = (x-1)*3 +1
                yAdjusted = (y-1)*3 +1
                
                if tissueType == 1: 
                    cell_ecm = self.new_cell(self.WALL)
                    for i in range(xAdjusted, xAdjusted+3):
                        for j in range(yAdjusted, yAdjusted+3):
                            self.cellField[i,j,1] = cell_ecm
                    
                # if tissueType == 3: # Fiber
                    # cell_fib = self.new_cell(self.FIBER)
                    # for i in range(xAdjusted, xAdjusted+2):
                        # for j in range(yAdjusted, yAdjusted+2):
                            # self.cellField[i,j,0] = cell_fib

                # if tissueType == 2: # ECM 
                    # cell_fib = self.new_cell(self.ECM)
                    # for i in range(xAdjusted, xAdjusted+3):
                        # for j in range(yAdjusted, yAdjusted+3):
                            # self.cellField[i,j,0] = cell_fib
        ## each square to become 3 by 3
    
    def step(self,mcs):
        return  

class MitosisSteppable(MitosisSteppableBase): # Only run to divide up fibers from orginial fiber segmentation image
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)
    
    def step(self, mcs):
        if mcs == 1:
            i = 0
            while i < 6: #randomly divide fiber fasicles into smaller regions for necrosis function
                cells_to_divide=[]
                for cell in self.cell_list_by_type(self.FIBER):
                    if cell.volume > 30:
                        cells_to_divide.append(cell)

                for cell in cells_to_divide:
                    self.divide_cell_random_orientation(cell)
                i = i+1

    def update_attributes(self):
        # reducing parent target volume
        self.parent_cell.targetVolume /= 2.0 
        self.parent_cell.lambdaVolume/= 2.0
        self.parent_cell.fluctAmpl = 50
        self.clone_parent_2_child()        

        # for more control of what gets copied from parent to child use cloneAttributes function
        self.clone_attributes(source_cell=self.parent_cell, target_cell=self.child_cell, no_clone_key_dict_list=[])

class ConstraintInitializerSteppable(SteppableBasePy): # Initialize properties of ecm and fiber imported with piff
    def __init__(self,frequency=1):
        SteppableBasePy.__init__(self,frequency)

    def start(self):
        # Initialize volume target and lambda for ecm and fiber imported with piff
        for cell in self.cell_list_by_type(self.FIBER):
            cell.targetVolume = cell.volume     # Set target volume equal to volume defined by piff
            cell.lambdaVolume = 10000           # Has to be large to have stiff fibers
            cell.dict['repair'] = 1             # 0 = needs repair, 1 = healthy, start all healthy 
            cell.dict['time2mature'] = -1       # counter for maturation lag for myotube after myocyte myocyte fusion
            VEGFsecretor = self.get_field_secretor("VEGF")
            VEGFsecretor.secreteOutsideCellAtBoundary(cell, VEGFsecretionValFib) # VEGF secreted by fibers
        
        for cell in self.cell_list_by_type(self.ECM):
            cell.targetVolume = cell.volume + 1 # Set target volume to slighly above that defined by piff
            cell.lambdaVolume = 10000           # Has to be large to have stiff ecm
            cell.dict['collagen'] = 1           # Define healthy collagen level 
            cell.dict['repair'] = 1             # 0 = needs repair, 1 = healthy, start all healthy
                                  
class NecrosisSteppable(SteppableBasePy): # Initialize muscle injury
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        # calculate total volume in pixels of fibers before necrosis happens. loop through all fiber cells, add their volume in pixels to FiberVolume
        # size of cell will be 3x3x1
        
        self.field = CompuCell.getConcentrationField(self.simulator, "HGF")
        
        return

    def step(self, mcs):
        if mcs == 0:
            xcoord = np.random.randint(50,250)
            ycoord = np.random.randint(50,350)#bounds restricted to always pick somewhere randomly in the tissue


            InitialFiber = 0
            for cell in self.cell_list_by_type(self.FIBER):
                InitialFiber += cell.volume
            FiberDamage = 0

            NecroticRadius = 10
            #print('starting while loop')
            while FiberDamage < InitialDamagePercent:
                for cell in self.cell_list_by_type(self.FIBER):
                    distance = np.sqrt((cell.xCOM-xcoord)**2 + (cell.yCOM-ycoord)**2)
                    if distance <= NecroticRadius:
                        #totally necrotic
                        cell.type = self.NECROTIC
                        cell.dict['repair'] = 0 # 0 for needs repair, 1 for repaired
                        cell.dict['capillary'] = 0 # 1 for necrotic capillary, 0 for necrotic fiber
                        #print('Dead')

                cluster2necrotic = []      
                for key,val in fiberGroups.items():
                    ClusterVolume = 0
                    NecroticVolume = 0
                    for cell in val:
                        ClusterVolume += cell.volume
                        if cell.type == self.NECROTIC:
                            NecroticVolume += cell.volume
                    ratio = NecroticVolume/ClusterVolume
                    if ratio >= NecroticFillRatio:
                        cluster2necrotic.append(key)
                        #print('ratio = ',ratio)
                        
                for cluster in cluster2necrotic:
                    for cell in fiberGroups[cluster]:
                        if cell.type != self.NECROTIC:
                            cell.type = self.NECROTIC
                            cell.dict['repair'] = 0 # 0 for needs repair, 1 for repaired
                            cell.dict['capillary'] = 0 # 1 for necrotic capillary, 0 for necrotic fiber
                            #print('FilledDeadCluster')
                
                CurrentMuscleHealth = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CurrentMuscleHealth += cell.volume
                FiberDamage = (InitialFiber-CurrentMuscleHealth)/InitialFiber
                NecroticRadius += 1

            #print('FiberDamagePercent= ',FiberDamage)
                # print('NecroticRadius+1= ',NecroticRadius)
        
        # Necrotic cells secrete HGF (proxy for nearby capillary burst bringing in HGF) and TGF
        HGFsecretor = self.get_field_secretor("HGF")
        TGFsecretor = self.get_field_secretor("TGF")
        if mcs %20 == 0:
            for cell in self.cell_list_by_type(self.NECROTIC):
                HGFsecretor.secreteOutsideCellAtBoundary(cell, necrosisSecretionVal) 
                TGFsecretor.secreteOutsideCellAtBoundary(cell, necrosisSecretionVal) 
            
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return

class NeutrophilSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        return

    def step(self, mcs):  
        if mcs == 0:
            # get inital necrosis to reference for neutrophil recruitment
            global intFiberNecrosis
            intFiberNecrosis = 0
            for cell in self.cell_list_by_type(self.NECROTIC):
                if cell.dict['capillary'] == 0: # necrotic fiber 
                    intFiberNecrosis += 1  

        #define neutrophil uptaking the HGF field
        HGFsecretor = self.get_field_secretor("HGF")
        
        # Number of recuited neutrophils is proportional to number of necrotic fiber cells
        fieldVal = [] 
        global fiberNecrosis
        fiberNecrosis = 0
        #if mcs %4 == 0:
        for cell in self.cell_list_by_type(self.NECROTIC):
            if cell.dict['capillary'] == 0: # necrotic fiber 
                fiberNecrosis += 1
        numRecruited = np.ceil(neutroRecruitmentProportion * fiberNecrosis)
        numRecruited = numRecruited.astype(int)              
         
        # Neutrophils transported to injury site from healthy capillaries as long as there is enough necrosis  
        global necrosisRemain
        necrosisRemain = fiberNecrosis/intFiberNecrosis # get amount necrosis relative to initial 
        if necrosisRemain > necrosisThreshold and numRecruited > 0:
            _cap_locs = [[c.xCOM, c.yCOM] for c in self.cell_list_by_type(self.CAPILLARY) if c.dict['fragmented'] == 2]
            CapillaryLocations = np.array(_cap_locs) if _cap_locs else np.empty([0, 2])
            if CapillaryLocations.shape[0] > 0: # guard against no healthy capillaries, which would otherwise spin forever below
                numRecruited = min(numRecruited, CapillaryLocations.shape[0]) # can't recruit more than there are capillaries to draw from
                fieldVal = [self.field.HGF[int(x), int(y), 1] for x, y in CapillaryLocations]
                indexTopRegion = np.argsort(fieldVal)[-numRecruited:]  # sorts to get top numbers (going from the end because sorts low to high)
                toPlace = CapillaryLocations[indexTopRegion] # where neutrophils get placed
            else:
                toPlace = np.empty([0, 2])

            # Place neutrophils if cell doesn't exist in that location
            for x,y in toPlace:
                cell = self.cellField[x.item(),y.item(),1]
                if not cell:
                    Neutrophil = self.new_cell(self.NEUTROPHIL)
                    self.cell_field[x.item(), y.item(), 1] = Neutrophil
                    Neutrophil.targetVolume = 12
                    Neutrophil.lambdaVolume = 50
                    #set neutrophil dictionary terms
                    Neutrophil.dict['Birth'] = mcs
                    Neutrophil.dict['NumPhagocytosed'] = 0
                    Neutrophil.dict['Apoptotic'] = 0
                    Neutrophil.dict['secretionCountdown'] = -1
                    Neutrophil.dict['phagocytoseLagTime'] = 0
                    HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(Neutrophil, secretionUptakeValNeu, uptakeRel) #rel is a number between 0-1
              
        # Convert necrosis to low collagen ECM and secrete MCP and MMP and TNF when eating 
        MCPsecretor = self.get_field_secretor("MCP")
        MMPsecretor = self.get_field_secretor("MMP")
        HGFsecretor = self.get_field_secretor("HGF")
        TNFsecretor = self.get_field_secretor("TNF")
        #if mcs > TimeNeuStart:        
        for cell in self.cell_list_by_type(self.NEUTROPHIL):
            CorrespondingNecrotic = self.cell_field[cell.xCOM, cell.yCOM, cell.zCOM]
            # If cell below is necrotic convert to low collagen ecm
            if CorrespondingNecrotic and CorrespondingNecrotic.type == self.NECROTIC and CorrespondingNecrotic.dict['capillary'] == 0:
                #cell.dict['secretionCountdown'] = necBreakdownLag # countdown to turn off mcp and mmp secretion
                if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                    CorrespondingNecrotic.type = self.ECM
                    CorrespondingNecrotic.dict['repair'] = 0
                    CorrespondingNecrotic.dict['collagen'] = 0.1
                    global lowColIdx
                    lowColIdx.insert(CorrespondingNecrotic.id, (CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM, CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM)) # adding low collagen to the index                  
                    #print('added', CorrespondingNecrotic.id)
                    cell.dict['NumPhagocytosed'] += 1
                    cell.dict['phagocytoseLagTime'] = 0 # reset phagocytosis lag counter
                    # stop secretion once phagocytosis is done
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                    MCPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                    TNFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                    cdHGF = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                    cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                    if cdHGF:
                        cdHGF.setLambda(lmNeu)
                    if cdRep:
                        cdRep.setLambda(lmRepel)
                    #define neutrophil uptaking the HGF field
                    HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValNeu, uptakeRel) #rel is a number between 0-1
                else:
                    # secrete while phagocytosing 
                    MCPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValNeu)
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValNeu)
                    TNFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValNeu)
                    HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) # turn off uptake when not chemotaxing 
                    # stop movement
                    cdHGF = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                    if cdHGF:
                        cdHGF.setLambda(0)
                    cell.dict['phagocytoseLagTime'] += 1
                
        #Make neutrophils apoptotic when appropriate 
        # for cell in self.cell_list_by_type(self.NEUTROPHIL):
            if cell and (mcs - cell.dict['Birth']) >= neutroDeathTime or cell.dict['NumPhagocytosed'] >= phagoThresholdforNeuM2:
                cell.dict['Apoptotic'] = 1 
                # stop movement
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                if cd:
                    cd.setLambda(0)

class SSCSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        # Track cell type or differentiation level of ssc for visualization
        self.track_cell_level_scalar_attribute(field_name='cellType', attribute_name='cellType')

    def start(self):
        # Initialize SSC (1 ssc per 4 fibers)
        global num_cells 
        # num_cells = np.round(num_fiber_cells * sscToInitialFiber)
        num_cells = math.floor(numFiberCells/sscToInitialFiber) 

        global newMyotube
        newMyotube = 0
        
        # Generate given number of cells where other cells don't already exist
        for i in range(num_cells):
            isPlaced = False
            
            while not isPlaced:
                xrand = np.random.randint(1,self.dim.x)
                yrand = np.random.randint(1, self.dim.y)
                
                cell = self.cellField[xrand,yrand,1]
                if not cell:
                    ssc = self.new_cell(self.SSC)
                    ssc.targetVolume = 10
                    ssc.lambdaVolume = 50
                    ssc.dict['activationState'] = 0 # 0 = nonactive, 1 = active
                    ssc.dict['time2activate'] = -1 # counter for activation lag
                    ssc.dict['time2divide'] = -1 # counter for division lag
                    ssc.dict['time2diff'] = -1 # counter for differentiation lag
                    ssc.dict['numDiv'] = 0 # number of times ssc has divided 
                    ssc.dict['cellType'] = 0 # 0 = SSC, 1 = myoblast, 2 = myocyte
                    ssc.dict['moveCount'] = 0 # number of moves without being above low collagen ECM
                    
                    self.cell_field[xrand, yrand, 1] = ssc
                    isPlaced = True  

    def step(self, mcs):
        # define SSC secretion/uptake for HGF, MMP, MCP, and VEGF
        HGFsecretor = self.get_field_secretor("HGF")
        MMPsecretor = self.get_field_secretor("MMP")
        MCPsecretor = self.get_field_secretor("MCP")
        VEGFsecretor = self.get_field_secretor("VEGF")
        TNFsecretor = self.get_field_secretor("TNF")
        # Calculate mean and max values of fields
        HGFfield = self.field.HGF
        MMPfield = self.field.MMP
        VEGFfield = self.field.VEGF
        TNFfield = self.field.TNF
        TGFfield = self.field.TGF
        IL10field = self.field.IL10

        if mcs % SSCRecruitmentFreq == 0 and necrosisRemain > macRecruitStop:
            _sx = np.random.randint(0, self.dim.x-1, 500)
            _sy = np.random.randint(0, self.dim.y-1, 500)
            HGFMean = float(np.mean([HGFfield[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))
            MMPMean = float(np.mean([MMPfield[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))
            TGFMean = float(np.mean([TGFfield[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))
             
            # Recruit SSC proportional to sum of mean HGF + MMP - TGF every time step
            numRecruited = np.ceil(recruitmentProportionSSC * (HGFMean + MMPMean - TGFMean))
            numRecruited = numRecruited.astype(int)

            # Randomly select spots and check to make sure potential spots have low enough repel
            fieldVal = []
            fieldRepel = self.field.REPEL
            coord = np.empty([0 , 2])
            numRecruitedTemp = numRecruited * recruitMultiplier
            while numRecruitedTemp > 0:
                xcoord1 = np.random.randint(1,self.dim.x)
                ycoord1 = np.random.randint(1,self.dim.y)
                coord1 = np.array([xcoord1, ycoord1])
                
                valueRepel = fieldRepel[xcoord1, ycoord1, 1] # Temp fix for layers
                if valueRepel < 0.1:
                  coord = np.vstack((coord,coord1))
                  numRecruitedTemp -= 1 
                  
            
            # Prioritize locations with high HGF
            for x,y in coord:
                fieldVal.append(self.field.HGF[x.item(), y.item(), 1])
            indexTopRegion = np.argsort(fieldVal)[-numRecruited:]  # sorts to get top numbers (going from the end because sorts low to high)
            toPlace = coord[indexTopRegion]
            for x,y in toPlace:
                cell = self.cellField[x.item(),y.item(),1]
                if not cell:
                    SSC = self.new_cell(self.SSC)
                    self.cell_field[x.item(), y.item(), 1] = SSC
                    SSC.targetVolume = 10
                    SSC.lambdaVolume = 50
                    SSC.dict['activationState'] = 1 # 0 = nonactive, 1 = active
                    SSC.dict['time2activate'] = -1 # counter for activation lag
                    SSC.dict['time2divide'] = -1
                    SSC.dict['time2diff'] = -1
                    SSC.dict['numDiv'] = 0 # number of times ssc has divided
                    SSC.dict['cellType'] = 0 # 0 = SSC, 1 = myoblast, 2 = myocyte
                    SSC.dict['moveCount'] = 0
                    # Remove some MMP and HGF to simulate the binding event that recruited the SSC             
                    HGFvalue = HGFfield[SSC.xCOM, SSC.yCOM, SSC.zCOM]
                    if HGFvalue >= secretionUptakeValSSC: # prevents HGF field from being negative
                        HGFfield[SSC.xCOM, SSC.yCOM, SSC.zCOM] = HGFvalue - secretionUptakeValSSC
                    MMPvalue = MMPfield[SSC.xCOM, SSC.yCOM, SSC.zCOM]
                    if MMPvalue >= secretionUptakeValSSC: # prevents HGF field from being negative
                        MMPfield[SSC.xCOM, SSC.yCOM, SSC.zCOM] = MMPvalue - secretionUptakeValSSC                   
                
        # Actions
        _ssc_count = len(self.cell_list_by_type(self.SSC))
        for cell in self.cell_list_by_type(self.SSC):
            # simulates E2 binding to ER on SSC
            
            if cell.dict['activationState'] == 1 and cell.dict['time2divide'] < 0 and cell.dict['time2diff'] < 0 and cell.dict['time2activate'] < 0:
                if mcs % 2 == 0:
                    VEGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValSSC) # VEGF secreted from activated SSC
                    MCPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValSSC) # MCP secreted from activated SSC
                
                cellBelow = self.cell_field[cell.xCOM, cell.yCOM, 0]

                if cellBelow: #and cellBelow is not None:
                    # check for neighboring fiber 
                    fiberPresent = False
                    for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cellBelow):
                        if neighbor and neighbor.type == self.FIBER and 'time2mature' in neighbor.dict: 
                            # checking if the neighbor is an immature myotube and increasing the maturation time if it is
                            if neighbor.dict['time2mature'] >-1: 
                                neighbor.dict['time2mature'] -= 1
                                # print('mature time for myotube is ',  neighbor.dict['time2mature'])
                            if neighbor.dict['time2mature'] < 0:  # checking that neighbor is a mature fiber before fusing, dividing, or differentiating
                                fiberPresent = True
                            
                        
                    stopMoving = False
                    # If on fiber edge stop movement and divide or differentiate. If myocyte can fuse to fiber edge
                    if fiberPresent:   
                        TNFval = TNFfield[cell.xCOM, cell.yCOM, cell.zCOM]
                        VEGFval = VEGFfield[cell.xCOM, cell.yCOM, cell.zCOM]
                        TGFval = TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] 
                        IL10val = IL10field[cell.xCOM, cell.yCOM, cell.zCOM] 
                        HGFval = HGFfield[cell.xCOM, cell.yCOM, cell.zCOM]
                        # SSC or Myoblast dividing
                        if cell.dict['cellType'] == 1 or cell.dict['cellType'] == 0: 
                           # Sense environment to determine if dividing (referenced Kelley's 2018 paper)
                            
                            #print('division signal: TNF +VEGF -TGF', TNFval + VEGFval - TGFval)
                            if TNFval + VEGFval - TGFval > SSCdivisionThreshold or np.random.uniform() < sscDivideProb: 
                                if cell.dict['numDiv'] > len(sscDivisionChanceSubseq)-1:
                                    divIndex = 3
                                else: 
                                    divIndex = cell.dict['numDiv']
                                if np.random.uniform() < sscDivisionChanceSubseq[divIndex]:
                                    if TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdivisionThreshold/3): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                        TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] = TNFval - (SSCdivisionThreshold/3) # binding uptake (divided by 3 because it's taking from 3 cytokines)
                                    else:
                                        TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] = 0  
                                    if VEGFfield[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdivisionThreshold/3): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                        VEGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = VEGFval - (SSCdivisionThreshold/3) # binding uptake (divided by 3 because it's taking from 3 cytokines)
                                    else:
                                        VEGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = 0 
                                    if TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdivisionThreshold/3): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                        TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = TGFval - (SSCdivisionThreshold/3) # binding uptake (divided by 3 because it's taking from 3 cytokines)
                                    else:
                                        TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = 0 
                                    
                                    cell.dict['time2divide'] = sscDivisionTime
                                    stopMoving = True
                
                        # Differentiation can occur in active SSCs (myoblast) and myoblasts (diff to myocyte)
                        if (cell.dict['cellType'] == 0 and cell.dict['activationState'] == 1) or cell.dict['cellType'] == 1:
                            if 3*IL10val - HGFval - TNFval - TGFval > SSCdiffThreshold or np.random.uniform() <sscDiffProb:
                                # IL10field[cell.xCOM, cell.yCOM, cell.zCOM] = IL10val - (SSCdiffThreshold/4) # binding uptake (divided by 4 because it's taking from4 cytokines)
                                if IL10field[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdiffThreshold/4): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                    IL10field[cell.xCOM, cell.yCOM, cell.zCOM] = IL10val - (SSCdiffThreshold/4) # binding uptake 
                                else:
                                    IL10field[cell.xCOM, cell.yCOM, cell.zCOM] = 0 
                                if HGFfield[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdiffThreshold/4): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                    HGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = HGFval - (SSCdiffThreshold/4) # binding uptake 
                                else:
                                    HGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = 0 
                                if TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdiffThreshold/4): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                    TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = TGFval - (SSCdiffThreshold/4) # binding uptake 
                                else:
                                    TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = 0 
                                if TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] >= (SSCdiffThreshold/4): # check to prevent from becoming negative, if not enough there to remove, just remove whatever is there
                                    TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] = TNFval - (SSCdiffThreshold/4) 
                                else:
                                    TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] = 0
                                cell.dict['time2diff'] = sscDifferentiationTime
                                stopMoving = True

                        # Myocyte fuse to fiber
                        if cell.dict['cellType'] == 2: 
                            cellBelow = self.cell_field[cell.xCOM, cell.yCOM, 0]
                            # Can only grow fiber if there is enough ECM to stabilize the fiber but not fibrotic amounts
                            if (cellBelow and cellBelow.type == self.ECM and cellBelow.dict['repair'] == 0) and ((cellBelow.dict['collagen'] >= lowCollagenCutoff and cellBelow.dict['collagen'] <= highCollagenCutoff) or np.random.uniform() < fuseProb):
                                cellBelow.type = self.FIBER
                                self.remove_all_cell_fpp_links(cellBelow) # removes the link once repaired so fibroblast don't move towards
                                cellBelow.dict['repair'] = 1
                                #print('ssc fusion')
                                cell.targetVolume = 0 
                                if lowColIdx.count((cellBelow.xCOM, cellBelow.yCOM, cellBelow.xCOM, cellBelow.yCOM)) > 0: # checks that ECM is at that index and deletes from low collagen list
                                    lowColIdx.delete(cellBelow.id, (cellBelow.xCOM, cellBelow.yCOM, cellBelow.xCOM, cellBelow.yCOM)) 
                    
                    if stopMoving:
                        # Stop moving while waiting to divide/diff    
                        cdMMP = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                        if cdMMP:
                            cdMMP.setLambda(0)
                        MMPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)
                        cdTNF = self.chemotaxisPlugin.getChemotaxisData(cell, "TNF")
                        if cdTNF:
                            cdTNF.setLambda(0)
                        
                    # If not on fiber edge or dividing/diff, migrate toward MMP (but not on healthy fibers)
                    if not fiberPresent or not stopMoving:
                        cd = self.chemotaxisPlugin.getChemotaxisData(cell, "MMP")
                        if cd:
                            cd.setLambda(lmSSC)
                            MMPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValSSC, uptakeRel) # Uptake MMP as chemotaxing along to simulate binding during chemo
                        cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                        if cdRep:
                            cdRep.setLambda(lmRepel)
                            
                        # if myoblast move along TNF-alpha
                        if cell.dict['cellType'] == 1: # If it'a a myoblast chemotax along TNF and uptake as it moves along
                            cdTNF = self.chemotaxisPlugin.getChemotaxisData(cell, "TNF")
                            if cdTNF:
                                cdTNF.setLambda(lmSSC)
                                TNFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValSSC, uptakeRel) # Uptake MMP as chemotaxing along to simulate binding during chemo
                        

                # myocyte binding to other myocytes: must have neighboring myocyte and appropraite collagen density below
                if cell.dict['cellType'] == 2:
                    cellBelow = self.cell_field[cell.xCOM, cell.yCOM, 0]
                    # Can only fuse to form myotube if there is enough ECM to stabilize the fiber but not fibrotic amounts
                    if (cellBelow and cellBelow.type == self.ECM and cellBelow.dict['repair'] == 0) and ((cellBelow.dict['collagen'] >= lowCollagenCutoff and cellBelow.dict['collagen'] <= highCollagenCutoff) or np.random.uniform() < fuseProb):
                        listMyocytes = []
                        for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                            if neighbor and neighbor.type == self.SSC and neighbor.dict['cellType'] == 2:
                                # add to list
                                listMyocytes.append(neighbor)
                                #print('Monocyte 2 crit')
                                
                        # pick random
                        if len(listMyocytes) > 0:
                            neighbor = listMyocytes[random.randint(0,len(listMyocytes)-1)]
                            # myotube formation from myocyte fusion
                            cellBelow.type = self.FIBER
                            cellBelow.dict['time2mature'] = myotubeMatureTime + random.randint(0, myotubeMatureTime) # setting the required time for it to go from an immature myotube to a myofiber. Min is 1 up to 2 (stoachstic)
                            self.remove_all_cell_fpp_links(cellBelow) # removes the link once repaired so fibroblast don't move towards
                            cellBelow.dict['repair'] = 1
                            global newMyotube
                            newMyotube += 1
                            #print('myocyte to myocyte fusion', newMyotube)
                            # once fiber is placed below both myocytes gone from simulation 
                            cell.targetVolume = 0 
                            neighbor.targetVolume = 0 
                            if lowColIdx.count((cellBelow.xCOM, cellBelow.yCOM, cellBelow.xCOM, cellBelow.yCOM)) > 0: # checks that ECM is at that index and deletes from low collagen list
                                lowColIdx.delete(cellBelow.id, (cellBelow.xCOM, cellBelow.yCOM, cellBelow.xCOM, cellBelow.yCOM))
                            #print('Myocyte binding to myocyte')
                    
            # apoptosis
            TGFlevelCheck = TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] # Get amount of TGF at the cell
            HGFvalue = HGFfield[cell.xCOM, cell.yCOM, cell.zCOM] # Get amount of HGF at the cell
            #if (cell.dict['moveCount'] > sscApoptosisCutoff or TGFlevelCheck > sscTGFApoptosisThreshold) and np.random.uniform() < sscApoptosisProb: # can apop after certain number of moves without low collagen or from TGF signal
            if TGFlevelCheck > sscTGFApoptosisThreshold and np.random.uniform() < sscApoptosisProb:    
                VEGFlevelCheck = VEGFfield[cell.xCOM, cell.yCOM, cell.zCOM] # Get amount of VEGF at the cell
                MacroNeighbor = False
                for neighbor, common_surface_area in self.get_cell_neighbor_data_list(cell):
                    if neighbor and neighbor.type == self.MACROPHAGE:
                        MacroNeighbor = True
                if VEGFlevelCheck < VEGFblockApop and MacroNeighbor == False: # VEGF and MPs(through adhesion) can block SSC apop 
                    cell.targetVolume = 0
                    cell.lambdaVolume=1000000
                    MCPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                    VEGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                    MCPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                    TGFvalue = TGFfield[cell.xCOM, cell.yCOM, cell.zCOM]
                    TGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = TGFvalue - sscTGFApoptosisThreshold # remove TGF from binding that induced apop
                else: # if there is enough VEGF to block SSC apop then remove that VEGF to simulate binding that occured to block apop
                    if VEGFlevelCheck >= VEGFblockApop: # prevents VEGF field from being negative
                        VEGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = VEGFlevelCheck - VEGFblockApop
                        
            elif HGFvalue < quiescentThreshold:
                cell.dict['activationState'] = 0   
                VEGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                MCPsecretor.secreteOutsideCellAtBoundary(cell, 0)
            
            if cell.dict['activationState'] == 0:
                # attempt to activate by looking for growth factor
                HGFvalue = HGFfield[cell.xCOM, cell.yCOM, cell.zCOM]
                if HGFvalue >= sscActivationThreshold:
                    cell.dict['time2activate'] = sscActivationTime
                    HGFfield[cell.xCOM, cell.yCOM, cell.zCOM] = HGFvalue - sscActivationThreshold # HGF uptake from binding required to activate SSC
                else: # Check if quiescent and not part of the min number in the ssc pool and use probability
                    if np.random.randint(int(quiescentProb)) == 0 and _ssc_count > num_cells: #% of the time that inactive ssc does not have HGF for activation it will apop
                        cell.targetVolume = 0
                        cell.lambdaVolume=1000000
                    
                    
            # Increment lag counters 
            cell.dict['time2divide'] -= 1
            #print('ssc division countdown', cell.dict['time2divide'])
            cell.dict['time2diff'] -=1  
            cell.dict['time2activate'] -=1

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return       

class SSCMitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)
    
    def step(self, mcs):
        # loop through all ssc and if lag times are over then divide or differentiate
        _all_fibros = list(self.cell_list_by_type(self.FIBROBLAST))
        for cell in self.cell_list_by_type(self.SSC):
            # divide cell with possibility of changing type
            if cell.dict['time2divide'] == 0:
                self.divide_cell_random_orientation(cell)
                cell.dict['numDiv'] += 1 # count number of ssc division to update prob of another division occuring
                # if SSC is proliferating it triggers the nearby fibroblast to proliferate (SSC fibroblast coupled behavior)
                fibros = _all_fibros
                if fibros:
                    cx, cy = cell.xCOM, cell.yCOM
                    fibro_pos = np.array([[f.xCOM, f.yCOM] for f in fibros])
                    dists = np.sqrt((fibro_pos[:,0]-cx)**2 + (fibro_pos[:,1]-cy)**2)
                    divTime = fibDivisionTime*(1-cell.dict['activationState']) + fibDivisionTime*cell.dict['activationState']
                    for fibro, dist in zip(fibros, dists):
                        divIndex = min(fibro.dict['numDiv'], len(fibroDivisionChanceSubseq)-1)
                        if dist < distNearSSCdiv and fibro.dict['time2divide'] < 0 and np.random.uniform() < fibroDivisionChanceSubseq[divIndex]:
                            fibro.dict['time2divide'] = divTime
                            fibro.dict['numDiv'] += 1
                        elif fibro.dict['time2divide'] < 0 and dist < distNearSSCdiv*10 and fibro.dict['numDiv'] < 1:
                            fibro.dict['time2divide'] = divTime
                            fibro.dict['numDiv'] += 1
                    
            # Turn activated SSC into myoblast
            if cell.dict['cellType'] == 0 and cell.dict['time2diff'] == 0: 
                cell.dict['cellType'] = 1

            # Turn myoblast into myocyte 
            if cell.dict['cellType'] == 1 and cell.dict['time2diff'] == 0: 
                cell.dict['cellType'] = 2

            if cell.dict['time2activate'] == 0: 
                cell.dict['activationState'] = 1
                
    def update_attributes(self):
        # Clone cell
        self.clone_parent_2_child()   
        
        # 50 percent chance same type, 50 percent change next type (SSC->blast,blast->cyte)
        self.child_cell.dict['cellType'] = np.ceil(2*np.random.uniform()) -1 + self.parent_cell.dict['cellType']
        self.parent_cell.dict['cellType'] = np.ceil(2*np.random.uniform()) -1 + self.parent_cell.dict['cellType']
        
class MacrophageSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        # Initialize resident macrophages       
        num_cells = math.floor(numFiberCells*macToInitialFiber)
        
        for i in range(num_cells):
            isPlaced = False
            
            while not isPlaced:
                xrand = np.random.randint(1,self.dim.x)
                yrand = np.random.randint(1, self.dim.y) 
                # Check to make sure there is no cell in that random spot
                cell = self.cellField[xrand,yrand,1]
                if not cell:
                    Macrophage = self.new_cell(self.MACROPHAGE)
                    Macrophage.dict['type'] = 4 # 0 for monocyte, 1 for M1, 2 for M2, 4 for resident 
                    Macrophage.targetVolume = 21
                    Macrophage.lambdaVolume = 50
                    Macrophage.dict['Birth'] = 0
                    Macrophage.dict['prolifCount'] = 0
                    Macrophage.dict['NumPhagocytosed'] = 0
                    #Macrophage.dict['DeathIterator'] = MacrophageRandomDeath
                    Macrophage.dict['phagocytoseLagTime'] = 0
                    Macrophage.dict['secretionCountdown'] = -1
                    self.cell_field[xrand, yrand, 1] = Macrophage
                    isPlaced = True
        return
        
    def step(self, mcs):
        # define fields
        TNFfield = self.field.TNF
        IL10field = self.field.IL10
        # define macrophage secretion/uptake of HGF, MCP, TGF, VEGF, MMP, TNF, IL-10 and E2
        TGFsecretor = self.get_field_secretor("TGF") 
        MCPsecretor = self.get_field_secretor("MCP")
        VEGFsecretor = self.get_field_secretor("VEGF")
        HGFsecretor = self.get_field_secretor("HGF")
        E2secretor = self.get_field_secretor("E2")
        MMPsecretor = self.get_field_secretor("MMP")
        TNFsecretor = self.get_field_secretor("TNF") 
        IL10secretor = self.get_field_secretor("IL10") 
                            
        # Macrophage recruitment proportional to mean level of MCP
        if mcs %6 == 0:
            MCPfield = self.field.MCP
            TGFfield = self.field.TGF
            _sx = np.random.randint(0, self.dim.x-1, 500)
            _sy = np.random.randint(0, self.dim.y-1, 500)
            MCPMean = float(np.mean([MCPfield[int(x), int(y), 1] for x, y in zip(_sx, _sy)]))
            if MCPMean > macRecruitThres and necrosisRemain > macRecruitStop:
                numRecruited = np.ceil(macroRecruitmentProportion * MCPMean)
                numRecruited = numRecruited.astype(int)
                      
                # Monocytes transported to injury site from healthy capillaries
                _cap_locs = [[c.xCOM, c.yCOM] for c in self.cell_list_by_type(self.CAPILLARY) if c.dict['fragmented'] == 2] # Only get location of capillaries that are healthy/perfusing
                CapillaryLocations = np.array(_cap_locs) if _cap_locs else np.empty([0, 2])
                if CapillaryLocations.shape[0] > 0: # guard against no healthy capillaries, which would otherwise spin forever below
                    numRecruited = min(numRecruited, CapillaryLocations.shape[0]) # can't recruit more than there are capillaries to draw from
                    CapillaryLocations = CapillaryLocations[:numRecruited]
                else:
                    CapillaryLocations = np.empty([0, 2])

                # Place monocyte if cell doesn't exist in that location 
                for x,y in CapillaryLocations:
                    cell = self.cellField[x.item(),y.item(),1]
                    if not cell:
                        Macrophage = self.new_cell(self.MACROPHAGE)
                        self.cell_field[x.item(), y.item(), 1] = Macrophage
                        Macrophage.dict['type'] = 0 # 0 for monocyte, 1 for M1, 2 for M2, 4 for resident 
                        Macrophage.dict['monoM1TransTime'] = mcs + np.random.normal(meanM1trans,stdM1trans) # time for monocyte to convert to M1 (1-3 days using a gaussian distribution centered around 2, std .5 days)
                        Macrophage.targetVolume = 8.5
                        Macrophage.lambdaVolume = 50
                        Macrophage.dict['Birth'] = mcs
                        Macrophage.dict['secretionCountdown'] = -1
                        Macrophage.dict['prolifCount'] = 0
                        Macrophage.dict['NumPhagocytosed'] = 0
                        Macrophage.dict['phagocytoseLagTime'] = 0 
                        MCPvalue = MCPfield[Macrophage.xCOM, Macrophage.yCOM, Macrophage.zCOM] 
                        MCPfield[Macrophage.xCOM, Macrophage.yCOM, Macrophage.zCOM] = MCPvalue - macRecruitThres # remove corresponding with bidning evenets with MCP that recruited monocyte
                
        # Behaviors for each macrophage type  
        for cell in self.cell_list_by_type(self.MACROPHAGE):
            if cell.dict['type'] == 4: # if resident macrophage 
                # Resident macrophages secrete MMP and TNF and MCP
                MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                TNFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                MCPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                # Phagocytose necrosis
                CorrespondingNecrotic = self.cell_field[cell.xCOM, cell.yCOM, 0]
                if CorrespondingNecrotic and CorrespondingNecrotic.type == self.NECROTIC and CorrespondingNecrotic.dict['capillary'] == 0:
                    #cell.dict['secretionCountdown'] = necBreakdownLag
                    if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                        CorrespondingNecrotic.type = self.ECM
                        CorrespondingNecrotic.dict['repair'] = 0
                        CorrespondingNecrotic.dict['collagen'] = 0.1
                        global lowColIdx
                        lowColIdx.insert(CorrespondingNecrotic.id, (CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM, CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM)) # adding low collagen to the index
                        cell.dict['NumPhagocytosed'] += 1
                        cell.dict['phagocytoseLagTime'] = 0 # reset phagocytosis lag counter
                        # stop phagocytosis specific secretion once phagocytosis is done
                        HGFsecretor.secreteOutsideCellAtBoundary(cell, 0) 
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                        IL10secretor.secreteOutsideCellAtBoundary(cell, 0)
                        # chemotax and uptake as migrating once phgocytosis is done 
                        cdHGF = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                        cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                        cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                        if cdHGF:
                            cdHGF.setLambda(lmMac)
                        if cdMCP:
                            cdMCP.setLambda(lmMac)
                        if cdRep:
                            cdRep.setLambda(lmRepel)
                        HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel) 
                        MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
               
                    else:
                        # secrete while phagocytosing 
                        HGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac) 
                        MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        # stop movement while phagocytosing 
                        cdHGF = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                        cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                        HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) #turn off uptake while phagocytosing 
                        MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)
                        if cdHGF:
                            cdHGF.setLambda(0)
                        if cdMCP:
                            cdMCP.setLambda(0)    
                        cell.dict['phagocytoseLagTime'] += 1
                        
                # Eat apoptotic neutrophils
                for neighbor in self.get_cell_neighbor_data_list(cell):
                    if neighbor[0]:
                        if neighbor[0].type == self.NEUTROPHIL and neighbor[0].dict['Apoptotic'] == 1:
                            if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                                neighbor[0].targetVolume=0
                                neighbor[0].lambdaVolume=1000000
                                cell.dict['NumPhagocytosed'] += 1  
                                # stop phagocytosis specific secretion once phagocytosis is done
                                HGFsecretor.secreteOutsideCellAtBoundary(cell, 0) 
                                TGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                                IL10secretor.secreteOutsideCellAtBoundary(cell, 0)
                                cdHGF = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                                cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                                if cdHGF:
                                    cdHGF.setLambda(lmMac)
                                if cdMCP:
                                    cdMCP.setLambda(lmMac)
                                if cdRep:
                                    cdRep.setLambda(lmRepel)
                                HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel) 
                                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                            else:
                                # secrete while phagocytosing 
                                HGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac) 
                                MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                #print('IL-10 secreted from monocytes')
                                # stop movement while phagocytosing 
                                cdHGF = self.chemotaxisPlugin.getChemotaxisData(cell, "HGF")
                                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                                HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) #turn off uptake while phagocytosing 
                                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)
                                #E2secretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) 
                                if cdHGF:
                                    cdHGF.setLambda(0)
                                if cdMCP:
                                    cdMCP.setLambda(0)
                                cell.dict['phagocytoseLagTime'] += 1

            if cell.dict['type'] == 0: # if monocyte
                #chemotax and uptake MCP, VEGF, and TGF
                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                cdVEGF = self.chemotaxisPlugin.getChemotaxisData(cell, "VEGF")
                cdTGF = self.chemotaxisPlugin.getChemotaxisData(cell, "TGF")
                cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                # Phagocytose necrosis
                CorrespondingNecrotic = self.cell_field[cell.xCOM, cell.yCOM, 0]
                if CorrespondingNecrotic and CorrespondingNecrotic.type == self.NECROTIC and CorrespondingNecrotic.dict['capillary'] == 0:
                    if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                        CorrespondingNecrotic.type = self.ECM
                        CorrespondingNecrotic.dict['repair'] = 0
                        CorrespondingNecrotic.dict['collagen'] = 0.1
                        lowColIdx.insert(CorrespondingNecrotic.id, (CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM, CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM)) # adding low collagen to the index
                        cell.dict['NumPhagocytosed'] += 1
                        cell.dict['phagocytoseLagTime'] = 0 # reset phagocytosis lag counter
                        # stop secretion once phagocytosis is done
                        HGFsecretor.secreteOutsideCellAtBoundary(cell, 0) 
                        MMPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                        IL10secretor.secreteOutsideCellAtBoundary(cell, 0)
                        # Return to chemotaxing and uptaking once phagocytosis is one 
                        if cdMCP:
                            cdMCP.setLambda(lmMono)
                        if cdVEGF:
                            cdVEGF.setLambda(lmMono)
                        if cdTGF:
                            cdTGF.setLambda(lmMono)
                        if cdRep:
                            cdRep.setLambda(lmRepel)
                        # Uptake of cytokines by monocytes           
                        TGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel) #rel is a number between 0-1
                        MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                        VEGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)

                    else:
                        # Secretion of cytokines by monocytes during phagocytosis         
                        HGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac) 
                        MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        #print('IL-10 secreted from monocytes')
                        # stop movement while phagocytosing 
                        TGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) #turn off uptake while phagocytosing 
                        MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)
                        VEGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) 
                        if cdMCP:
                            cdMCP.setLambda(0)
                        if cdVEGF:
                            cdVEGF.setLambda(0)
                        if cdTGF:
                            cdTGF.setLambda(0)
                        cell.dict['phagocytoseLagTime'] += 1

                else: 
                    # if no necrosis nearby, chemotax and uptake MCP, VEGF, and TGF
                    cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                    cdVEGF = self.chemotaxisPlugin.getChemotaxisData(cell, "VEGF")
                    cdTGF = self.chemotaxisPlugin.getChemotaxisData(cell, "TGF")
                    cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                    if cdMCP:
                        cdMCP.setLambda(lmMono)
                    if cdVEGF:
                        cdVEGF.setLambda(lmMono)
                    if cdTGF:
                        cdTGF.setLambda(lmMono)
                    if cdRep:
                        cdRep.setLambda(lmRepel)
                    # Uptake of cytokines by monocytes           
                    TGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel) #rel is a number between 0-1
                    MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                    VEGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                        
                # Eat apoptotic neutrophils
                for neighbor in self.get_cell_neighbor_data_list(cell):
                    if neighbor[0]:
                        if neighbor[0].type == self.NEUTROPHIL and neighbor[0].dict['Apoptotic'] == 1:
                            if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                                neighbor[0].targetVolume=0
                                neighbor[0].lambdaVolume=1000000
                                cell.dict['NumPhagocytosed'] += 1  
                                # stop phagocytosis specific secretion once phagocytosis is done
                                HGFsecretor.secreteOutsideCellAtBoundary(cell, 0) 
                                MMPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                                TGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                                IL10secretor.secreteOutsideCellAtBoundary(cell, 0)
                                # chemotax and uptake MCP, VEGF, and TGF once phagocytosis is done
                                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                                cdVEGF = self.chemotaxisPlugin.getChemotaxisData(cell, "VEGF")
                                cdTGF = self.chemotaxisPlugin.getChemotaxisData(cell, "TGF")
                                cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                                if cdMCP:
                                    cdMCP.setLambda(lmMono)
                                if cdVEGF:
                                    cdVEGF.setLambda(lmMono)
                                if cdTGF:
                                    cdTGF.setLambda(lmMono)
                                if cdRep:
                                    cdRep.setLambda(lmRepel)
                                # Uptake of cytokines by monocytes           
                                TGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel) #rel is a number between 0-1
                                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                                VEGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                            else:
                                # secrete while phagocytosing 
                                HGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac) 
                                MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                # stop movement while phagocytosing 
                                TGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) #turn off uptake while phagocytosing 
                                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0)
                                VEGFsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) 
                                if cdMCP:
                                    cdMCP.setLambda(0)
                                if cdVEGF:
                                    cdVEGF.setLambda(0)
                                if cdTGF:
                                    cdTGF.setLambda(0)   
                                cell.dict['phagocytoseLagTime'] += 1
                
                # Transition to M1 from monocyte in 1-3 days using a gaussian distribution
                TNFvalue = TNFfield[cell.xCOM, cell.yCOM, cell.zCOM]
                if mcs > cell.dict['monoM1TransTime'] or TNFvalue > TNFconverter:
                    cell.dict['type'] = 1 # transition monocyte into M1
                    if TNFvalue > TNFconverter:
                        TNFfield[cell.xCOM, cell.yCOM, cell.zCOM] = TNFvalue - TNFconverter # remove TNF required for transition
                    
                
            if cell.dict['type'] == 1: # if M1 macrophage 
                # Phagocytose necrosis
                CorrespondingNecrotic = self.cell_field[cell.xCOM, cell.yCOM, 0]
                if CorrespondingNecrotic and CorrespondingNecrotic.type == self.NECROTIC and CorrespondingNecrotic.dict['capillary'] == 0:
                    if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                        CorrespondingNecrotic.type = self.ECM
                        CorrespondingNecrotic.dict['repair'] = 0
                        CorrespondingNecrotic.dict['collagen'] = 0.1
                        lowColIdx.insert(CorrespondingNecrotic.id, (CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM, CorrespondingNecrotic.xCOM, CorrespondingNecrotic.yCOM)) # adding low collagen to the index
                        cell.dict['NumPhagocytosed'] += 1
                        cell.dict['phagocytoseLagTime'] = 0 # reset phagocytosis lag counter
                        cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                        cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                        if cdMCP:
                            cdMCP.setLambda(lmMac)
                        if cdRep:
                            cdRep.setLambda(lmRepel)   
                        # stop phagocytosis specific secretion once phagocytosis is done
                        MMPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                        HGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                        IL10secretor.secreteOutsideCellAtBoundary(cell, 0)      
                        # Uptake of cytokines by M1 once phagocytosis complete          
                        MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel) #rel is a number between 0-1
               
                    else:
                        # stop movement and uptake while phagocytosing and secrete HGF, MMP, TGF, and IL-10 as phagocytosing 
                        cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                        MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        HGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                        IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)  
                        if cdMCP:
                            cdMCP.setLambda(0)   
                        MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) #turn off uptake while phagocytosing  
                        cell.dict['phagocytoseLagTime'] += 1
                 
                else: 
                    # Chemotax and uptake MCP if there is no necrosis nearby
                    cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                    cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                    if cdMCP:
                        cdMCP.setLambda(lmMac)
                    if cdRep:
                        cdRep.setLambda(lmRepel)
                    MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                    # M1 macrophages secrete TNF, MMP, and VEGF
                    TNFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                    VEGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                    #IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)

                # Eat apoptotic neutrophils
                for neighbor in self.get_cell_neighbor_data_list(cell):
                    if neighbor[0]:
                        if neighbor[0].type == self.NEUTROPHIL and neighbor[0].dict['Apoptotic'] == 1:
                            if cell.dict['phagocytoseLagTime'] >= phagocytoseLagTime:
                                neighbor[0].targetVolume=0
                                neighbor[0].lambdaVolume=1000000
                                cell.dict['NumPhagocytosed'] += 1  
                                # stop phagocytosis specific secretion once phagocytosis is done
                                MMPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                                HGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                                TGFsecretor.secreteOutsideCellAtBoundary(cell, 0)
                                IL10secretor.secreteOutsideCellAtBoundary(cell, 0)  
                                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                                cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                                if cdMCP:
                                    cdMMP.setLambda(lmMac)
                                if cdRep:
                                    cdRep.setLambda(lmRepel)
                                # Uptake of cytokines by M1 
                                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                            else:
                                # secrete while phagocytosing and stop movement and uptake 
                                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                                MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                HGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                                IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)  
                                if cdMCP:
                                    cdMCP.setLambda(0)  
                                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, 0, 0) #turn off uptake while phagocytosing
                
                global fiberNecrosis
                IL10val = IL10field[cell.xCOM, cell.yCOM, cell.zCOM]
                
                # If macrophage has lived its full lifespan then die 
                if np.random.uniform() < (1-np.exp(-MacrophageDeathPoissonM1)):
                    #print('M1 death')
                    cell.targetVolume=0
                    cell.lambdaVolume=1000000

                else: 
                    # Transition from M1 to M2 
                    if cell.dict['NumPhagocytosed'] >= phagoThresholdforNeuM2 or IL10val > IL10TransThreshold:
                    # if mcs > cell.dict['M1M2TransTime']: #time required to go from M1 to M2
                      cell.dict['type'] = 2 # transition M1 into M2
                      if IL10val > IL10TransThreshold:
                        IL10field[cell.xCOM, cell.yCOM, cell.zCOM] = IL10val - IL10TransThreshold # remove Il-10 required for transition 
            
            if cell.dict['type'] == 2: # if M2 macrophage 
                # Chemotax and uptake MCP
                cdMCP = self.chemotaxisPlugin.getChemotaxisData(cell, "MCP")
                cdRep = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL")
                if cdMCP:
                    cdMCP.setLambda(lmMac)
                if cdRep:
                    cdRep.setLambda(lmRepel)
                MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(cell, secretionUptakeValMac, uptakeRel)
                # M2 macrophages secrete TFG-b and IL-10
                TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac) # anti-inflammatory effect of M2 is to secrete TGF-b and IL-10
                IL10secretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)

                # If macrophage has lived its full lifespan then die 
                if np.random.uniform() < (1-np.exp(-MacrophageDeathPoissonM2)):
                        cell.targetVolume=0
                        cell.lambdaVolume=1000000
                if cell.dict['secretionCountdown'] >= 0: # continue secreting
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValMac)
                else: # stop secretion
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, 0)
                cell.dict['secretionCountdown'] -= 1 # increment countdown      
                                
             
           
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return       

class PlotSteppable(SteppableBasePy): # Create and add data to plots 
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        if ifPlot == 1:
            # initialize graphs for CSA, #SSC, #Neutrophils, #Macro, and E2
            self.plot_win_csa = self.add_new_plot_window(title='Cross Sectional Area per step',
                                                     x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Area', x_scale_type='linear', y_scale_type='linear',
                                                     grid=True)
            self.plot_win_csa.add_plot("FibArea", style='Lines', color='red', size=5)

            self.plot_win_ssc = self.add_new_plot_window(title='Stem Cell Count',
                                                     x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Cell Count', x_scale_type='linear', y_scale_type='linear',
                                                     grid=True)
            self.plot_win_ssc.add_plot("TotalSSC", style='Lines', color='red', size=5)
            self.plot_win_ssc.add_plot("ActiveSSC", style='Lines', color='orange', size=5)
            self.plot_win_ssc.add_plot("SSCNum", style='Lines', color='blue', size=5)
            self.plot_win_ssc.add_plot("MyoblastNum", style='Lines', color='purple', size=5)
            self.plot_win_ssc.add_plot("MyocyteNum", style='Lines', color='magenta', size=5)
            self.plot_win_ssc.add_plot("InactiveNum", style='Lines', color='yellow', size=5)
            
            self.plot_win_neu = self.add_new_plot_window(title='Neutrophil Count',
                                                     x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Cell Count', x_scale_type='linear', y_scale_type='linear',
                                                     grid=True)
            self.plot_win_neu.add_plot("NeuNum", style='Lines', color='red', size=5)
            
            self.plot_win_macro = self.add_new_plot_window(title='Macrophage Count',
                                                     x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Cell Count', x_scale_type='linear', y_scale_type='linear',
                                                     grid=True)
            self.plot_win_macro.add_plot("MacroNum", style='Lines', color='green', size=5)
            self.plot_win_macro.add_plot("MonoNum", style='Lines', color='purple', size=5)
            self.plot_win_macro.add_plot("M1Num", style='Lines', color='red', size=5)
            self.plot_win_macro.add_plot("M2Num", style='Lines', color='blue', size=5)
            
            self.plot_win_fibro = self.add_new_plot_window(title='Fibroblast Count',
                                                     x_axis_title='MonteCarlo Step (MCS)',
                                                     y_axis_title='Cell Count', x_scale_type='linear', y_scale_type='linear',
                                                     grid=True)
            self.plot_win_fibro.add_plot("FibroNum", style='Lines', color='red', size=5)
            
        return
        
    def step(self, mcs):
        if ifPlot == 1:
            # add data to graphs for CSA, #SSC, #Neutrophils, #Macro, and E2
            # arguments are (name of the data series, x, y)
            CellVolume = 0
            for cell in self.cell_list_by_type(self.FIBER):
                CellVolume += cell.volume
            self.plot_win_csa.add_data_point("FibArea", mcs, CellVolume)
                  
            
            active = 0
            ssc = 0
            myoblast = 0
            myocyte = 0
            inactive = 0
            ssc_cells = list(self.cell_list_by_type(self.SSC))
            for cell in ssc_cells:
                if cell.dict['activationState'] == 1:
                    active +=1
                if cell.dict['cellType'] == 0 and cell.dict['activationState'] == 1:
                    ssc +=1 # ssc that is pre differntiated
                if cell.dict['cellType'] == 1:
                    myoblast +=1
                if cell.dict['cellType'] == 2:
                    myocyte +=1
                if cell.dict['activationState'] == 0:
                    inactive += 1

            self.plot_win_ssc.add_data_point("TotalSSC", mcs, len(ssc_cells))
            self.plot_win_ssc.add_data_point("ActiveSSC", mcs, active)
            self.plot_win_ssc.add_data_point("SSCNum", mcs, ssc)
            self.plot_win_ssc.add_data_point("MyoblastNum", mcs, myoblast)
            self.plot_win_ssc.add_data_point("MyocyteNum", mcs, myocyte)
            self.plot_win_ssc.add_data_point("InactiveNum", mcs, inactive)
         
            neuCount = 0
            for cell in self.cell_list_by_type(self.NEUTROPHIL):
                if cell.dict['Apoptotic'] == 0:
                    neuCount += 1

            self.plot_win_neu.add_data_point("NeuNum", mcs, neuCount)

            mono = 0
            m1 = 0
            m2 = 0
            for cell in self.cell_list_by_type(self.MACROPHAGE):
                if cell.dict['type'] == 0:
                    mono +=1
                if cell.dict['type'] == 1:
                    m1 +=1 
                if cell.dict['type'] == 2:
                    m2 +=1
               
            self.plot_win_macro.add_data_point("MacroNum", mcs, len(self.cell_list_by_type(self.MACROPHAGE)))
            self.plot_win_macro.add_data_point("MonoNum", mcs, mono)
            self.plot_win_macro.add_data_point("M1Num", mcs, m1)
            self.plot_win_macro.add_data_point("M2Num", mcs, m2)
            
            self.plot_win_fibro.add_data_point("FibroNum", mcs, len(self.cell_list_by_type(self.FIBROBLAST)))
            
           
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return

class FileSteppable(SteppableBasePy): # Write output to files
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        self._last_time = None
    def start(self):
        self._last_time = time.time()
        if ifDataSave == 1:
            # Define path to store data and open files with headers
            if ifPythonCall:
                fileDirUp = outputPath+f'/Sample_{SampleNumber}'
            else:
                fileDir = os.path.dirname(os.path.abspath(__file__))
                fileDirUp = os.path.dirname(fileDir)
            # fileName2 = fileDirUp+"\\logfile2MR.txt"
            fileName2 = fileDirUp+"/logfile2MR.txt"
            self.file2 = open(fileName2,"w", buffering=1)
            outputText = "mcs, fiberVolume, sscCount, macroCount, neutroCount\n"
            self.file2.write(outputText)
            # # SSCfile = fileDirUp+"\\SSCdata.txt"
            SSCfile = fileDirUp+"/SSCdata.txt"
            self.file3 = open(SSCfile, "w", buffering=1)
            outputText = "mcs, active, inactive, preDiffSSC, myoblast, myocyte\n"
            self.file3.write(outputText)
            # # neuFile = fileDirUp+"\\neuData.txt"
            # neuFile = fileDirUp+"/neuData.txt"
            # self.file4 = open(neuFile, "w")
            # outputText = "mcs, cellID, cellType, xCOM, yCOM, numPhagocytosed, Apoptotic, HGF, MMP, MCP\n"
            # self.file4.write(outputText)
            # # macFile = fileDirUp+"\\macData.txt"
            # macFile = fileDirUp+"/macData.txt"
            # self.file5 = open(macFile, "w")
            # outputText = "mcs, cellID, cellType, xCOM, yCOM, numPhagocytosed, HGF, MMP, MCP\n"
            # self.file5.write(outputText)
            # CSAfile = fileDirUp+"\\CSAdata.txt"
            CSAfile = fileDirUp+"/CSAdata.txt"
            self.file6 = open(CSAfile, "w", buffering=1)
            outputText = "mcs, numHealthyFiber, numDamagedNecrotic, numRecoveringECM\n"
            self.file6.write(outputText)
            # fieldFile = fileDirUp+"\\fieldData.txt"
            fieldFile = fileDirUp+"/fieldData.txt"
            self.file7 = open(fieldFile, "w", buffering=1)
            outputText = "mcs, MCPMean, HGFMean, MMPMean, TGFMean, TNFMean, IL10Mean, VEGFMean\n"
            self.file7.write(outputText)
            CapillaryData = fileDirUp+"/CapillaryData.txt"
            self.file8 = open(CapillaryData, "w", buffering=1)
            outputText = "mcs, CapillaryCount, CapillaryMyofiberRatio, fragmented, sprouting, healthy\n"
            self.file8.write(outputText)
            Fibroblast = fileDirUp+"/FibroblastData.txt"
            self.file9 = open(Fibroblast, "w", buffering=1)
            outputText = "mcs, TotalFibroblastCount, ActiveFibroblastCount, MyoFibroblastCount\n"
            self.file9.write(outputText)
            MacDynamics = fileDirUp+"/MacDynamics.txt"
            self.file10 = open(MacDynamics, "w", buffering=1)
            outputText = "mcs, M1Count, M2Count, MonocyteCount, ResidentCount\n"
            self.file10.write(outputText)
            ecmDynamics =fileDirUp+"/ecmDynamics.txt"
            self.file11 = open(ecmDynamics, "w", buffering=1)
            outputText = "mcs, LowCollagenECM, HighCollagenECM, NormalCollagenECM\n"
            self.file11.write(outputText)
            # if ifPythonCall:
            #     ParamFile = fileDirUp+"\\Param.txt"
            #     # ParamFile = fileDirUp+"/Param.txt"
            #     self.file8 = open(ParamFile, "a")
            #     outputText = str(values2pass)+"\n"
            #     self.file8.write(outputText)
    def step(self, mcs):
        total_steps = self.simulator.getNumSteps()
        if mcs % 10 == 0:
            now = time.time()
            elapsed = now - self._last_time if self._last_time else 0
            mcs_gap = mcs - getattr(self, '_last_mcs', 0) if mcs > 0 else 1
            self._last_time = now
            self._last_mcs = mcs
            print(f"Progress: {mcs}/{total_steps} ({100*mcs/total_steps:.1f}%) — {elapsed/mcs_gap:.2f}s/step", flush=True)
        if ifDataSave == 1:
            # Every hour (1 mcs = 15 mins -> 4 mcs) record outputs
            if mcs %4 == 0:
                # Record fiber cell volume and agent counts
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                _ssc_cells    = list(self.cell_list_by_type(self.SSC))
                _macro_cells  = list(self.cell_list_by_type(self.MACROPHAGE))
                _fibro_cells  = list(self.cell_list_by_type(self.FIBROBLAST))
                _cap_cells    = list(self.cell_list_by_type(self.CAPILLARY))
                macroCount = len(_macro_cells)
                sscCount = len(_ssc_cells)
                TotalFibroblastCount = len(_fibro_cells)
                _stop_sim = sscCount > 12000 or TotalFibroblastCount > 12000
                neutroCount = len(self.cell_list_by_type(self.NEUTROPHIL))

                outputText = str(mcs)+", "+str(CellVolume)+", "+str(sscCount)+", "+str(macroCount)+", "+str(neutroCount)+ "\n"
                # outputText = str(mcs)+", "+str(CellVolume)+", "+str(sscCount)+", "+str(macroCount)+", "+str(neutroCount)+ "," + str(TGFMean)+"\n"
                self.file2.write(outputText)
                # Record fiber, ecm, necrosis counters
                numHealthy = len(self.cell_list_by_type(self.FIBER))
                numDamaged = len(self.cell_list_by_type(self.NECROTIC))
                global startingECM
                _ecm_len = len(self.cell_list_by_type(self.ECM))
                if mcs < 2:
                    startingECM = _ecm_len
                numRecovering = _ecm_len - startingECM
                outputTextCSA = str(mcs)+", "+str(numHealthy)+", "+str(numDamaged)+", "+str(numRecovering)+"\n"
                self.file6.write(outputTextCSA)
            
#UNCOMMENT FOR FULL DATA, COMMENTED TO SAVE TIME ON LHS SWEEPS FOR CALIBRATION
                active = 0
                ssc = 0
                myoblast = 0
                myocyte = 0
                inactive = 0
                for cell in _ssc_cells:
                    if cell.dict['activationState'] == 1:
                        active +=1
                    if cell.dict['cellType'] == 0 and cell.dict['activationState'] == 1:
                        ssc +=1 # ssc that is pre differntiated
                    if cell.dict['cellType'] == 1:
                        myoblast +=1
                    if cell.dict['cellType'] == 2:
                        myocyte +=1
                    if cell.dict['activationState'] == 0:
                        inactive += 1
                outputTextSSC = str(mcs)+", "+str(active)+", "+ str(inactive)+", "+str(ssc)+", "+str(myoblast)+", "+str(myocyte)+"\n"
                self.file3.write(outputTextSSC)
            #     for cell in self.cell_list_by_type(self.NEUTROPHIL):
            #         outputTextNeu = str(mcs)+", "+str(cell.id)+", "+str(cell.type)+", "+str(cell.xCOM)+", "+str(cell.yCOM)+ ", "+str(cell.dict['NumPhagocytosed'])+", " +str(cell.dict['Apoptotic']) +", " +str(self.field.HGF[cell.xCOM, cell.yCOM,0]) +", " +str(self.field.MMP[cell.xCOM, cell.yCOM,0]) +", " +str(self.field.MCP[cell.xCOM, cell.yCOM,0])+"\n"
            #         self.file4.write(outputTextNeu)
            #     for cell in self.cell_list_by_type(self.MACROPHAGE):
            #         outputTextMac = str(mcs)+", "+str(cell.id)+", "+str(cell.type)+", "+str(cell.xCOM)+", "+ str(cell.yCOM) +", "+str(cell.dict['NumPhagocytosed'])+", " +str(self.field.HGF[cell.xCOM, cell.yCOM,0]) +", " +str(self.field.MMP[cell.xCOM, cell.yCOM,0]) +", " +str(self.field.MCP[cell.xCOM, cell.yCOM,0])+ "\n"
            #         self.file5.write(outputTextMac)
                
            #UNCOMMENT ABOVE FOR FULL DATA
                CapNumber = len(_cap_cells)
                MyoFiberNumb = numHealthy
                CapMyoRatio = CapNumber/MyoFiberNumb if MyoFiberNumb > 0 else 0
                fragmented = 0
                sprouting = 0
                healthy = 0
                for cell in _cap_cells:
                    if cell.dict['fragmented'] == 0:
                        fragmented += 1
                    elif cell.dict['fragmented'] == 1:
                        sprouting += 1
                    elif cell.dict['fragmented'] == 2:
                        healthy += 1
                outputTextCapillary = str(mcs)+", "+str(CapNumber)+", "+str(CapMyoRatio)+", "+str(fragmented)+", "+str(sprouting)+", "+str(healthy)+"\n"
                self.file8.write(outputTextCapillary)
                ActiveFibroblastCount = 0
                MyoFibroblastCount = 0
                for fib in _fibro_cells:
                    if fib.dict['state'] == 1:
                        ActiveFibroblastCount += 1
                    elif fib.dict['state'] == 2:
                        MyoFibroblastCount += 1
                OutputTextFibro = str(mcs)+", "+str(TotalFibroblastCount)+", "+str(ActiveFibroblastCount)+", "+str(MyoFibroblastCount)+"\n"
                self.file9.write(OutputTextFibro)
                MonocyteCount = 0
                M1Count = 0
                M2Count = 0
                ResidentCount = 0
                for cell in _macro_cells:
                    if cell.dict['type'] == 0: # 0 for monocyte, 1 for M1, 2 for M2, 4 for resident
                        MonocyteCount += 1
                    elif cell.dict['type'] == 1:
                        M1Count += 1
                    elif cell.dict['type'] == 2:
                        M2Count += 1
                    elif cell.dict['type'] == 4:
                        ResidentCount += 1
                OutputTextMacDynamics = str(mcs)+", "+str(M1Count)+", "+str(M2Count)+", "+str(MonocyteCount)+", "+str(ResidentCount)+"\n"
                self.file10.write(OutputTextMacDynamics)
                if _stop_sim:
                    self.stop_simulation()

            cytokineCheck = [48, 96, 288, 480, 672, 1344, 2016, 2684] # check cytokine levels and ecm dynamics at 12 hrs, 1 d, 3 d, 5 d, 7 d, 14 d, 21 d, 28 d
            if mcs in cytokineCheck:
                _sx = np.random.randint(0, self.dim.x-1, 500)
                _sy = np.random.randint(0, self.dim.y-1, 500)
                _pts = list(zip(_sx.tolist(), _sy.tolist()))
                MCPMean  = float(np.mean([self.field.MCP [x, y, 1] for x, y in _pts]))
                HGFMean  = float(np.mean([self.field.HGF [x, y, 1] for x, y in _pts]))
                MMPMean  = float(np.mean([self.field.MMP [x, y, 1] for x, y in _pts]))
                TGFMean  = float(np.mean([self.field.TGF [x, y, 1] for x, y in _pts]))
                TNFMean  = float(np.mean([self.field.TNF [x, y, 1] for x, y in _pts]))
                IL10Mean = float(np.mean([self.field.IL10[x, y, 1] for x, y in _pts]))
                VEGFMean = float(np.mean([self.field.VEGF[x, y, 1] for x, y in _pts]))
                outputTextField = str(mcs)+", "+str(MCPMean)+", "+str(HGFMean)+", "+str(MMPMean)+", "+str(TGFMean)+", "+str(TNFMean)+", "+str(IL10Mean)+", "+str(VEGFMean)+"\n"
                self.file7.write(outputTextField)
                LowCollagenCounter = 0
                HighCollagenCounter = 0
                NormalCollagenCounter = 0
                for cell in self.cell_list_by_type(self.ECM):
                    if cell.dict['collagen'] < 0.5:
                        LowCollagenCounter +=1
                    elif cell.dict['collagen'] >= 0.5 and cell.dict['collagen'] <= 10:
                        NormalCollagenCounter +=1
                    elif cell.dict['collagen'] > 10:
                        HighCollagenCounter += 1
                OutputTextCollagen = str(mcs)+", "+str(LowCollagenCounter)+", "+str(HighCollagenCounter)+", "+str(NormalCollagenCounter)+"\n"
                self.file11.write(OutputTextCollagen)



    def finish(self):
        if ifDataSave == 1:
            # Display closing of files and close files
            self.file2.write("Closing file\n\n")
            self.file2.close()
            self.file3.close()
            # self.file4.write("Closing file\n\n")
            # self.file4.close()
            # self.file5.write("Closing file\n\n")
            # self.file5.close()
            self.file6.write("Closing file\n\n")
            self.file6.close()
            self.file7.write("Closing file\n\n")
            self.file7.close()
            self.file8.write("Closing file\n\n")
            self.file8.close()
            self.file9.write("Closing file\n\n")
            self.file9.close()
            self.file10.write("Closing file\n\n")
            self.file10.close()
            self.file11.write("Closing file\n\n")
            self.file11.close()
        return
    def on_stop(self):
        if ifDataSave == 1:
            # Display closing of files and close files
            self.file2.write("Closing file\n\n")
            self.file2.close()
            self.file3.close()
            # self.file4.write("Closing file\n\n")
            # self.file4.close()
            # self.file5.write("Closing file\n\n")
            # self.file5.close()
            self.file6.write("Closing file\n\n")
            self.file6.close()
            self.file7.write("Closing file\n\n")
            self.file7.close()
            self.file8.write("Closing file\n\n")
            self.file8.close()
            self.file9.write("Closing file\n\n")
            self.file9.close()
            self.file10.write("Closing file\n\n")
            self.file10.close()
            self.file11.write("Closing file\n\n")
            self.file11.close()
        return

class FiberSteppable(SteppableBasePy): # make fiber clusters and adjust repeal secretion
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        # Used to visualize clusters of fibers
        self.track_cell_level_scalar_attribute(field_name='clusterNum', attribute_name='clusterNum')

    def start(self):
        print("Initializing fiber clusters...", flush=True)
        # join all connected fibers into one cluster
        global fiberGroups
        fiberGroups = {}
        for cell in self.cell_list_by_type(self.FIBER): # Check all fibers
            cluster = None
            for neighbor, _ in self.get_cell_neighbor_data_list(cell): # Check current fiber neighbors
                if neighbor and neighbor.type == self.FIBER: # Make sure neighbor is a fiber
                    for group in fiberGroups: # For each cluster
                        if group != cluster:
                            if any(neighbor.id == item.id for item in fiberGroups[group]):
                            # if neighbor in fiberGroups[group]: Go through existing clusters to see if neighbor is in cluster
                                if cluster != None: # already assigned to a cluster 
                                    fiberGroups[cluster].extend(fiberGroups[group]) # Adding connecting groups
                                    fiberGroups.pop(group) # Gets rid of the group that we added so not duplicated
                                else: 
                                   fiberGroups[group].append(cell) # Not in a cluster so adding to nearby existing cluster
                                   cluster = group
                                break
            if cluster == None:
                fiberGroups[cell.clusterId] = [cell] # Starting a new cluster if no neighbors are a cluster
        
        compartmentNum = 1        
        for group in fiberGroups:
            for cell in fiberGroups[group]:
                self.reassign_cluster_id(cell, group)
                cell.dict['clusterNum'] = compartmentNum
                compartmentNum += 1
        
        print(f"Fiber clusters done. Starting simulation...", flush=True)
        global numFiberCells
        numFiberCells = len(fiberGroups)
        
        # Repel cells from wall/keep on the muscle cross section
        secretor = self.get_field_secretor("REPEL")
        for cell in self.cell_list_by_type(self.WALL):
             secretor.secreteOutsideCellAtBoundary(cell, 100)
        
        return

    def step(self, mcs):
        #Make any fiber with one necrotic element permiable to cells
        secretor = self.get_field_secretor("REPEL")
        lowColCount = 0
        if mcs == 5: #or mcs % 1300 == 0: # Only checking every 5 mcs to reduce comp cost
            # Loop through clusters
            for compartments in self.clusters:
                ifFiber = False
                ifNecrotic = False
                ifECM = False
                #ifCap = False
                for cell in compartments:
                    if cell.type == self.FIBER:
                        ifFiber = True
                    if cell.type == self.NECROTIC:
                        ifNecrotic = True
                    if cell.type == self.ECM:
                        ifECM = True  
            
                # Make 'solid'
                if ifFiber and not ifNecrotic and not ifECM:
                    for cell in compartments:
                        secretor.secreteOutsideCellAtBoundary(cell, 100)
                        VEGFsecretor = self.get_field_secretor("VEGF")
                        VEGFsecretor.secreteOutsideCellAtBoundary(cell, VEGFsecretionValFib) # VEGF secreted by fibers
                        if ifNecrotic:
                            secretor.secreteOutsideCellAtBoundary(cell, 0)
                
        # Calc avg collagen density and adjust diffusion based on ECM conditions 
        if mcs > 2 and mcs % 50 == 0: # Only checking every 50 mcs to reduce comp cost
            totalCollagen = 0;
            numECMele = len(self.cell_list_by_type(self.ECM))
            for cell in self.cell_list_by_type(self.ECM):
                totalCollagen = totalCollagen + cell.dict['collagen'] # add collagen amount in each ECM element
            avgCollagen = totalCollagen/numECMele
            #print(avgCollagen)
            # Calc change in diffusivity for each cytokine based on the average collagen (see evernote for reference equations)
            updatedCollagenFraction = 0.14*avgCollagen # 0.14 coming from Gabhann et al. 2006
            #print(updatedCollagenFraction)
            # VEGF calc
            VEGFdiffusivity = (133*math.exp(-updatedCollagenFraction**(1/2)*(2.46/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(2.46/0.55)))
            # get the xml element and place new value
            VEGFdiffusion = self.get_xml_element('VEGFdiffusion')
            #print('VEGF', VEGFdiffusion.cdata)
            VEGFdiffusion.cdata = float(VEGFdiffusivity*10**-3)  
            # HGF calc
            HGFdiffusivity = (86.44*math.exp(-updatedCollagenFraction**(1/2)*(3.80/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(3.80/0.55)))
            # get the xml element and place new value
            HGFdiffusion = self.get_xml_element('HGFdiffusion')
            HGFdiffusion.cdata = float(HGFdiffusivity*10**-3)  
            # MMP calc
            MMPdiffusivity = (83.36*math.exp(-updatedCollagenFraction**(1/2)*(3.94/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(3.94/0.55)))
            # get the xml element and place new value
            MMPdiffusion = self.get_xml_element('MMPdiffusion')
            MMPdiffusion.cdata = float(MMPdiffusivity*10**-3)  
            # MCP calc
            MCPdiffusivity = (207.88*math.exp(-updatedCollagenFraction**(1/2)*(1.58/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(1.58/0.55)))
            # get the xml element and place new value
            MCPdiffusion = self.get_xml_element('MCPdiffusion')
            MCPdiffusion.cdata = float(MCPdiffusivity*10**-3)  
            # TGF calc
            TGFdiffusivity = (110.96*math.exp(-updatedCollagenFraction**(1/2)*(2.96/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(2.96/0.55)))
            # get the xml element and place new value
            TGFdiffusion = self.get_xml_element('TGFdiffusion')
            TGFdiffusion.cdata = float(TGFdiffusivity*10**-3)  
            # TNF calc  
            TNFdiffusivity = (160.22*math.exp(-updatedCollagenFraction**(1/2)*(2.05/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(2.05/0.55)))
            # get the xml element and place new value
            TNFdiffusion = self.get_xml_element('TNFdiffusion')
            TNFdiffusion.cdata = float(TNFdiffusivity*10**-3) 
            # IL-10 calc  
            IL10diffusivity = (156.41*math.exp(-updatedCollagenFraction**(1/2)*(2.10/20))) * (math.exp(-(7.8*10**-4)**(1/2)*(2.10/0.55)))
            # get the xml element and place new value
            IL10diffusion = self.get_xml_element('IL10diffusion')
            IL10diffusion.cdata = float(IL10diffusivity*10**-3)  
                     
class LymphaticSteppable(SteppableBasePy): # pulls cells toward lymph, removes chemokines, removes ssc, neutrophils, and macrophages
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        return

    def step(self, mcs):
        # Lymph uptakes hgf, mcp, mmp, e2, ssc, neutrophils, and macrophages
        HGFsecretor = self.get_field_secretor("HGF") #secretors for uptake
        MCPsecretor = self.get_field_secretor("MCP")
        MMPsecretor = self.get_field_secretor("MMP")
        VEGFsecretor = self.get_field_secretor("VEGF")
        TNFsecretor = self.get_field_secretor("TGF")
        TGFsecretor = self.get_field_secretor("TNF")
        E2secretor = self.get_field_secretor("E2")
        for lymp in self.cell_list_by_type(self.LYMPHATIC):
            # lymp uptakes fields
            HGFsecretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20, .8) #rel is a number between 0-1
            MCPsecretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20, .8)
            MMPsecretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20, .8)
            TGFsecretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20, .8)
            TNFsecretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20, .8)
            VEGFsecretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20, .8)
            E2secretor.uptakeInsideCellAtBoundaryTotalCount(lymp, 20/100.0, .8/100.0)

            for neighbor, _ in self.get_cell_neighbor_data_list(lymp): # get all cells connected to lymphatic cell
              if neighbor and (neighbor.type == self.SSC or neighbor.type == self.NEUTROPHIL or neighbor.type == self.MACROPHAGE or neighbor.type == self.FIBROBLAST):
                self.delete_cell(neighbor) #remove connected neighbors

            # force towards lymp — lambdaVec is persistent so only update every 10 steps
            if mcs % 10 == 0:
                for cell in self.cell_list_by_type(self.SSC, self.NEUTROPHIL, self.MACROPHAGE, self.FIBROBLAST):
                    if cell.xCOM - lymp.xCOM != 0:
                        cell.lambdaVecX = lympForce/(cell.xCOM - lymp.xCOM)
                    if cell.yCOM - lymp.yCOM != 0:
                        cell.lambdaVecY = lympForce/(cell.yCOM - lymp.yCOM)
        return

    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return
   
class PassValuesSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        
    def start(self):
        
        if ifPythonCall:
            values2pass = pg.input_object
            global neutroRecruitmentProportion
            global secretionUptakeValNeu
            global necrosisThreshold
            global macroRecruitmentProportion
            global secretionUptakeValMac
            global TNFconverter
            global macRecruitThres
            global macRecruitStop
            #global M2TransitionProb
            global IL10TransThreshold
            global phagoThresholdforNeuM2
            global sscActivationThreshold
            global recruitmentProportionSSC
            global quiescentProb
            global secretionUptakeValSSC
            global sscTGFApoptosisThreshold
            global sscApoptosisProb
            global VEGFblockApop
            global SSCdivisionThreshold
            global SSCdiffThreshold
            global sscDivideProb
            global sscDiffProb
            global fuseProb
            global quiescentThreshold
            global necrosisSecretionVal
            global VEGFsecretionValFib
            global VEGF_MMPthreshold
            global capVEGFUptakeMax
            global MMPsecretionValCap
            global NumFibers
            global sproutProb
            global secretionUptakeValFibro
            global collagenSecretionValFibro
            global distNearSSCdiv
            #global recruitmentProportionFibro
            global TGFActivationThreshold
            global TGFMyoThreshold
            #global TGFProliferationThreshold
            global fibroblastProliferationProb
            global fibroblastTNFApoptosisThreshold
            global TGFBlockApoptosisThreshold
            global fibroblastApoptosisProb
            global IL10decay
            global HGFdecay
            global TGFdecay
            global TNFdecay
            global MCPdecay
            global MMPdecay
            global VEGFdecay
            global MinCapDist
            global capPlacementThreshold
            global M2prolifProb
            global M1HalfLife
            global M2HalfLife
            global reperfusionLag

            global SampleNumber
            global outputPath



            neutroRecruitmentProportion = float(values2pass[0])
            secretionUptakeValNeu = float(values2pass[1])
            necrosisThreshold = float(values2pass[2])
            macroRecruitmentProportion = float(values2pass[3])
            secretionUptakeValMac = float(values2pass[4])
            TNFconverter = float(values2pass[5])
            macRecruitThres = float(values2pass[6])
            macRecruitStop = float(values2pass[7])
            #M2TransitionProb = float(values2pass[8])
            IL10TransThreshold = float(values2pass[8])
            phagoThresholdforNeuM2 = float(values2pass[9])
            sscActivationThreshold = float(values2pass[10])
            recruitmentProportionSSC = float(values2pass[11])
            quiescentProb = float(values2pass[12])
            secretionUptakeValSSC = float(values2pass[13])
            sscTGFApoptosisThreshold = float(values2pass[14])
            sscApoptosisProb = float(values2pass[15])
            VEGFblockApop = float(values2pass[16])
            SSCdivisionThreshold = float(values2pass[17])
            SSCdiffThreshold = float(values2pass[18])
            sscDivideProb = float(values2pass[19])
            sscDiffProb = float(values2pass[20])
            fuseProb = float(values2pass[21])
            quiescentThreshold = float(values2pass[22])
            necrosisSecretionVal = float(values2pass[23])
            VEGFsecretionValFib = float(values2pass[24])
            VEGF_MMPthreshold = float(values2pass[25])
            capVEGFUptakeMax = float(values2pass[26])
            MMPsecretionValCap = float(values2pass[27])
            NumFibers = float(values2pass[28])
            sproutProb = float(values2pass[29])
            secretionUptakeValFibro = float(values2pass[30])
            collagenSecretionValFibro = float(values2pass[31])
            distNearSSCdiv = float(values2pass[32])
            #recruitmentProportionFibro = float(values2pass[33])
            TGFActivationThreshold = float(values2pass[33])
            TGFMyoThreshold = float(values2pass[34])
            #TGFProliferationThreshold = float(values2pass[35])
            fibroblastProliferationProb = float(values2pass[35])
            fibroblastTNFApoptosisThreshold = float(values2pass[36])
            TGFBlockApoptosisThreshold = float(values2pass[37])
            fibroblastApoptosisProb = float(values2pass[38])
            
            IL10decay = self.get_xml_element('IL10decay')
            IL10decay.cdata = float(values2pass[39])
            HGFdecay = self.get_xml_element('HGFdecay')
            HGFdecay.cdata = float(values2pass[40])
            TGFdecay = self.get_xml_element('TGFdecay')
            TGFdecay.cdata = float(values2pass[41])
            TNFdecay = self.get_xml_element('TNFdecay')
            TNFdecay.cdata = float(values2pass[42])
            MCPdecay = self.get_xml_element('MCPdecay')
            MCPdecay.cdata = float(values2pass[43])
            MMPdecay = self.get_xml_element('MMPdecay')
            MMPdecay.cdata = float(values2pass[44])
            VEGFdecay = self.get_xml_element('VEGFdecay')
            VEGFdecay.cdata = float(values2pass[45])

            MinCapDist = float(values2pass[46])
            capPlacementThreshold = float(values2pass[47])
            M2prolifProb = float(values2pass[48])

            M1HalfLife = float(values2pass[49])
            M2HalfLife = float(values2pass[50])
            reperfusionLag = float(values2pass[51])


            SampleNumber = int(float(values2pass[52]))
            outputPath = values2pass[53]

            self.sscOutput = []
            self.csaOutput = []
            self.macroOutput = []
            self.neuOutput = []

            # Perturbation variables 
            if neuDeplete == 1:
                neutroRecruitmentProportion = 0
            if macDeplete == 1:
                macroRecruitmentProportion = 0
            if M2PolDir == 1: 
                IL10TransThreshold = IL10TransThreshold/100            # Local IL-10 that can induce M1 to M2 transition 
            if defAngio == 1:
                VEGF_MMPthreshold = VEGF_MMPthreshold * 5
            if IL10_KO==1:
                IL10diffuse = self.get_xml_element('IL10diffusion')
                IL10diffuse.cdata = float(0)

                IL10decay = self.get_xml_element('IL10decay')
                IL10decay.cdata = float(10)
            if TNF_KO==1:
                TNFdiffuse = self.get_xml_element('TNFdiffusion')
                TNFdiffuse.cdata = float(0)

                TNFdecay = self.get_xml_element('TNFdecay')
                TNFdecay.cdata = float(10)
            if MCP_KO == 1:
                MCPdiffuse = self.get_xml_element('MCPdiffusion')
                MCPdiffuse.cdata = float(0)

                MCPdecay = self.get_xml_element('MCPdecay')
                MCPdecay.cdata = float(10)
            if VEGFhalfDiff == 1:
                VEGFdiffuse = self.get_xml_element('VEGFdiffusion')
                VEGFdiffuse.cdata = float(0.05605)
            if VEGFdoubleDiff == 1:
                VEGFdiffuse = self.get_xml_element('VEGFdiffusion')
                VEGFdiffuse.cdata = float(0.2242)

        return
        
    def step(self, mcs):
        if ifPythonCall:
            if mcs == 0:
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                self.startingVolume = CellVolume  #record starting volume to create CSA recovery percentage to compare with literature data
            
            #MCS in minutes
            #Neutrophils: 1, 3, 5 days
            #Macrophages: 3, 5, 10 days
            #SSC: 1, 3, 5, 7, 14, 21 days
            #CSA: 1, 3, 5, 14, 21 days
            
            if mcs == 24*timeconverter: # DAY 1
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                csaOutput = CellVolume/self.startingVolume
                sscCount = len(self.cell_list_by_type(self.SSC))
                neutroCount = len(self.cell_list_by_type(self.NEUTROPHIL))
                self.sscOutput.append(sscCount)
                self.csaOutput.append(csaOutput)
                self.neuOutput.append(neutroCount)
            
            if mcs == 72*timeconverter: # DAY 3
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                csaOutput = CellVolume/self.startingVolume
                macroCount = len(self.cell_list_by_type(self.MACROPHAGE))
                sscCount = len(self.cell_list_by_type(self.SSC))
                neutroCount = len(self.cell_list_by_type(self.NEUTROPHIL))
                self.sscOutput.append(sscCount)
                self.csaOutput.append(csaOutput)
                self.macroOutput.append(macroCount)
                self.neuOutput.append(neutroCount)
            
            if mcs == 120*timeconverter: # DAY 5
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                csaOutput = CellVolume/self.startingVolume
                macroCount = len(self.cell_list_by_type(self.MACROPHAGE))
                sscCount = len(self.cell_list_by_type(self.SSC))
                neutroCount = len(self.cell_list_by_type(self.NEUTROPHIL))
                self.sscOutput.append(sscCount)
                self.csaOutput.append(csaOutput)
                self.macroOutput.append(macroCount)
                self.neuOutput.append(neutroCount)
            
            if mcs == 168*timeconverter: # DAY 7
                sscCount = len(self.cell_list_by_type(self.SSC))
                self.sscOutput.append(sscCount)
            
            if mcs == 240*timeconverter: # DAY 10
                macroCount = len(self.cell_list_by_type(self.MACROPHAGE))
                self.macroOutput.append(macroCount)
            
            if mcs == 336*timeconverter: # DAY 14
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                csaOutput = CellVolume/self.startingVolume
                sscCount = len(self.cell_list_by_type(self.SSC))
                self.sscOutput.append(sscCount)
                self.csaOutput.append(csaOutput)
            
            if mcs == 504*timeconverter: # DAY 21
                CellVolume = 0
                for cell in self.cell_list_by_type(self.FIBER):
                    CellVolume += cell.volume
                csaOutput = CellVolume/self.startingVolume
                sscCount = len(self.cell_list_by_type(self.SSC))
                self.sscOutput.append(sscCount)
                self.csaOutput.append(csaOutput)
                
                toReturn = [self.sscOutput,self.csaOutput,self.macroOutput,self.neuOutput]
                pg.return_object= toReturn
                print("TO RETURN = ",toReturn)
        return


    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return
        
class MicrovesselSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)
        

    def start(self):
        for cell in self.cell_list_by_type(self.CAPILLARY):
            cell.dict['fragmented'] = 2 # healthy/perfusing
            cell.dict['nonperfusingTime'] = 0 # no time without perfusing
            cell.targetVolume = cell.volume     # Set target volume equal to volume defined by piff
            cell.lambdaVolume = 50 
        return


    def step(self, mcs):
        # Following inital damage, capillaries nearby will be fragmented  
        for cell in self.cell_list_by_type(self.CAPILLARY):
            if mcs == 2:
                for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                    if neighbor and neighbor.type == self.NECROTIC: 
                        # Damage capillary near necrosis 
                        cell.dict['fragmented'] = 0 # 0 for fragmented/non perfusing, 1 for sprouting/recovering, 2 for healthy/perfusing
            if cell.dict['fragmented'] == 0 or cell.dict['fragmented'] == 1:
                cell.dict['nonperfusingTime'] += 1 # counts the amount of time the capillary has been damaged or sprouting
                fieldVEGF = self.field.VEGF
                VEGFvalue = fieldVEGF[cell.xCOM, cell.yCOM, cell.zCOM]
                #if cell.dict['nonperfusingTime'] >= capRecovTime: 
                if VEGFvalue >= VEGF_MMPthreshold*np.random.uniform(0.5,10) and cell.dict['fragmented'] == 0 and np.random.uniform() < reperfusionLag: 
                    cell.dict['fragmented'] = 1 # capillary sprouting 
                    fieldVEGF[cell.xCOM, cell.yCOM, cell.zCOM] = VEGFvalue - VEGF_MMPthreshold # VEGF removal caused by binding required to sprout capillaries
                    MMPsecretor = self.get_field_secretor("MMP")
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, MMPsecretionValCap) # MMP elevated during capillary growth 
                if cell.dict['nonperfusingTime'] >= reperfusionTime*np.random.uniform(0.5,3) and cell.dict['fragmented'] == 1 and np.random.uniform() < reperfusionLag: 
                    cell.dict['fragmented'] = 2 # healthy/perfusing
                    MMPsecretor = self.get_field_secretor("MMP")
                    MMPsecretor.secreteOutsideCellAtBoundary(cell, 0) # turn off MMP secretion once done growing/regenerating                                    
                                    
        # Increasing capillariy to myofiber ratio via new endothelial sprouts
        fieldVEGF = self.field.VEGF
        fieldMMP = self.field.MMP
        if mcs % 10 == 0:
            # Pre-compute fiber and capillary positions once for the whole block
            fiber_list = list(self.cell_list_by_type(self.FIBER))
            cap_list   = list(self.cell_list_by_type(self.CAPILLARY))
            fiber_pos = np.array([[f.xCOM, f.yCOM] for f in fiber_list]) if fiber_list else np.empty((0,2))
            cap_pos   = np.array([[c.xCOM, c.yCOM] for c in cap_list])   if cap_list   else np.empty((0,2))

            for cell in self.cell_list_by_type(self.ECM):
                if cell.dict['collagen'] >= lowCollagenCutoff and cell.dict['repair'] == 1:
                    VEGFvalue = fieldVEGF[cell.xCOM, cell.yCOM, cell.zCOM]
                    MMPvalue  = fieldMMP[cell.xCOM, cell.yCOM, cell.zCOM]
                    if VEGFvalue >= VEGF_MMPthreshold and MMPvalue >= VEGF_MMPthreshold and np.random.uniform() < sproutProb:
                        cp = np.array([cell.xCOM, cell.yCOM])

                        # Vectorized fiber count within threshold
                        if fiber_pos.shape[0] > 0:
                            fiber_dists = np.sqrt(np.sum((fiber_pos - cp)**2, axis=1))
                            FibersNear = int(np.sum(fiber_dists < capPlacementThreshold))
                        else:
                            FibersNear = 0

                        NotNearby = True
                        if FibersNear > NumFibers:
                            # Vectorized cap proximity check
                            if cap_pos.shape[0] > 0:
                                cap_dists = np.sqrt(np.sum((cap_pos - cp)**2, axis=1))
                                NotNearby = not np.any(cap_dists < MinCapDist)

                            if NotNearby:
                                necroticNeighbor = False
                                for neighbor, _ in self.get_cell_neighbor_data_list(cell):
                                    if neighbor and neighbor.type == self.NECROTIC:
                                        necroticNeighbor = True
                                        break

                                if not necroticNeighbor:
                                    global lowColIdx
                                    if lowColIdx.count((cell.xCOM, cell.yCOM, cell.xCOM, cell.yCOM)) > 0:
                                        lowColIdx.delete(cell.id, (cell.xCOM, cell.yCOM, cell.xCOM, cell.yCOM))
                                    cell.type = self.CAPILLARY
                                    cell.dict['fragmented'] = 1
                                    cell.dict['nonperfusingTime'] = 0
                                    cell.targetVolume = cell.volume
                                    cell.lambdaVolume = 50
                                    fieldVEGF[cell.xCOM, cell.yCOM, 0] = VEGFvalue - VEGF_MMPthreshold
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return
     
class FibroblastSteppable(SteppableBasePy):
    def __init__(self, frequency=1):
        SteppableBasePy.__init__(self, frequency)

    def start(self):
        # Initialize Fibroblasts (1 fibroblast per 2 fiber)
        num_cells =  math.floor(numFiberCells/fibroblastToInitialFiber)
        timeChronicExposureTGF  = 0 # counter for the amount of time fibroblast has been exposed to chronic levels of TGF
        
        # Generate given number of cells where other cells don't already exist
        for i in range(num_cells):
            isPlaced = False
            
            while not isPlaced:
                xrand = np.random.randint(1,self.dim.x)
                yrand = np.random.randint(1, self.dim.y)
                
                cell = self.cellField[xrand,yrand,1]
                if not cell:
                    fib = self.new_cell(self.FIBROBLAST)
                    fib.targetVolume = 15
                    fib.lambdaVolume = 50
                    fib.dict['state'] = 0 # 0 = quiescent fibroblast, 1 = activated fibroblast 2 = myofibroblast
                    fib.dict['time2activate'] = -1 # counter for activation lag
                    fib.dict['time2divide'] = -1 # counter for division lag
                    fib.dict['numDiv'] = 0 # number of times fibroblast has divided 
                    fib.dict['timeChronExposure'] = 0 # counter for the amount of time fibroblast has been exposed to chronic levels of TGF
                    fib.dict['time2apop'] = 0 # counter for amount of time for apop signal before death

                    self.cell_field[xrand, yrand, 1] = fib
                    isPlaced = True
                                  
                    cd = self.chemotaxisPlugin.addChemotaxisData(fib, "REPEL") # keep fibroblasts from moving on healthy fibers
                    cd.setLambda(lmRepel)                    
                    

    def step(self, mcs):
        fieldTGF = self.field.TGF
        fieldTNF = self.field.TNF
        # fieldIL10 = self.field.IL10
        TNFsecretor = self.get_field_secretor("TNF")
        TGFsecretor = self.get_field_secretor("TGF")
        MMPsecretor = self.get_field_secretor("MMP")
        VEGFsecretor = self.get_field_secretor("VEGF")
        
        if mcs > 2:
            for cell in self.cell_list_by_type(self.FIBROBLAST):
                cellBelow = self.cell_field[cell.xCOM, cell.yCOM, 0]
                #check activation 
                if fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] > TGFActivationThreshold and cell.dict['state'] == 0: #and np.random.uniform() < fibroblastActivationProb:
                    cell.dict['time2activate'] = fibActivationTime
                    TGFvalue = fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM]
                    fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] = TGFvalue - TGFActivationThreshold # remove TGF from binding that induced activation
                cd = self.chemotaxisPlugin.getChemotaxisData(cell, "REPEL") # keep fibroblasts from moving on healthy fibers
                if cd:
                    cd.setLambda(lmRepel)

                if cell.dict['state'] == 1 or cell.dict['state'] == 2:  # must be activated fibroblast or myofibroblast to migrate or secrete  
                    # activated fibroblasts or myofibroblasts secrete collagen
                    if cellBelow and cellBelow is not None:
                        if cellBelow.type == self.ECM and cell.dict['state'] == 2: # more collagen secreted if activated/myofibroblast and does not require low density collagen
                            cellBelow.dict['collagen'] = cellBelow.dict['collagen'] + (collagenSecretionValFibro * 2)  # myofibroblasts secrete 2x collagen    
                            #print('myofibro added collagen',cellBelow.dict['collagen'])
                        if cell.dict['state'] == 1 and cellBelow.type == self.ECM and cellBelow.dict['collagen'] <= lowCollagenCutoff: # activated fibroblast should secrete collagen at low density ECM spots
                            cellBelow.dict['collagen'] = cellBelow.dict['collagen'] + collagenSecretionValFibro
                            #print('fibro added collagen',cellBelow.dict['collagen'])

                    idListNearLowCol = list(lowColIdx.nearest((cell.xCOM, cell.yCOM, cell.xCOM, cell.yCOM), 1)) # starting list for nearest low collagen elements
                    
                    if len(idListNearLowCol) > 0:                
                        # Fibroblast chemotaxes toward low collagen ECM 
                        idNearLowCol = idListNearLowCol[0] # gets the cell ID of the nearest low collagen ECM 
                        # print(idNearLowCol)
                        nearestLowCol = self.fetch_cell_by_id(idNearLowCol) # nearest low collagen cell
                                            
                        # if nearestLowCol is not None:
                        if len(self.get_fpp_links_by_cell(cell)) < 1 and len(self.get_fpp_links_by_cell(nearestLowCol)) < 1: # make sure that it doesn't form tons of links to fibroblast or a single low collagen ECM
                            link = self.new_fpp_link(cell, nearestLowCol, forceFibroToECM, 1, 1000) # Link never break, fibroblast need to get on top of low col ECM 
                         
                        for linked_cell in self.get_fpp_linked_cells(cell):
                            if linked_cell.dict['collagen'] >= lowCollagenCutoff:
                                self.remove_all_cell_fpp_links(cell) # Break link if no longer low collagen ECM so it will find another to move to once no longer low collagen
                    
                    if cellBelow and cellBelow is not None and cellBelow.type == self.ECM and cellBelow.dict['collagen'] >= lowCollagenCutoff and lowColIdx.count((cellBelow.xCOM, cellBelow.yCOM, cellBelow.xCOM, cellBelow.yCOM)) > 0: # check that it's above low collagen and is part of the low collagen list
                      lowColIdx.delete(cellBelow.id, (cellBelow.xCOM, cellBelow.yCOM, cellBelow.xCOM, cellBelow.yCOM)) 
                    
                    # If not activating or dividing
                    if  cell.dict['time2divide'] < 0 and cell.dict['time2activate'] < 0:
                        # Fibroblasts secrete TGF, TNF, VEGF, and MMP
                        TNFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValFibro) # reference Kyle's model sources
                        TGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValFibro)
                        MMPsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValFibro)
                        VEGFsecretor.secreteOutsideCellAtBoundary(cell, secretionUptakeValFibro)
                        # check for activated fibroblast to myofibroblast transition 
                        if fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] > TGFMyoThreshold and cell.dict['state'] == 1: # checking if fibroblast should become myofibroblast/count exposure time
                            cell.dict['timeChronExposure']  += 1
                            if cell.dict['timeChronExposure'] > reqTimeExposed:
                                cell.dict['state'] = 2 # becomes myofibroblast
                                fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] =  fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] - TGFMyoThreshold # simulating uptake from binding that transitioned fibroblast to myofibroblast
                        
                if fieldTNF[cell.xCOM, cell.yCOM, cell.zCOM] > fibroblastTNFApoptosisThreshold and fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] < TGFBlockApoptosisThreshold and np.random.uniform() < fibroblastApoptosisProb: 
                    cell.dict['time2apop'] = cell.dict['time2apop'] + 1
                    if cell.dict['time2apop'] >= apopTime: # referenced Kelley's code for this time
                        cell.targetVolume = 0
                        cell.lambdaVolume=1000000 
                        self.remove_all_cell_fpp_links(cell) 
                        TNFvalue = fieldTNF[cell.xCOM, cell.yCOM, cell.zCOM]
                        fieldTNF[cell.xCOM, cell.yCOM, cell.zCOM] = TNFvalue - fibroblastTNFApoptosisThreshold # remove TNF from binding that induced apop
                elif fieldTNF[cell.xCOM, cell.yCOM, cell.zCOM] > fibroblastTNFApoptosisThreshold and fieldTGF[cell.xCOM, cell.yCOM, 1] > TGFBlockApoptosisThreshold: # if there was enough TNF to cause apop but the TGF blocked it then remove that amount of TGF
                    TGFvalue = fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM]
                    fieldTGF[cell.xCOM, cell.yCOM, cell.zCOM] = TGFvalue - TGFBlockApoptosisThreshold # remove TGF from binding that blocked apop
                
      
                # Increment lag counters 
                cell.dict['time2divide'] -= 1 
                #print('fibro division countdown', cell.dict['time2divide'])
                cell.dict['time2activate'] -=1
           
    def finish(self):
        # this function may be called at the end of simulation - used very infrequently though
        return

    def on_stop(self):
        # this gets called each time user stops simulation
        return   
     
class FibroblastMitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)
    
    def step(self, mcs):
        # loop through all fibroblast and if lag times are over then divide, activate, or apoptose
        for cell in self.cell_list_by_type(self.FIBROBLAST):
            # divide cell with possibility of changing type
            if cell.dict['time2divide'] == 0:      
                self.divide_cell_random_orientation(cell)
            

            if cell.dict['time2activate'] == 0: 
                cell.dict['state'] = 1 
                
    def update_attributes(self):
        # Clone cell
        self.clone_parent_2_child()    
        
class M2MitosisSteppable(MitosisSteppableBase):
    def __init__(self,frequency=1):
        MitosisSteppableBase.__init__(self,frequency)

    def step(self, mcs):

        for cell in self.cell_list_by_type(self.MACROPHAGE):
            # divide cell if not proliferated and if an M2
            if cell.dict['type'] == 2 and cell.dict['prolifCount'] == 0 and np.random.uniform() < M2prolifProb:   
               cell.dict['prolifCount'] = 1
               self.divide_cell_random_orientation(cell)
               

    def update_attributes(self):
       # Clone cell
        self.clone_parent_2_child() 
        
