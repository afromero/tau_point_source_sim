# tau_point_source_sim

**1.Requirements:**  
	- Python 2.7  
	- Matplotlib, scipy, and numpy. We recommend you use anaconda to install python 2.7 so that you can automatically meet these requirements.  
	- A clone or fork of repo that calculates the isotropoic nuTau acceptance [nutau_acceptance](https://github.com/swissel/nutau_acceptance)   
	- Lookup tables from [nuTauSim](https://github.com/harmscho/NuTauSim/)   
	- Lookup tables of the peak electric fields from ZHAireS, based on parameterizations of electric fields at ANITA detector altitudes -or- electric field lookup tables for mountaintop altitudes  

**2. Environment Setup:**  

Set the folowing environment variables, either manually on the commandline or in your ~/.bashrc file.   
	- `TAU_ACC_DIR` : the directory where nutau_acceptance [nutau_acceptance](https://github.com/swissel/nutau_acceptance)  is installed  
	- `TAU_ACC_LUT_DIR` : the directory where runs of [nuTauSim](https://github.com/harmscho/NuTauSim/) are stored. The names of the LUT files should be in the format: os.environ['TAU_ACC_LUT_DIR']+'/'+self.ice+'km_ice_midCS_stdEL/LUT_'+self.TEScall+'_eV.npz (see decay_properties.py)  
	- `TAU_ACC_EFIELD_LUT_DIR` : the directory where the paramerizations and/or efield LUTs from ZHAireS are stored. The ANITA parameterizations should be called "anita_generic_parameterization.npz".   

**3. TODO:**  
	- Replace the hard-coded names of the lookup tables with user input file names  


