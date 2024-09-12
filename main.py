
import matplotlib as mpl
import time
import tikzplotlib
import matlab.engine
from functions import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})
import pandas as pd
from numpy.random import uniform
import scipy as scy


""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
fraction = 1#1.5
dpi = 300
PgWidthPt = 236#245
defBack = mpl.get_backend()


""" for plotting histogram and averaging over lambda """
n_bins = 20

""" for MwG"""
burnIn = 50

betaG = 1e-4
betaD = 1e-10  # 1e-4

""" for B_inve"""
tol = 1e-8

df = pd.read_excel('ExampleOzoneProfiles.xlsx')
#print the column names
print(df.columns)

#get the values for a given column
press = df['Pressure (hPa)'].values #in hectpascal or millibars
O3 = df['Ozone (VMR)'].values
minInd = 7
maxInd = 44
pressure_values = press[minInd:maxInd]
VMR_O3 = O3[minInd:maxInd]
scalingConstkm = 1e-3
# https://en.wikipedia.org/wiki/Pressure_altitude
# https://www.weather.gov/epz/wxcalc_pressurealtitude
heights = 145366.45 * (1 - ( press /1013.25)**0.190284 ) * 0.3048 * scalingConstkm

height_values = heights[minInd:maxInd]

""" analayse forward map without any real data values"""

MinH = height_values[0]
MaxH = height_values[-1]
R = 6371 # earth radiusin km
ObsHeight = 500 # in km

''' do svd for one specific set up for linear case and then exp case'''

#find best configuration of layers and num_meas
#so that cond(A) is not inf
#exp case first
SpecNumMeas = 105
SpecNumLayers = len(height_values)

# find minimum and max angle in radians
# min and max angle are defined by the height values of the retrived profile

MaxAng = np.arcsin((height_values[-1]+ R) / (R + ObsHeight))
MinAng = np.arcsin((height_values[0] + R) / (R + ObsHeight))
meas_ang = np.linspace(MinAng, MaxAng, SpecNumMeas)
A_lin, tang_heights_lin, extraHeight = gen_sing_map(meas_ang,height_values,ObsHeight,R)


ATA_lin = np.matmul(A_lin.T,A_lin)
#condition number for A
A_lin = A_lin
A_linu, A_lins, A_linvh = np.linalg.svd(A_lin)
cond_A_lin =  np.max(A_lins)/np.min(A_lins)
print("normal: " + str(orderOfMagnitude(cond_A_lin)))



#to test that we have the same dr distances
tot_r = np.zeros(SpecNumMeas)
#calculate total length
for j in range(0, SpecNumMeas):
    tot_r[j] = (np.sqrt( ( extraHeight + R)**2 - (tang_heights_lin[j] + R )**2) )
print('Distance through layers check: ' + str(np.allclose( sum(A_lin.T), tot_r)))









##


# graph Laplacian
# direchlet boundary condition
NOfNeigh = 2#4
neigbours = np.zeros((len(height_values),NOfNeigh))
# neigbours[0] = np.nan, np.nan, 1, 2
# neigbours[-1] = len(height_values)-2, len(height_values)-3, np.nan, np.nan
# neigbours[0] = np.nan, 1
# neigbours[-1] = len(height_values)-2, np.nan
for i in range(0,len(height_values)):
    neigbours[i] = i-1, i+1
    #neigbours[i] = i-2, i-1, i+1, i+2#, i+3 i-3,


neigbours[neigbours >= len(height_values)] = np.nan
neigbours[neigbours < 0] = np.nan

L = generate_L(neigbours)
startInd = 22
#L[startInd-6, startInd-6] = L[startInd-6, startInd-6] * 2
L[startInd-10:startInd, startInd-10:startInd] = L[startInd-10:startInd, startInd-10:startInd] * 3
L[startInd::, startInd::] = L[startInd::, startInd::] * 10

L[startInd-10, startInd-10] = - L[startInd-10, startInd-11] - L[startInd-10, startInd-9]
#L[startInd-6, startInd-5] = L[startInd-6, startInd-5] * 3
#L[startInd-6, startInd-6] = - L[startInd-6, startInd-5] - L[startInd-6, startInd-7]

L[startInd-1, startInd] = L[startInd-1, startInd] * 5
L[startInd, startInd-1] = L[startInd, startInd-1] * 5

# L[startInd-6, startInd-5] = L[startInd-6, startInd-5] * 3
# L[startInd-5, startInd-2] = L[startInd-5, startInd-6] * 3
L[startInd-1, startInd-1] = -L[startInd-1, startInd-2] - L[startInd-1, startInd] #-L[startInd, startInd-2] - L[startInd, startInd+2]
L[startInd, startInd]= -L[startInd-1, startInd] - L[startInd+1, startInd] #-L[startInd, startInd-2] - L[startInd, startInd+2]

#L[startInd+1, startInd+1] = -L[startInd+1, startInd+1-1] - L[startInd+1,startInd+1+1] -L[startInd+1, startInd+1-2] - L[startInd+1, startInd+1+2]
# L[16, 16] = 13

np.savetxt('GraphLaplacian.txt', L, header = 'Graph Lalplacian', fmt = '%.15f', delimiter= '\t')



#taylor exapnsion for f to do so we need y (data)

''' load data and pick wavenumber/frequency'''
#
##check absoprtion coeff in different heights and different freqencies
filename = 'tropical.O3.xml'

#VMR_O3, height_values, pressure_values = testReal.get_data(filename, ObsHeight * 1e3)
#[parts if VMR_O3 * 1e6 = ppm], [m], [Pa] = [kg / (m s^2) ]\
#height_values = np.around(height_values * 1e-3,2)#in km 1e2 # in cm
#d_height = (height_values[1::] - height_values[0:-1] )
#d_height = layers[1::] - layers[0:-1]
N_A = constants.Avogadro # in mol^-1
k_b_cgs = constants.Boltzmann * 1e7#in J K^-1
R_gas = N_A * k_b_cgs # in ..cm^3

# https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html
temperature = get_temp_values(heights)
temp_values = temperature[minInd:maxInd]
#x = VMR_O3 * N_A * pressure_values /(R_gas * temp_values)#* 1e-13
#https://hitran.org/docs/definitions-and-units/
#files = '/home/lennartgolks/Python/firstModelCheck/634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects
files = '634f1dc4.par' #/home/lennartgolks/Python /Users/lennart/PycharmProjects

my_data = pd.read_csv(files, header=None)
data_set = my_data.values

size = data_set.shape
wvnmbr = np.zeros((size[0],1))
S = np.zeros((size[0],1))
F = np.zeros((size[0],1))
g_air = np.zeros((size[0],1))
g_self = np.zeros((size[0],1))
E = np.zeros((size[0],1))
n_air = np.zeros((size[0],1))
g_doub_prime= np.zeros((size[0],1))


for i, lines in enumerate(data_set):
    wvnmbr[i] = float(lines[0][5:15]) # in 1/cm
    S[i] = float(lines[0][16:25]) # in cm/mol
    F[i] = float(lines[0][26:35])
    g_air[i] = float(lines[0][35:40])
    g_self[i] = float(lines[0][40:45])
    E[i] = float(lines[0][46:55])
    n_air[i] = float(lines[0][55:59])
    g_doub_prime[i] = float(lines[0][155:160])


#load constants in si annd convert to cgs units by multiplying
h = scy.constants.h #* 1e7#in J Hz^-1
c_cgs = constants.c * 1e2# in m/s
k_b_cgs = constants.Boltzmann #* 1e7#in J K^-1
#T = temp_values[0:-1] #in K
N_A = constants.Avogadro # in mol^-1
R = constants.gas_constant


mol_M = 48 #g/mol for Ozone
#ind = 293
ind = 623
#pick wavenumber in cm^-1
v_0 = wvnmbr[ind][0]#*1e2
#wavelength
lamba = 1/v_0
f_0 = c_cgs*v_0
print("Frequency " + str(np.around(v_0*c_cgs/1e9,2)) + " in GHz")

C1 =2 * scy.constants.h * scy.constants.c**2 * v_0**3 * 1e8
C2 = scy.constants.h * scy.constants.c * 1e2 * v_0  / (scy.constants.Boltzmann * temp_values )
#plancks function
Source = np.array(C1 /(np.exp(C2) - 1) ).reshape((SpecNumLayers,1))

#differs from HITRAN, implemented as in Urban et al
T_ref = 296 #K usually
p_ref = pressure_values[0]




'''weighted absorption cross section according to Hitran and MIPAS instrument description
S is: The spectral line intensity (cm^−1/(molecule cm^−2))
f_broad in (1/cm^-1) is the broadening due to pressure and doppler effect,
 usually one can describe this as the convolution of Lorentz profile and Gaussian profile
 VMR_O3 is the ozone profile in units of molecule (unitless)
 has to be extended if multiple gases are to be monitored
 I multiply with 1e-4 to go from cm^2 to m^2
 '''
f_broad = 1
w_cross =  VMR_O3 * f_broad * 1e-4
#w_cross[0], w_cross[-1] = 0, 0

#from : https://hitran.org/docs/definitions-and-units/
HitrConst2 = 1.4387769 # in cm K

# internal partition sum
Q = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ temp_values)
Q_ref = g_doub_prime[ind,0] * np.exp(- HitrConst2 * E[ind,0]/ 296)
LineInt = S[ind,0] * Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))
LineIntScal =  Q_ref / Q * np.exp(- HitrConst2 * E[ind,0]/ temp_values)/ np.exp(- HitrConst2 * E[ind,0]/ 296) * (1 - np.exp(- HitrConst2 * wvnmbr[ind,0]/ temp_values))/ (1- np.exp(- HitrConst2 * wvnmbr[ind,0]/ 296))

''' calculate model depending on where the Satellite is and 
how many measurements we want to do in between the max angle and min angle
 or max height and min height..
 we specify the angles
 because measurment will collect more than just the stuff around the tangent height'''

#take linear
num_mole = 1 / ( scy.constants.Boltzmann )#* temp_values)

AscalConstKmToCm = 1e3
#1e2 for pressure values from hPa to Pa
A_scal = pressure_values.reshape((SpecNumLayers,1)) * 1e2 * LineIntScal * Source * AscalConstKmToCm/ ( temp_values)
scalingConst = 1e11
#theta =(num_mole * w_cross.reshape((SpecNumLayers,1)) * Source * scalingConst )
theta = num_mole * w_cross.reshape((SpecNumLayers,1)) * scalingConst * S[ind,0]




''' caclutlate non-linear terms'''


nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, num_mole * w_cross.reshape((SpecNumLayers,1)) * S[ind,0], AscalConstKmToCm, SpecNumLayers, SpecNumMeas )

sim_A = A_lin * A_scal.T * nonLin
sim_ATA = np.matmul(sim_A.T,sim_A)
Au, As, Avh = np.linalg.svd(sim_A)
cond_A =  np.max(A_lins)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(sim_ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
#theta[0] = 0
#theta[-1] = 0
sim_Ax = np.matmul(sim_A, theta)




#convolve measurements and add noise
#y = add_noise(Ax, 0.01)
y, gamma = add_noise(sim_Ax, 40)
#y[y<=0] = 0

#y = np.loadtxt('NonLinDataY.txt').reshape((SpecNumMeas,1))

fig3, ax1 = plt.subplots()
ax1.plot(sim_Ax, tang_heights_lin)
ax1.scatter(y, tang_heights_lin)
ax1.plot(y, tang_heights_lin)
plt.show()

sim_ATy = np.matmul(sim_A.T, y)

np.savetxt('NonLinDataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
np.savetxt('NonLinForWardMatrix.txt', sim_A, header = 'Forward Matrix A', fmt = '%.15f', delimiter= '\t')
np.savetxt('height_values.txt', height_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('tan_height_values.txt', tang_heights_lin, fmt = '%.15f', delimiter= '\t')
np.savetxt('pressure_values.txt', pressure_values, fmt = '%.15f', delimiter= '\t')
np.savetxt('VMR_O3.txt', VMR_O3, fmt = '%.15f', delimiter= '\t')

print('bla')



'''give the non linear term and forward model a constant ozone Profile'''
first_nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, np.mean(VMR_O3) * num_mole  * f_broad * 1e-4 * S[ind,0] * np.ones((SpecNumLayers,1)), AscalConstKmToCm, SpecNumLayers, SpecNumMeas )

first_A = A_lin * A_scal.T * first_nonLin
first_ATA = np.matmul(first_A.T,first_A)
first_ATy = np.matmul(first_A.T, y)
"""start the mtc algo with first guesses of noise and lumping const delta"""


vari = np.zeros((len(theta)-2,1))

for j in range(1,len(theta)-1):
    vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])

#find minimum for first guesses
'''params[1] = delta
params[0] = gamma'''
def MinLogMargPost(params):#, coeff):

    # gamma = params[0]
    # delta = params[1]
    gamma = params[0]
    lamb = params[1]
    if lamb < 0  or gamma < 0:
        return np.nan

    n = SpecNumLayers
    m = SpecNumMeas

    Bp = first_ATA + lamb * L


    B_inv_A_trans_y, exitCode = gmres(Bp, first_ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(first_A, L,  lamb)
    F = f(first_ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [1/(np.max(sim_Ax) * 0.01)**2,1/(np.mean(vari))*(np.max(sim_Ax) * 0.01)**2])

lam0 = minimum[1]
print(minimum)
##
"""prepare f and g for sampling"""
lam= np.logspace(-5,15,500)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (first_ATA + lam[j] * L)

    B_inv_A_trans_y, exitCode = gmres(B, first_ATy[0::, 0], tol=tol, restart=25)
    #print(exitCode)

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
    if np.linalg.norm(first_ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(first_ATy[0::, 0])<=tol:
        f_func[j] = f(first_ATy, y, B_inv_A_trans_y)
    else:
        f_func[j] = np.nan

    g_func[j] = g(first_A, L, lam[j])


np.savetxt('f_func.txt', f_func, fmt = '%.15f')
np.savetxt('g_func.txt', g_func, fmt = '%.15f')
np.savetxt('lam.txt', lam, fmt = '%.15f')

##
""" taylor series around lam_0 """

B = (first_ATA + lam0 * L)

B_inv_A_trans_y0, exitCode = gmres(B, first_ATy[0::, 0], tol=tol, restart=25)

CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y0)

B_inv_L0 = np.zeros(np.shape(B))

for i in range(len(B)):
    B_inv_L0[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))


B_inv_L_2 = np.matmul(B_inv_L0, B_inv_L0)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L0)

f_0_1 = np.matmul(np.matmul(first_ATy[0::, 0].T, B_inv_L0), B_inv_A_trans_y0)
f_0_2 = -1 * np.matmul(np.matmul(first_ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
f_0_3 = 1 * np.matmul(np.matmul(first_ATy[0::, 0].T, B_inv_L_3), B_inv_A_trans_y0)

g_0_1 = np.trace(B_inv_L0)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 / 6 * np.trace(B_inv_L_3)


##

'''do the sampling'''
number_samples = 10000


#inintialize sample
gamma0 = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
#0.275#1/(2*np.mean(vari))0.1#
lambda0 = minimum[1]#deltas[0]/gammas[0]
#deltas[0] =  minimum[1] * minimum[0]

Bu, Bs, Bvh = np.linalg.svd(B)
cond_B =  np.max(Bs)/np.min(Bs)
print("Condition number B: " + str(orderOfMagnitude(cond_B)))



alphaG = 1
alphaD = 1
rate = f(first_ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG

f_new = f(first_ATy, y,  B_inv_A_trans_y0)
wLam = 6e4#0.7e3

startTime = time.time()
lambdas, gammas, k = MHwG(number_samples, burnIn, y, L, first_ATA, SpecNumMeas, SpecNumLayers, B_inv_A_trans_y0, first_ATy, wLam, lambda0, gamma0, tol, f_0_1, f_0_2, f_0_3, g_0_1, g_0_2, g_0_3)
elapsed = time.time() - startTime
print('MTC Done in ' + str(elapsed) + ' s')



print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
deltas = lambdas * gammas
np.savetxt('samples.txt', np.vstack((gammas[burnIn::], deltas[burnIn::], lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')

#delt_aav, delt_diff, delt_ddiff, delt_itau, delt_itau_diff, delt_itau_aav, delt_acorrn = uWerr(deltas, acorr=None, s_tau=1.5, fast_threshold=5000)


eng = matlab.engine.start_matlab()
eng.Run_Autocorr_Ana_MTC(nargout=0)
eng.quit()

AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')
#IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'

with open("auto_corr_dat.txt") as fID:
    for n, line in enumerate(fID):
       if n == 1:
            IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
            break



#refine according to autocorrelation time
new_lamb = lambdas[burnIn::math.ceil(IntAutoLam)]
#SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
new_gam = gammas[burnIn::math.ceil(IntAutoGam)]
#SetGamma = new_gam[np.random.randint(low = 0,high =len(new_gam),size =1)]
new_delt = deltas[burnIn::math.ceil(IntAutoDelt)]
#SetDelta = new_delt[np.random.randint(low = 0,high =len(new_delt),size =1)]

##


mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)

fig, axs = plt.subplots(figsize = (7,  2))
# We can set the number of bins with the *bins* keyword argument.
axs.hist(new_gam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
#axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
axs.set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
#axs.set_xlabel(str(len(new_gam)) + ' effective $\gamma$ samples')
tikzplotlib.save("HistoResults1.tex",axis_height='3cm', axis_width='7cm')
plt.close()


fig, axs = plt.subplots( )
axs.hist(new_delt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
axs.set_title(str(len(new_delt)) + ' $\delta$ samples, the prior precision')
#axs.set_xlabel(str(len(new_delt)) + ' $\delta$ samples, the prior precision')
tikzplotlib.save("HistoResults2.tex",axis_height='3cm', axis_width='7cm')
plt.close()

fig, axs = plt.subplots( )
axs.hist(new_lamb,bins=n_bins, color = 'k')#10)
#axs.set_xlabel(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
#axs[2].xaxis.set_major_formatter(scientific_formatter)
#axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
axs.set_title(str(len(new_lamb)) + ' $\lambda$ samples, the regularization parameter')
#plt.savefig('HistoResults.png')
#plt.show()

tikzplotlib.save("HistoResults3.tex",axis_height='3cm', axis_width='7cm')


##
f_mode = f(first_ATy, y, B_inv_A_trans_y0)
BinHist = 200#n_bins
lambHist, lambBinEdges = np.histogram(new_lamb, bins= BinHist, density =True)


mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 10})

fCol = [0, 144/255, 178/255]
gCol = [230/255, 159/255, 0]
gmresCol = [204/255, 121/255, 167/255]


fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=0.85))#, dpi = dpi)

axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')
axs.scatter(minimum[1],f_mode, color = gmresCol, zorder=0, marker = 's')
axs.set_yscale('log')
axs.set_xlabel('$\lambda$')
axs.set_ylabel('$f(\lambda)$')#, color = fCol)
axs.tick_params(axis = 'y',  colors=fCol, which = 'both')

ax2 = axs.twinx() # ax1 and ax2 share y-axis
ax2.plot(lam,g_func, color = gCol, zorder = 2, linestyle=  'dashed')
ax2.scatter(minimum[1],g(first_A, L, minimum[1]), color = gmresCol, zorder=0, marker = 's')
ax2.set_ylabel('$g(\lambda)$')#,color = gCol)
ax2.tick_params(axis = 'y', colors= gCol)
axs.set_xscale('log')
axins = axs.inset_axes([0.05,0.5,0.4,0.45])

axins.plot(lam,f_func, color = fCol, zorder=3, linestyle=  'dotted', linewidth = 3)
delta_lam = lambBinEdges - minimum[1]
axins.plot(lambBinEdges,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, 0), color = 'k', linewidth = 1, zorder = 1, label = 'Taylor series' )
axs.plot(lambBinEdges,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, 0), color = 'k', linewidth = 1, zorder = 2, label = 'Taylor series' )

axins.scatter(minimum[1],f_mode, color = gmresCol, s= 95, zorder=0, marker = 's', label = r'\texttt{optimize.fmin()}')

x_val = axins.get_xticks()
axins.tick_params(axis='y', which='both',  left=False, labelleft=False)

axins.tick_params(axis='x', which  = 'both', labelbottom=False )

axins.set_xlim(min(new_lamb),max(new_lamb))
axins.set_ylim(5e8,9e8)
axins.set_xlabel('$\lambda$')
axins.set_xlim([np.mean(lambdas) - np.sqrt(np.var(lambdas)), 1.5*np.mean(lambdas) + np.sqrt(np.var(lambdas))])# apply the x-limits
axins.set_yscale('log')
axins.set_xscale('log')
axin2 = axins.twinx()
axin2.spines['top'].set_visible(False)
axin2.spines['right'].set_visible(False)
axin2.spines['bottom'].set_visible(False)
axin2.spines['left'].set_visible(False)
axin2.tick_params(axis = 'y', which = 'both',labelright=False, right=False)
axin2.tick_params(axis='y', which='both', length=0)
axin2.plot(lam,g_func, color = gCol, zorder=3, linestyle=  'dashed', linewidth = 3)

axin2.plot(lambBinEdges, g_tayl(delta_lam, g(first_A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, 0,0, 0), color = 'k', linewidth = 1, zorder = 2 )
axin2.scatter(minimum[1],g(first_A, L, minimum[1]), color = gmresCol, s=95, zorder=0, marker = 's')
axin2.set_ylim(360,460)
axin2.set_xlim(min(new_lamb),max(new_lamb))
axin2.set_xscale('log')

ax2.plot(lambBinEdges, g_tayl(delta_lam, g(first_A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, 0,0, 0), color = 'k', linewidth = 1, zorder = 1)

lines, lab0 = axins.get_legend_handles_labels()

axs.spines['right'].set_visible(False)

axs.spines['left'].set_color('k')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color('k')

ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

#tikzplotlib.save("f_and_g_paper.tex")

#plt.savefig('f_and_g_paper.png',bbox_inches='tight')
plt.show()
#for legend
# tikzplotlib_fix_ncols(fig)
# tikzplotlib.save("f_and_g_papers.tex")


##
plt.close()
mpl.use('pgf')
pgf_params = { "pgf.texsystem": "pdflatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
'axes.labelsize': 10,  # -> axis labels
'legend.fontsize': 10}
mpl.rcParams.update(pgf_params)
fig.savefig('f_and_g_paper.pgf', bbox_inches='tight', dpi = 300)


##
'''find ozone profile and update non Linear A matrix'''

#draw paramter samples
paraSamp = 100#n_bins
Results = np.zeros((paraSamp+1,len(theta)))

Results[0, :] = np.mean(VMR_O3) * num_mole  * scalingConst * f_broad * 1e-4 * S[ind,0] * np.ones((SpecNumLayers))
relError = np.zeros((paraSamp+1,1))
NormRes = np.zeros(paraSamp+1)
xTLxRes = np.zeros(paraSamp+1)
SetGammas = new_gam[np.random.randint(low=0, high=len(new_gam), size=paraSamp)]
SetDeltas  = new_delt[np.random.randint(low=0, high=len(new_delt), size=paraSamp)]
update_A = first_A
update_ATA = first_ATA
update_ATy = first_ATy
startTimeX = time.time()
for p in range(1,paraSamp+1):

    SetGamma = SetGammas[p-1] #minimum[0]
    SetDelta  = SetDeltas[p-1] #minimum[1]
    W = np.random.multivariate_normal(np.zeros(len(update_A)), np.eye(len(update_A)))
    v_1 = np.sqrt(SetGamma) *  update_A.T @ W
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(SetDelta) * W2

    SetB = SetGamma * update_ATA + SetDelta * L
    RandX = (SetGamma * update_ATy[0::, 0] + v_1 + v_2)

    update_B_inv_A_trans_y, exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)


    if exitCode != 0:
        print(exitCode)

    Results[p, :] = update_B_inv_A_trans_y

    NormRes[p] = np.linalg.norm( np.matmul(update_A,update_B_inv_A_trans_y) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(update_B_inv_A_trans_y.T, L), update_B_inv_A_trans_y))

    relError[p] = np.linalg.norm(update_B_inv_A_trans_y -  Results[p-1, :] ) /np.linalg.norm( Results[p-1, :])
    print('Relative Error for ' + str(p) + 'th sample: ' + str(relError[p]))

    #update forward model matrix
    update_nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, update_B_inv_A_trans_y.reshape((SpecNumLayers,1))/ scalingConst, AscalConstKmToCm,
                        SpecNumLayers, SpecNumMeas)


    update_A = A_lin * A_scal.T * update_nonLin
    update_ATA = np.matmul(update_A.T,update_A)
    update_ATy = np.matmul(update_A.T, y)

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))

np.savetxt('O3Res.txt', Results/(num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst), fmt = '%.15f', delimiter= '\t')


###
plt.close('all')


DatCol =  'gray' # 'k'"#332288"#"#009E73"
ResCol = "#1E88E5"#"#0072B2"
TrueCol = 'k'#'limegreen'
x = np.mean(theta) * np.ones((SpecNumLayers,1)) / (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)

mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)

fig, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(x,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = 'None', color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =8)

for n in range(0,5):#,paraSamp,15):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)
    ax1.plot(Sol,height_values,marker= '+',label = r'$x_{'+str(n)+'}$', zorder = 1, linewidth = 0.5, markersize = 5)
    #label = r'$\bm{x}_{'+str(n)+'}$'
ax1.set_xlabel(r'Ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

legend = ax1.legend(handles = [handles[1], handles[2],handles[3],handles[4],handles[5]], loc='upper right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])

# turn on when saving as .tex
#ax2.set_xlabel(r'Spectral radiance in $\frac{\text{W cm}}{\text{m}^2 \text{ sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)

plt.show()
#import tikzplotlib
#tikzplotlib_fix_ncols(fig)
#tikzplotlib.save("NonLinRes.tex")

print('bla')
