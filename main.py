import numpy as np
import matplotlib as mpl
#from importetFunctions import *
import time
import pickle as pl
import tikzplotlib
import matlab.engine
from functions import *
#from errors import *
from scipy import constants, optimize
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
import pandas as pd
from numpy.random import uniform, normal, gamma
import scipy as scy
#from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

#mpl.rc('text.latex', preamble=r"\boldmath")

""" for plotting figures,
PgWidth in points, either collumn width page with of Latex"""
fraction = 1.5
dpi = 300
PgWidthPt = 245

defBack = mpl.get_backend()
pgf_params = { "pgf.texsystem": "pdflatex",
    'text.usetex': True,
    'pgf.rcfonts': False,
'axes.labelsize': 12,  # -> axis labels
'legend.fontsize': 12}

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


# fig, axs = plt.subplots(tight_layout=True)
# plt.scatter(range(len(tang_heights_lin)),tang_heights_lin)
# #plt.show()

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
startInd = 24
L[startInd::, startInd::] = L[startInd::, startInd::] * 10
L[startInd, startInd] = -L[startInd, startInd-1] - L[startInd, startInd+1] #-L[startInd, startInd-2] - L[startInd, startInd+2]

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


""" plot forward model values """
numDensO3 =  N_A * press * 1e2 * O3 / (R * temp_values[0,:]) * 1e-6



''' caclutlate non-linear terms'''


nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, num_mole * w_cross.reshape((SpecNumLayers,1)) * S[ind,0], AscalConstKmToCm, SpecNumLayers, SpecNumMeas )

A = A_lin * A_scal.T * nonLin
ATA = np.matmul(A.T,A)
Au, As, Avh = np.linalg.svd(A)
cond_A =  np.max(A_lins)/np.min(As)
print("normal: " + str(orderOfMagnitude(cond_A)))

ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
cond_ATA = np.max(ATAs)/np.min(ATAs)
print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
#theta[0] = 0
#theta[-1] = 0
Ax = np.matmul(A, theta)

#convolve measurements and add noise
#y = add_noise(Ax, 0.01)
#y[y<=0] = 0

y = np.loadtxt('NonLinDataY.txt').reshape((SpecNumMeas,1))


ATy = np.matmul(A.T, y)

#np.savetxt('NonLinDataY.txt', y, header = 'Data y including noise', fmt = '%.15f')
np.savetxt('NonLinForWardMatrix.txt', A, header = 'Forward Matrix A', fmt = '%.15f', delimiter= '\t')

fig2, ax = plt.subplots()
#plt.plot( VMR_O3 * 1e6 ,layers)
plt.plot( y,tang_heights_lin)
#ax.set_ylim([tang_heights_lin])
plt.xlabel('Volume Mixing Ratio Ozone in ppm')
plt.ylabel('Height in km')
plt.savefig('measurement.png')
plt.show()

print('bla')



'''give the non linear term and forward model a constant ozone Profile'''
nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, np.mean(num_mole * w_cross.reshape((SpecNumLayers,1)) * S[ind,0]) * np.ones((SpecNumLayers,1)), AscalConstKmToCm, SpecNumLayers, SpecNumMeas )

A = A_lin * A_scal.T * nonLin
ATA = np.matmul(A.T,A)
ATy = np.matmul(A.T, y)
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

    Bp = ATA + lamb * L


    B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    G = g(A, L,  lamb)
    F = f(ATy, y,  B_inv_A_trans_y)

    return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)

#minimum = optimize.fmin(MargPostU, [5e-5,0.5])
minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01)**2,1/(np.mean(vari))*(np.max(Ax) * 0.01)**2])

lam0 = minimum[1]
print(minimum)
##
"""prepare f and g for sampling"""
lam= np.logspace(-5,15,500)
f_func = np.zeros(len(lam))
g_func = np.zeros(len(lam))



for j in range(len(lam)):

    B = (ATA + lam[j] * L)

    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    #print(exitCode)

    CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y)
    if np.linalg.norm(ATy[0::, 0]- CheckB_inv_ATy)/np.linalg.norm(ATy[0::, 0])<=tol:
        f_func[j] = f(ATy, y, B_inv_A_trans_y)
    else:
        f_func[j] = np.nan

    g_func[j] = g(A, L, lam[j])


np.savetxt('f_func.txt', f_func, fmt = '%.15f')
np.savetxt('g_func.txt', g_func, fmt = '%.15f')
np.savetxt('lam.txt', lam, fmt = '%.15f')

##
""" taylor series around lam_0 """

B = (ATA + lam0 * L)

B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#print(exitCode)

CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y0)



B_inv_L0 = np.zeros(np.shape(B))

for i in range(len(B)):
    B_inv_L0[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
    if exitCode != 0:
        print('B_inv_L ' + str(exitCode))

#relative_tol_L = tol
#CheckB_inv_L = np.matmul(B, B_inv_L)
#print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)

B_inv_L_2 = np.matmul(B_inv_L0, B_inv_L0)
B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L0)
B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L0)
B_inv_L_6 = np.matmul(B_inv_L_4, B_inv_L_2)


f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L0), B_inv_A_trans_y0)
f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0)
f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y0)
#f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)


g_0_1 = np.trace(B_inv_L0)
g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
g_0_3 = 1 /6 * np.trace(B_inv_L_3)
g_0_4 = -1 /24 * np.trace(B_inv_L_4)
g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)




##

'''do the sampling'''
number_samples = 10000


#inintialize sample
gamma0 = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
#0.275#1/(2*np.mean(vari))0.1#
lambda0 = minimum[1]#deltas[0]/gammas[0]
#deltas[0] =  minimum[1] * minimum[0]
ATy = np.matmul(A.T, y)
B = (ATA + lambda0 * L)

B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
if exitCode != 0:
    print(exitCode)

Bu, Bs, Bvh = np.linalg.svd(B)
cond_B =  np.max(Bs)/np.min(Bs)
print("Condition number B: " + str(orderOfMagnitude(cond_B)))



#wgam = 1e-5
#wdelt = 1e-1

alphaG = 1
alphaD = 1
rate = f(ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0
# draw gamma with a gibs step
shape = SpecNumMeas/2 + alphaD + alphaG

f_new = f(ATy, y,  B_inv_A_trans_y0)
#g_old = g(A, L,  lambdas[0])

def MHwG(number_samples, burnIn, lambda0, gamma0):
    wLam = 0.7e3#7e1

    alphaG = 1
    alphaD = 1
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lambda0

    B = (ATA + lambda0 * L)
    B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)

    #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
    if exitCode != 0:
        print(exitCode)

    shape = SpecNumMeas / 2 + alphaD + alphaG
    rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambda0


    for t in range(number_samples + burnIn-1):
        #print(t)

        # # draw new lambda
        lam_p = normal(lambdas[t], wLam)

        while lam_p < 0:
                lam_p = normal(lambdas[t], wLam)

        delta_lam = lam_p - lambdas[t]
        # B = (ATA + lam_p * L)
        # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
        # if exitCode != 0:
        #     print(exitCode)


        # f_new = f(ATy, y,  B_inv_A_trans_y)
        # g_new = g(A, L,  lam_p)
        #
        # delta_f = f_new - f_old
        # delta_g = g_new - g_old

        delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
        delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3

        log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam

        #accept or rejeict new lam_p
        u = uniform()
        if np.log(u) <= log_MH_ratio:
        #accept
            k = k + 1
            lambdas[t + 1] = lam_p
            #only calc when lambda is updated

            B = (ATA + lam_p * L)
            B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0,tol=tol, restart=25)
            #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)

            # if exitCode != 0:
            #         print(exitCode)

            f_new = f(ATy, y,  B_inv_A_trans_y)
            #g_old = np.copy(g_new)
            rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]

        else:
            #rejcet
            lambdas[t + 1] = np.copy(lambdas[t])




        gammas[t+1] = np.random.gamma(shape = shape, scale = 1/rate)

        #deltas[t+1] = lambdas[t+1] * gammas[t+1]

    return lambdas, gammas,k



startTime = time.time()
lambdas ,gammas, k = MHwG(number_samples, burnIn, lambda0, gamma0)
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

fig, axs = plt.subplots(3, 1,tight_layout=True)
# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(new_gam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
#axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
axs[0].set_title(str(len(new_gam)) + r' $\gamma$ samples, the noise precision')
axs[1].hist(new_delt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
axs[1].set_title(str(len(new_delt)) + ' $\delta$ samples, the prior precision')
axs[2].hist(new_lamb,bins=n_bins, color = 'k')#10)
#axs[2].xaxis.set_major_formatter(scientific_formatter)
#axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
axs[2].set_title(str(len(new_lamb)) + ' $\lambda$ samples, the regularization parameter')
plt.savefig('HistoResults.png')
plt.show()

print('bla')

##
f_mode = f(ATy, y, B_inv_A_trans_y0)
BinHist = 200#n_bins
lambHist, lambBinEdges = np.histogram(new_lamb, bins= BinHist, density =True)

mpl.use(defBack)
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
fCol = [0, 144/255, 178/255]
gCol = [230/255, 159/255, 0]
#gCol = [240/255, 228/255, 66/255]
#gCol = [86/255, 180/255, 233/255]
gmresCol = [204/255, 121/255, 167/255]
fig,axs = plt.subplots(figsize=set_size(PgWidthPt, fraction=fraction))#, dpi = dpi)

axs.plot(lam,f_func, color = fCol, zorder = 2, linestyle=  'dotted')

axs.scatter(minimum[1],f_mode, color = gmresCol, zorder=0, marker = 's')#

axs.set_yscale('log')
axs.set_xlabel('$\lambda$')
axs.set_ylabel('$f(\lambda)$')#, color = fCol)
axs.tick_params(axis = 'y',  colors=fCol, which = 'both')

ax2 = axs.twinx() # ax1 and ax2 share y-axis
ax2.plot(lam,g_func, color = gCol, zorder = 2, linestyle=  'dashed')
ax2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, zorder=0, marker = 's')
ax2.set_ylabel('$g(\lambda)$')#,color = gCol)
ax2.tick_params(axis = 'y', colors= gCol)
axs.set_xscale('log')
axins = axs.inset_axes([0.05,0.5,0.4,0.45])

axins.plot(lam,f_func, color = fCol, zorder=3, linestyle=  'dotted', linewidth = 3)
delta_lam = lambBinEdges - minimum[1]
axins.plot(lambBinEdges,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, f_0_4), color = 'k', linewidth = 1, zorder = 1, label = 'Taylor series' )
axs.plot(lambBinEdges,f_tayl(delta_lam, f_mode, f_0_1, f_0_2, f_0_3, f_0_4), color = 'k', linewidth = 1, zorder = 2, label = 'Taylor series' )

axins.scatter(minimum[1],f_mode, color = gmresCol, s= 95, zorder=0, marker = 's', label = r'\texttt{optimize.fmin()}')


x_val = axins.get_xticks()
axins.tick_params(axis='y', which='both',  left=False, labelleft=False)

axins.tick_params(axis='x', which  = 'both', labelbottom=False )
#axins.set_xticks([lam0])
#axins.set_xticklabels( [])#,labels=['','','','lam0'])
#axins.set_xticklabels(['lam0'])


#axins.tick_params(axis='y', which='both', length=0)
axins.set_xlim(min(new_lamb),max(new_lamb))
axins.set_ylim(5e8,9e8)
axins.set_xlabel('$\lambda$')
axins.set_xlim([np.mean(lambdas) - np.sqrt(np.var(lambdas)), 1.5*np.mean(lambdas) + np.sqrt(np.var(lambdas))])# apply the x-limits
axins.set_yscale('log')
axins.set_xscale('log')
axin2 = axins.twinx()
# # axin2.tick_params(axis='x',which  = 'minor', labelbottom=False)
# #
# # axin2.set_xticks( [lam0])#,labels=['','','','lam0'])
# # axin2.set_xticklabels(['lam0'])
#
#
axin2.spines['top'].set_visible(False)
axin2.spines['right'].set_visible(False)
axin2.spines['bottom'].set_visible(False)
axin2.spines['left'].set_visible(False)
axin2.tick_params(axis = 'y', which = 'both',labelright=False, right=False)
axin2.tick_params(axis='y', which='both', length=0)
axin2.plot(lam,g_func, color = gCol, zorder=3, linestyle=  'dashed', linewidth = 3)

axin2.plot(lambBinEdges, g_tayl(delta_lam, g(A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, g_0_4,g_0_5, g_0_6), color = 'k', linewidth = 1, zorder = 2 )
axin2.scatter(minimum[1],g(A, L, minimum[1]), color = gmresCol, s=95, zorder=0, marker = 's')
axin2.set_ylim(360,460)
axin2.set_xlim(min(new_lamb),max(new_lamb))
axin2.set_xscale('log')

ax2.plot(lambBinEdges, g_tayl(delta_lam, g(A, L, minimum[1]) ,g_0_1, g_0_2, g_0_3, g_0_4,g_0_5, g_0_6), color = 'k', linewidth = 1, zorder = 1)

lines, lab0 = axins.get_legend_handles_labels()

axs.spines['right'].set_visible(False)

axs.spines['left'].set_color('k')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_color('k')

ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)


#tikzplotlib.save("f_and_g_paper.tex")

plt.savefig('f_and_g_paper.png',bbox_inches='tight')
plt.show()
#for legend
# tikzplotlib_fix_ncols(fig)
#

##
plt.close()
mpl.use('pgf')
mpl.rcParams.update(pgf_params)
fig.savefig('f_and_g_paper.pgf', bbox_inches='tight', dpi = 300)


##
'''find ozone profile and update non Linear A matrix'''

#draw paramter samples
paraSamp = 100#n_bins
Results = np.zeros((paraSamp,len(theta)))
relError = np.zeros((paraSamp,1))
NormRes = np.zeros(paraSamp)
xTLxRes = np.zeros(paraSamp)
SetGammas = new_gam[np.random.randint(low=0, high=len(new_gam), size=paraSamp)]
SetDeltas  = new_delt[np.random.randint(low=0, high=len(new_delt), size=paraSamp)]

startTimeX = time.time()
for p in range(paraSamp):

    SetGamma = SetGammas[p] #minimum[0]
    SetDelta  = SetDeltas[p] #minimum[1]
    W = np.random.multivariate_normal(np.zeros(len(A)), np.eye(len(A)))
    v_1 = np.sqrt(SetGamma) *  A.T @ W
    W2 = np.random.multivariate_normal(np.zeros(len(L)), L)
    v_2 = np.sqrt(SetDelta) * W2

    SetB = SetGamma * ATA + SetDelta * L
    RandX = (SetGamma * ATy[0::, 0] + v_1 + v_2)

    B_inv_A_trans_y, exitCode = gmres(SetB, RandX, x0=B_inv_A_trans_y0, tol=tol)


    if exitCode != 0:
        print(exitCode)

    Results[p, :] = B_inv_A_trans_y

    NormRes[p] = np.linalg.norm( np.matmul(A,B_inv_A_trans_y) - y[0::,0])
    xTLxRes[p] = np.sqrt(np.matmul(np.matmul(B_inv_A_trans_y.T, L), B_inv_A_trans_y))

    relError[p] = np.linalg.norm(B_inv_A_trans_y -  Results[p-1, :] ) /np.linalg.norm( Results[p-1, :])
    print('Relative Error for ' + str(p) + 'th sample: ' + str(relError[p]))
    #update forward model matrix
    nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, B_inv_A_trans_y.reshape((SpecNumLayers,1))/ scalingConst, AscalConstKmToCm,
                        SpecNumLayers, SpecNumMeas)


    A = A_lin * A_scal.T * nonLin
    ATA = np.matmul(A.T,A)
    ATy = np.matmul(A.T, y)

elapsedX = time.time() - startTimeX
print('Time to solve for x ' + str(elapsedX/paraSamp))
##
mpl.use(defBack)
fig, ax = plt.subplots(figsize=set_size(245, fraction=fraction))
#ax.plot(range(paraSamp), NormRes)
ax.scatter( NormRes[:100], xTLxRes[:100])
#ax.plot(range(paraSamp), relError)
plt.show()

##
plt.close('all')
Sol= Results[2,:]/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
x = np.mean(Results,0 )/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#xerr = np.sqrt(np.var(Results / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst), 0)) / 2
#xerr = np.sqrt(np.var(Results,0)/(num_mole *S[ind,0]  * f_broad * 1e-4 * scalingConst)**2)/2
#XOPT = x_opt /(num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
#MargX = MargInteg/ (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)
mpl.use(defBack)
#mpl.use("png") bbox_inches='tight'
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.serif"] = "cmr"
fig3, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
 # ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color ="#d62728" )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values,marker = 'o', color = "#009E73", label = 'true profile', zorder=4)#,linewidth = 5)

# edgecolor = [0, 158/255, 115/255]
#line1 = ax1.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 10, zorder=0)
for n in range(0,10):#,paraSamp,15):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)
    ax1.plot(Sol,height_values,marker= '.',color = 'k',label = 'posterior samples ', zorder = 0, linewidth = 0.5)
#$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $' , markerfacecolor = 'none'
#ax1.plot(XOPT, height_values, color="#D55E00",  markeredgecolor="#D55E00", marker='v', zorder=3, label='regularized sol. ', markersize =9)
#line2 = ax1.errorbar(x,height_values,capsize=5, yerr = np.zeros(len(height_values)) ,color = MTCCol,zorder=5,markersize = 5, fmt = 'o',label = r'$\mathbf{x} \sim \pi(\mathbf{x} |\mathbf{y}, \mathbf{\theta} ) $')#, label = 'MC estimate')
#line3 = ax1.plot(MargX,height_values, color ="#0072B2" ,zorder=2, marker = 's', label = 'posterior mean ', markersize =9)
#E$_{\mathbf{x},\mathbf{\theta}| \mathbf{y}}[h(\mathbf{x})]$
# markersize = 6
#line4 = ax1.errorbar(x, height_values,capsize=5, xerr = xerr,color = MTCCol, fmt = 'o', markersize = 5,zorder=5)#, label = 'MC estimate')
#line5 = ax1.errorbar(MargX,height_values, color = MargCol, markeredgecolor= MargCol, capsize=5,  markersize = 6, zorder=3, fmt = 's')
#xerr =MargXErr/2,yerr = np.zeros(len(height_values))



#line5 = ax1.plot(x_opt/(num_mole * S[ind,0] * f_broad * 1e-4 * scalingConst),height_values, color = 'crimson', linewidth = 7, label = 'reg. sol.', zorder=1)

ax1.set_xlabel(r'Ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Handles = [handles[0], handles[1], handles[2]]
# Labels =  [labels[0], labels[1], labels[2]]
# LegendVertical(ax1, Handles, Labels, 90, XPad=-45, YPad=12)

#legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]], loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

#plt.ylabel('Height in km')
ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])
#ax2.set_xlim([min(y),max(y)])
#ax1.set_xlim([min(x)-max(xerr)/2,max(x)+max(xerr)/2])


ax2.set_xlabel(r'Spectral Ozone radiance in $\frac{W}{m^2 sr} \times \frac{1}{\frac{1}{cm}}$',labelpad=10)# color =dataCol,
ax2.tick_params(colors = "#d62728", axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)


fig3.savefig('FirstRecRes.png')#, dpi = dpi)

plt.show()

##

plt.close('all')


x = np.mean(theta) * np.ones((SpecNumLayers,1)) / (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams.update({'font.size': 12})
plt.rcParams["font.serif"] = "cmr"
fig3, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
 # ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color ="#d62728" )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values,marker = 'o', color = 'k', label = 'true profile', zorder=4)#color = "#009E73",linewidth = 5)
ax1.plot(x, height_values, marker='.', color='k', label='$x_0$', zorder=0, linewidth=0.5)

# edgecolor = [0, 158/255, 115/255]
#line1 = ax1.plot(VMR_O3,height_values, color = [0, 158/255, 115/255], linewidth = 10, zorder=0)

ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])

ax2.set_xlabel(r'Spectral Ozone radiance in $\frac{W}{m^2 sr} \times \frac{1}{\frac{1}{cm}}$',labelpad=10)# color =dataCol,
ax2.tick_params(colors = "#d62728", axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)

ax1.legend()
fig3.savefig('NonLinRes.png')

plt.show()


###
plt.close('all')

mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


DatCol =  'gray' # 'k'"#332288"#"#009E73"
ResCol = "#1E88E5"#"#0072B2"
TrueCol = 'k'#'limegreen'
x = np.mean(theta) * np.ones((SpecNumLayers,1)) / (num_mole * S[ind,0]  * f_broad * 1e-4 * scalingConst)

# mpl.use(defBack)
# mpl.rcParams.update(mpl.rcParamsDefault)
# plt.rcParams.update({'font.size': 10})
# plt.rcParams["font.serif"] = "cmr"
fig, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
# ax1 and ax2 share y-axis
line3 = ax2.scatter(y, tang_heights_lin, label = r'data', zorder = 0, marker = '*', color =DatCol )#,linewidth = 5

ax1 = ax2.twiny()
#ax1.scatter(VMR_O3,height_values,marker = 'o', facecolor = 'None', color = "#009E73", label = 'true profile', zorder=1, s =12)#,linewidth = 5)
ax1.plot(VMR_O3,height_values,marker = 'o',markerfacecolor = 'None', color = TrueCol , label = 'true profile', zorder=0 ,linewidth = 1.5, markersize =9)


for n in range(0,4):#,paraSamp,15):
    Sol = Results[n, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)
    ax1.plot(Sol,height_values,marker= '+',label = '$x_{'+str(n+1)+'}$', zorder = 1, linewidth = 0.5, markersize = 5)
    #color = ResCol
#line5 = ax1.plot(x_opt/(num_mole * S[ind,0] * f_broad * 1e-4 * scalingConst),height_values, color = 'crimson', linewidth = 7, label = 'reg. sol.', zorder=1)

ax1.set_xlabel(r'Ozone volume mixing ratio ')
#multicolor_ylabel(ax1,('(Tangent)','Height in km'),('k', dataCol),axis='y')
ax2.set_ylabel('(Tangent) Height in km')
handles, labels = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Handles = [handles[0], handles[1], handles[2]]
# Labels =  [labels[0], labels[1], labels[2]]
# LegendVertical(ax1, Handles, Labels, 90, XPad=-45, YPad=12)

#legend = ax1.legend(handles = [handles[-3], handles2[0], handles[0],handles[-2],handles[-1]], loc='lower right', framealpha = 0.2,fancybox=True)#, bbox_to_anchor=(1.01, 1.01), frameon =True)

#plt.ylabel('Height in km')
ax1.set_ylim([heights[minInd-1], heights[maxInd+1]])
#ax2.set_xlim([min(y),max(y)])
#ax1.set_xlim([min(x)-max(xerr)/2,max(x)+max(xerr)/2]) Ozone

ax2.set_xlabel(r'Spectral radiance in $\frac{\text{W } \text{cm}}{\text{m}^2 \text{ sr}} $',labelpad=10)# color =dataCol,
ax2.tick_params(colors = DatCol, axis = 'x')
ax2.xaxis.set_ticks_position('top')
ax2.xaxis.set_label_position('top')
ax1.xaxis.set_ticks_position('bottom')
ax1.xaxis.set_label_position('bottom')
ax1.spines[:].set_visible(False)
#ax2.spines['top'].set_color(pyTCol)
#ax1.legend(loc = 'upper right')

#plt.show()
#import tikzplotlib
#tikzplotlib_fix_ncols(fig)
tikzplotlib.save("NonLinRes.pgf")


##
# import plotly.graph_objects as go
#
# # Create figure
# fig = go.Figure()
# # k_values = int(np.linspace(0, len(As)-1, len(As)))
#
# # Add traces, one for each slider step
# for k in range(0, paraSamp):
#     y = height_values  # np.linspace(0, len(Au[:, k]) - 1, len(Au[:, k]))
#
#     x = Results[k, :] / (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)  # *As[k]
#     df = pd.DataFrame(dict(x=x, y=y))
#
#     fig.add_trace(
#         go.Scatter(
#             x=df['x'],
#             y=df['y'],
#             visible=False,
#             line=dict(color="#00CED1", width=6),
#             name=f"index = {k}"
#         )
#     )
#
# # Make 0th trace visible
# fig.data[0].visible = True
# k = np.linspace(0, paraSamp - 1, paraSamp)
#
# # Create and add slider
# steps = []
# for i in range(len(fig.data)):
#         method="update",
#         args=[{"visible": [False] * len(fig.data)},
#               {"title": "Slider at tangent model layer: " + str(i) + " in m"}],
#         label=str(i),  # layout attribute
#     )
#     step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
#     steps.append(step)
#
# sliders = [dict(
#     active=10,
#     currentvalue={"prefix": "index= ", "suffix": ""},
#     pad={"b": 50},
#     steps=steps
# )]
#
# fig.update_layout(
#     sliders=sliders,
#     title="Left Singlar Vectors weighted with Singular Values",
#     xaxis_title="height values"
# )
# fig.update_yaxes(range=[np.min(height_values), np.max(height_values)])
#
# fig.show()
#
# fig.write_html('Samples.html')


print('bla')



##
""" do MWG again with updated forward model"""
# fig3, ax2 = plt.subplots(figsize=set_size(245, fraction=fraction))
# Sol = np.mean(Results,0)/ (num_mole * S[ind, 0] * f_broad * 1e-4 * scalingConst)
#
# ax2.plot(Sol, height_values, marker='.', zorder=0, linewidth=0.5)
#
# fig3.show()
# ##
#
# new_w_cross = Sol * f_broad * 1e-4
# nonLin = calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, np.mean(num_mole * new_w_cross.reshape((SpecNumLayers,1)) * S[ind,0]) * np.ones((SpecNumLayers,1)), AscalConstKmToCm, SpecNumLayers, SpecNumMeas )
#
# A = A_lin * A_scal.T * nonLin
# ATA = np.matmul(A.T,A)
# ATy = np.matmul(A.T, y)
#
#
# Au, As, Avh = np.linalg.svd(A)
# cond_A =  np.max(A_lins)/np.min(As)
# print("normal: " + str(orderOfMagnitude(cond_A)))
#
# ATAu, ATAs, ATAvh = np.linalg.svd(ATA)
# cond_ATA = np.max(ATAs)/np.min(ATAs)
# print("Condition Number A^T A: " + str(orderOfMagnitude(cond_ATA)))
# #theta[0] = 0
# #theta[-1] = 0
# #Ax = np.matmul(A, theta)
#
# #convolve measurements and add noise
# #y = add_noise(Ax, 0.01)
# #y[y<=0] = 0
#
# #y = np.loadtxt('NonLinDataY.txt').reshape((SpecNumMeas,1))
#
#
# #ATy = np.matmul(A.T, y)
#
#
# ##
# """start the mtc algo with first guesses of noise and lumping const delta"""
#
#
# vari = np.zeros((len(theta)-2,1))
#
# for j in range(1,len(theta)-1):
#     vari[j-1] = np.var([theta[j-1],theta[j],theta[j+1]])
#
# #find minimum for first guesses
# '''params[1] = delta
# params[0] = gamma'''
# def MinLogMargPost(params):#, coeff):
#
#     # gamma = params[0]
#     # delta = params[1]
#     gamma = params[0]
#     lamb = params[1]
#     if lamb < 0  or gamma < 0:
#         return np.nan
#
#     n = SpecNumLayers
#     m = SpecNumMeas
#
#     Bp = ATA + lamb * L
#
#
#     B_inv_A_trans_y, exitCode = gmres(Bp, ATy[0::, 0], tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     G = g(A, L,  lamb)
#     F = f(ATy, y,  B_inv_A_trans_y)
#
#     return -n/2 * np.log(lamb) - (m/2 + 1) * np.log(gamma) + 0.5 * G + 0.5 * gamma * F +  ( betaD *  lamb * gamma + betaG *gamma)
#
# #minimum = optimize.fmin(MargPostU, [5e-5,0.5])
# minimum = optimize.fmin(MinLogMargPost, [1/(np.max(Ax) * 0.01)**2,1/(np.mean(vari))*(np.max(Ax) * 0.01)**2])
#
# lam0 = minimum[1]
# print(minimum)
#
#
# ##
# """ taylor series around lam_0 """
#
# B = (ATA + lam0 * L)
#
# B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# #print(exitCode)
#
# CheckB_inv_ATy = np.matmul(B, B_inv_A_trans_y0)
#
#
#
# B_inv_L0 = np.zeros(np.shape(B))
#
# for i in range(len(B)):
#     B_inv_L0[:, i], exitCode = gmres(B, L[:, i], tol=tol, restart=25)
#     if exitCode != 0:
#         print('B_inv_L ' + str(exitCode))
#
# #relative_tol_L = tol
# #CheckB_inv_L = np.matmul(B, B_inv_L)
# #print(np.linalg.norm(L- CheckB_inv_L)/np.linalg.norm(L)<relative_tol_L)
#
# B_inv_L_2 = np.matmul(B_inv_L0, B_inv_L0)
# B_inv_L_3 = np.matmul(B_inv_L_2, B_inv_L0)
# B_inv_L_4 = np.matmul(B_inv_L_2, B_inv_L_2)
# B_inv_L_5 = np.matmul(B_inv_L_4, B_inv_L0)
# B_inv_L_6 = np.matmul(B_inv_L_4, B_inv_L_2)
#
#
# f_0_1 = np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L0), B_inv_A_trans_y0)
# f_0_2 = -1 * np.matmul(np.matmul(ATy[0::, 0].T, B_inv_L_2), B_inv_A_trans_y0)
# f_0_3 = 1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_3) ,B_inv_A_trans_y0)
# f_0_4 = -1 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y0)
# #f_0_5 = 120 * np.matmul(np.matmul(ATy[0::, 0].T,B_inv_L_4) ,B_inv_A_trans_y)
#
#
# g_0_1 = np.trace(B_inv_L0)
# g_0_2 = -1 / 2 * np.trace(B_inv_L_2)
# g_0_3 = 1 /6 * np.trace(B_inv_L_3)
# g_0_4 = -1 /24 * np.trace(B_inv_L_4)
# g_0_5 = 0#1 /120 * np.trace(B_inv_L_5)
# g_0_6 = 0#1 /720 * np.trace(B_inv_L_6)
#
#
# ##
#
# '''do the sampling'''
# number_samples = 10000
#
#
# #inintialize sample
# gamma0 = minimum[0] #3.7e-5#1/np.var(y) * 1e1 #(0.01* np.max(Ax))1e-5#
# #0.275#1/(2*np.mean(vari))0.1#
# lambda0 = minimum[1]#deltas[0]/gammas[0]
# #deltas[0] =  minimum[1] * minimum[0]
#
# B = (ATA + lambda0 * L)
#
# B_inv_A_trans_y0, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
# if exitCode != 0:
#     print(exitCode)
#
# Bu, Bs, Bvh = np.linalg.svd(B)
# cond_B =  np.max(Bs)/np.min(Bs)
# print("Condition number B: " + str(orderOfMagnitude(cond_B)))
#
#
#
# #wgam = 1e-5
# #wdelt = 1e-1
#
# alphaG = 1
# alphaD = 1
# rate = f(ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0
# # draw gamma with a gibs step
# shape = SpecNumMeas/2 + alphaD + alphaG
#
# f_new = f(ATy, y,  B_inv_A_trans_y0)
# #g_old = g(A, L,  lambdas[0])
#
# def MHwG(number_samples, burnIn, lambda0, gamma0):
#     wLam = 0.7e3#7e1
#
#     alphaG = 1
#     alphaD = 1
#     k = 0
#
#     gammas = np.zeros(number_samples + burnIn)
#     #deltas = np.zeros(number_samples + burnIn)
#     lambdas = np.zeros(number_samples + burnIn)
#
#     gammas[0] = gamma0
#     lambdas[0] = lambda0
#
#     B = (ATA + lambda0 * L)
#     B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0=B_inv_A_trans_y0, tol=tol)
#
#     #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#     if exitCode != 0:
#         print(exitCode)
#
#     shape = SpecNumMeas / 2 + alphaD + alphaG
#     rate = f(ATy, y, B_inv_A_trans_y) / 2 + betaG + betaD * lambda0
#
#
#     for t in range(number_samples + burnIn-1):
#         #print(t)
#
#         # # draw new lambda
#         lam_p = normal(lambdas[t], wLam)
#
#         while lam_p < 0:
#                 lam_p = normal(lambdas[t], wLam)
#
#         delta_lam = lam_p - lambdas[t]
#         # B = (ATA + lam_p * L)
#         # B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#         # if exitCode != 0:
#         #     print(exitCode)
#
#
#         # f_new = f(ATy, y,  B_inv_A_trans_y)
#         # g_new = g(A, L,  lam_p)
#         #
#         # delta_f = f_new - f_old
#         # delta_g = g_new - g_old
#
#         delta_f = f_0_1 * delta_lam + f_0_2 * delta_lam**2 + f_0_3 * delta_lam**3
#         delta_g = g_0_1 * delta_lam + g_0_2 * delta_lam**2 + g_0_3 * delta_lam**3
#
#         log_MH_ratio = ((SpecNumLayers)/ 2) * (np.log(lam_p) - np.log(lambdas[t])) - 0.5 * (delta_g + gammas[t] * delta_f) - betaD * gammas[t] * delta_lam
#
#         #accept or rejeict new lam_p
#         u = uniform()
#         if np.log(u) <= log_MH_ratio:
#         #accept
#             k = k + 1
#             lambdas[t + 1] = lam_p
#             #only calc when lambda is updated
#
#             B = (ATA + lam_p * L)
#             B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], x0= B_inv_A_trans_y0,tol=tol, restart=25)
#             #B_inv_A_trans_y, exitCode = gmres(B, ATy[0::, 0], tol=tol, restart=25)
#
#             # if exitCode != 0:
#             #         print(exitCode)
#
#             f_new = f(ATy, y,  B_inv_A_trans_y)
#             #g_old = np.copy(g_new)
#             rate = f_new/2 + betaG + betaD * lam_p#lambdas[t+1]
#
#         else:
#             #rejcet
#             lambdas[t + 1] = np.copy(lambdas[t])
#
#
#
#
#         gammas[t+1] = np.random.gamma(shape = shape, scale = 1/rate)
#
#         #deltas[t+1] = lambdas[t+1] * gammas[t+1]
#
#     return lambdas, gammas,k
#
#
#
# startTime = time.time()
# sec_lambdas ,sec_gammas, sec_k = MHwG(number_samples, burnIn, lambda0, gamma0)
# elapsed = time.time() - startTime
# print('MTC Done in ' + str(elapsed) + ' s')
#
#
#
# print('acceptance ratio: ' + str(k/(number_samples+burnIn)))
# sec_deltas = sec_lambdas * sec_gammas
# np.savetxt('samples.txt', np.vstack((sec_gammas[burnIn::], sec_deltas[burnIn::], sec_lambdas[burnIn::])).T, header = 'gammas \t deltas \t lambdas \n Acceptance Ratio: ' + str(sec_k/number_samples) + '\n Elapsed Time: ' + str(elapsed), fmt = '%.15f \t %.15f \t %.15f')
#
# #delt_aav, delt_diff, delt_ddiff, delt_itau, delt_itau_diff, delt_itau_aav, delt_acorrn = uWerr(deltas, acorr=None, s_tau=1.5, fast_threshold=5000)
#
#
# eng = matlab.engine.start_matlab()
# eng.Run_Autocorr_Ana_MTC(nargout=0)
# eng.quit()
#
#
# AutoCorrData = np.loadtxt("auto_corr_dat.txt", skiprows=3, dtype='float')
# #IntAutoLam, IntAutoGam , IntAutoDelt = np.loadtxt("auto_corr_dat.txt",userow = 1, skiprows=1, dtype='float'
#
# with open("auto_corr_dat.txt") as fID:
#     for n, line in enumerate(fID):
#        if n == 1:
#             IntAutoDelt, IntAutoGam, IntAutoLam = [float(IAuto) for IAuto in line.split()]
#             break
#
#
#
# #refine according to autocorrelation time
# new_sec_lamb = sec_lambdas[burnIn::math.ceil(IntAutoLam)]
# #SetLambda = new_lamb[np.random.randint(low=0, high=len(new_lamb), size=1)]
# new_sec_gam = sec_gammas[burnIn::math.ceil(IntAutoGam)]
# #SetGamma = new_gam[np.random.randint(low = 0,high =len(new_gam),size =1)]
# new_sec_delt = sec_deltas[burnIn::math.ceil(IntAutoDelt)]
# #SetDelta = new_delt[np.random.randint(low = 0,high =len(new_delt),size =1)]
#
# fig, axs = plt.subplots(3, 1,tight_layout=True)
# # We can set the number of bins with the *bins* keyword argument.
# axs[0].hist(new_sec_gam,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoGam)))
# #axs[0].set_title(str(len(new_gam)) + ' effective $\gamma$ samples')
# axs[0].set_title(str(len(new_sec_gam)) + r' $\gamma$ samples, the noise precision')
# axs[1].hist(new_sec_delt,bins=n_bins, color = 'k')#int(n_bins/math.ceil(IntAutoDelt)))
# axs[1].set_title(str(len(new_sec_delt)) + ' $\delta$ samples, the prior precision')
# axs[2].hist(new_sec_lamb,bins=n_bins, color = 'k')#10)
# #axs[2].xaxis.set_major_formatter(scientific_formatter)
# #axs[2].set_title(str(len(new_lamb)) + ' effective $\lambda =\delta / \gamma$ samples')
# axs[2].set_title(str(len(new_sec_lamb)) + ' $\lambda$ samples, the regularization parameter')
# plt.savefig('sec_HistoResults.png')
# plt.show()
#
# print('bla')