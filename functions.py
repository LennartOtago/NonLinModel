import math
import numpy as np
from scipy.sparse.linalg import gmres
from numpy.random import uniform, normal
def orderOfMagnitude(number):
    return math.floor(math.log(number, 10))

'''Generate Forward Map A
where each collum is one measurment defined by a tangent height
every entry is length in km each measurement goes through
first non-zero entry of each row is the lowest layer (which should have a Ozone amount of 0)
last entry of each row is highest layer (also 0 Ozone)'''
def gen_sing_map(meas_ang, height, obs_height, R):
    tang_height = np.around((np.sin(meas_ang) * (obs_height + R)) - R, 2)
    num_meas = len(tang_height)
    # add one extra layer so that the difference in height can be calculated
    layers = np.zeros(len(height)+1)
    layers[0:-1] = height
    layers[-1] = height[-1] + (height[-1] - height[-2])/2


    A_height = np.zeros((num_meas, len(layers)-1))
    t = 0
    for m in range(0, num_meas):

        while layers[t] <= tang_height[m]:

            t += 1

        # first dr
        # 0.5 cause of trapezoidal rule only for last
        A_height[m, t - 1] =  np.sqrt((layers[t] + R) ** 2 - (tang_height[m] + R) ** 2)
        dr = 2 * A_height[m, t - 1]
        for i in range(t, len(layers) - 1):
            A_height[m, i] = np.sqrt((layers[i + 1] + R) ** 2 - (tang_height[m] + R) ** 2) - dr
            dr = dr + A_height[m, i]
        A_height[m, i] = 0.5 * A_height[m, i]
    return A_height, tang_height, layers[-1]


def generate_L(neigbours):
    #Dirichlet Boundaries
    siz = int(np.size(neigbours, 0))
    neig = np.size(neigbours, 1)
    L = np.zeros((siz, siz))

    for i in range(0, siz):
        L[i, i] = neig
        for j in range(0, neig):
            if ~np.isnan(neigbours[i, j]):
                L[i, int(neigbours[i, j])] = -1
    #non periodic boundaries Neumann
    # L[0,0] = 1
    # L[-1,-1] = 1
    #periodic boundaires
    # L[0,-1] = -1
    # L[-1,0] = -1

    return L

def get_temp_values(height_values):
    """ used to be based on the ISA model see omnicalculator.com/physics/altitude-temperature
    now https://www.grc.nasa.gov/www/k-12/airplane/atmosmet.html """
    temp_values = np.zeros(len(height_values))
    #temp_values[0] = 288.15#15 - (height_values[0] - 0) * 6.49 + 273.15
    ###calculate temp values
    for i in range(0, len(height_values)):
        if height_values[i] < 11:
            temp_values[i] = np.around(15.04 - height_values[i] * 6.49 + 273.15,2)
        if 11 <= height_values[i] < 25:
            temp_values[i] = np.around(-55.46 + 273.15,2)
        if 25 <= height_values[i] :
            temp_values[i] = np.around(-131.21 + height_values[i] * 2.99 + 273.15,2)

    return temp_values.reshape((len(height_values),1))


# def add_noise(Ax, percent):
#     return Ax + np.random.normal(0, percent * np.max(Ax), (len(Ax), 1))

def add_noise(signal, snr):
    """
    Add noise to a signal based on the specified SNR (in percent).

    Parameters:
        signal: numpy array
            The original signal.
        snr_percent: float
            The desired signal-to-noise ratio in percent.

    Returns:
        numpy array
            The signal with added noise.
    """
    # Calculate root mean square of signal
    signal_power = np.sqrt(np.mean(np.abs(signal) ** 2))

    # Calculate noise power based on SNR (in percent)
    noise_power = signal_power / snr

    # Generate noise
    noise = np.random.normal(0, noise_power, signal.shape)

    # Add noise to the signal
    noisy_signal = signal + noise

    return noisy_signal, 1/noise_power**2

def calcNonLin(A_lin, pressure_values, LineIntScal, temp_values, theta, AscalConstKmToCm, SpecNumLayers, SpecNumMeas ):

    ConcVal = - pressure_values.reshape(
        (SpecNumLayers, 1)) * 1e2 * LineIntScal / temp_values * theta *AscalConstKmToCm


    mask = A_lin * np.ones((SpecNumMeas, SpecNumLayers))
    mask[mask != 0] = 1

    ConcValMat = np.tril(np.ones(len(ConcVal))) * ConcVal
    ValPerLayPre = mask * (A_lin @ ConcValMat)
    preTrans = np.exp(ValPerLayPre)

    ConcValMatAft = np.triu(np.ones(len(ConcVal))) * ConcVal
    ValPerLayAft = mask * (A_lin @ ConcValMatAft)
    BasPreTrans = (A_lin @ ConcValMat)[0::, 0].reshape((SpecNumMeas, 1)) @ np.ones((1, SpecNumLayers))
    afterTrans = np.exp( (BasPreTrans + ValPerLayAft) * mask )


    return preTrans + afterTrans



def g(A, L, l):
    """ calculate g"""
    B = np.matmul(A.T,A) + l * L
    Bu, Bs, Bvh = np.linalg.svd(B)
    # np.log(np.prod(Bs))
    return np.sum(np.log(Bs))

def f(ATy, y, B_inv_A_trans_y):
    return np.matmul(y[0::,0].T, y[0::,0]) - np.matmul(ATy[0::,0].T,B_inv_A_trans_y)


def g_tayl(delta_lam, g_0, trace_B_inv_L_1, trace_B_inv_L_2, trace_B_inv_L_3, trace_B_inv_L_4, trace_B_inv_L_5, trace_B_inv_L_6):
    return g_0 + trace_B_inv_L_1 * delta_lam + trace_B_inv_L_2 * delta_lam**2 + trace_B_inv_L_3 * delta_lam**3 + trace_B_inv_L_4 * delta_lam**4 + trace_B_inv_L_5 * delta_lam**5 + trace_B_inv_L_6 * delta_lam**6


def f_tayl( delta_lam, f_0, f_1, f_2, f_3, f_4):
    return f_0 + f_1 * delta_lam + f_2 * delta_lam**2 + f_3 * delta_lam**3 + f_4 * delta_lam**4# + f_5 * delta_lam**5 #- f_6 * delta_lam**6


def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = 1#(5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)



def MHwG(number_samples, burnIn, y, L, ATA, SpecNumMeas, SpecNumLayers, B_inv_A_trans_y0, ATy, wLam, lambda0, gamma0, tol, f_0_1, f_0_2, f_0_3, g_0_1, g_0_2, g_0_3):




    alphaG = 1
    alphaD = 1
    betaG = 1e-4
    betaD = 1e-10
    k = 0

    gammas = np.zeros(number_samples + burnIn)
    #deltas = np.zeros(number_samples + burnIn)
    lambdas = np.zeros(number_samples + burnIn)

    gammas[0] = gamma0
    lambdas[0] = lambda0


    shape = SpecNumMeas / 2 + alphaD + alphaG
    rate = f(ATy, y, B_inv_A_trans_y0) / 2 + betaG + betaD * lambda0


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